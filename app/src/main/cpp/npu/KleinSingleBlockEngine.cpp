#include "npu/KleinSingleBlockEngine.hpp"

#include <android/log.h>
#include <sys/stat.h>

#include <chrono>
#include <algorithm>
#include <cstring>
#include <thread>

#define NPU_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "PocketTavernDiffusion", __VA_ARGS__)

#include "litert/c/litert_any.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace pockettavern {

namespace {

constexpr int kTokenCount = 4608;
constexpr int kChunk = 1152;
constexpr int kNumChunks = kTokenCount / kChunk;  // 4
constexpr int kHiddenSize = 3072;
constexpr int kRopeComplexPairs = 64;
constexpr int kHeads = 24;
constexpr int kHeadDim = 128;
constexpr int kPeStride = kRopeComplexPairs * 2 * 2;  // 256
constexpr int kMlpHiddenDim = 18432;

bool FileExists(const std::string& path) {
  struct stat st{};
  return ::stat(path.c_str(), &st) == 0;
}

size_t ElementCount(const std::vector<int32_t>& shape) {
  size_t n = 1;
  for (int32_t d : shape) n *= static_cast<size_t>(d);
  return n;
}

LiteRtRankedTensorType MakeFloat32TensorType(const std::vector<int32_t>& shape) {
  LiteRtRankedTensorType t{};
  t.element_type = kLiteRtElementTypeFloat32;
  t.layout.rank = static_cast<unsigned int>(shape.size());
  t.layout.has_strides = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    t.layout.dimensions[i] = shape[i];
  }
  return t;
}

struct ModelHolder {
  LiteRtModel model = nullptr;
  ~ModelHolder() {
    if (model) LiteRtDestroyModel(model);
  }
};

struct OptionsHolder {
  LiteRtOptions options = nullptr;
  ~OptionsHolder() {
    if (options) LiteRtDestroyOptions(options);
  }
};

struct CompiledModelHolder {
  LiteRtCompiledModel compiled_model = nullptr;
  ~CompiledModelHolder() {
    if (compiled_model) LiteRtDestroyCompiledModel(compiled_model);
  }
};

struct TensorBufferHolder {
  LiteRtTensorBuffer buffer = nullptr;
  ~TensorBufferHolder() {
    if (buffer) LiteRtDestroyTensorBuffer(buffer);
  }
};

struct BufferSet {
  std::vector<TensorBufferHolder> holders;
  std::vector<LiteRtTensorBuffer> buffers;
  std::vector<size_t> sizes;
};

std::vector<float> Slice(const std::vector<float>& src, size_t offset, size_t count) {
  return std::vector<float>(src.begin() + static_cast<long>(offset),
                             src.begin() + static_cast<long>(offset + count));
}

}  // namespace

struct KleinSingleBlockEngine::CachedPiece {
  ModelHolder model;
  OptionsHolder options;
  CompiledModelHolder compiled;
  std::vector<TensorBufferHolder> input_holders;
  std::vector<TensorBufferHolder> output_holders;
  std::vector<LiteRtTensorBuffer> input_buffers;
  std::vector<LiteRtTensorBuffer> output_buffers;
  std::vector<size_t> output_sizes;
  std::vector<std::vector<int32_t>> input_shapes;
  std::vector<std::vector<int32_t>> output_shapes;

  ~CachedPiece() {
    // Destroy this before its DmaBufs: the Tensor dispatcher retains epoll state through the
    // compiled model and is unsafe if buffers are torn down first.
    if (compiled.compiled_model) {
      LiteRtDestroyCompiledModel(compiled.compiled_model);
      compiled.compiled_model = nullptr;
    }
  }
};

KleinSingleBlockEngine::KleinSingleBlockEngine() = default;

KleinSingleBlockEngine::~KleinSingleBlockEngine() {
  cached_pieces_.clear();
  if (env_) {
    LiteRtDestroyEnvironment(env_);
    env_ = nullptr;
  }
}

void KleinSingleBlockEngine::ReleaseCachedTensorBuffers() {
  for (auto& entry : cached_pieces_) {
    CachedPiece* piece = entry.second.get();
    piece->input_buffers.clear();
    piece->output_buffers.clear();
    piece->output_sizes.clear();
    piece->input_holders.clear();
    piece->output_holders.clear();
  }
}

bool KleinSingleBlockEngine::Load(std::string model_dir, const std::string& dispatch_lib_dir,
                                  int block_index) {
  if (env_ != nullptr) {
    NPU_LOGE("KleinSingleBlockEngine::Load called twice on the same instance\n");
    return false;
  }
  model_dir_ = std::move(model_dir);
  if (block_index < 0 || block_index >= 20) {
    NPU_LOGE("KleinSingleBlockEngine::Load: invalid block index %d\n", block_index);
    return false;
  }
  // Block zero predates namespacing; later exports include the block number to retain their
  // learned weights alongside it in one model directory.
  const std::string prefix = block_index == 0 ? "" : "single" + std::to_string(block_index) + "_";
  qkv_file_ = prefix + "qkv_proj_probe_1152_Google_Tensor_G5.tflite";
  out_file_ = prefix + "out_proj_probe_1152_Google_Tensor_G5.tflite";

  const std::string kRequiredFiles[] = {
      qkv_file_,
      "flash_step_probe_1152_Google_Tensor_G5.tflite",
      "flash_step_init_probe_Google_Tensor_G5.tflite",
      out_file_,
      "attn_finalize_probe_1152_Google_Tensor_G5.tflite",
  };
  for (const std::string& name : kRequiredFiles) {
    std::string path = model_dir_ + "/" + name;
    if (!FileExists(path)) {
      NPU_LOGE("KleinSingleBlockEngine::Load: missing piece file %s\n", path.c_str());
      return false;
    }
  }

  LiteRtAny dispatch_dir_value{};
  dispatch_dir_value.type = kLiteRtAnyTypeString;
  dispatch_dir_value.str_value = dispatch_lib_dir.c_str();
  LiteRtEnvOption option{};
  option.tag = kLiteRtEnvOptionTagDispatchLibraryDir;
  option.value = dispatch_dir_value;

  LiteRtStatus status = LiteRtCreateEnvironment(1, &option, &env_);
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("KleinSingleBlockEngine::Load: LiteRtCreateEnvironment failed, status=%d\n", status);
    env_ = nullptr;
    return false;
  }
  return true;
}

bool KleinSingleBlockEngine::RunPiece(const std::string& file_name,
                                       const std::vector<const std::vector<float>*>& inputs,
                                       const std::vector<std::vector<int32_t>>& input_shapes,
                                       const std::vector<std::vector<int32_t>>& output_shapes,
                                       std::vector<std::vector<float>>* out_results) {
  const std::string path = model_dir_ + "/" + file_name;
  const auto t_start = std::chrono::steady_clock::now();
  const auto elapsed_ms = [&t_start]() -> long long {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t_start)
        .count();
  };

  // See NpuUnetEngine::RunPiece for why these are declared before cm_holder (destruction-order
  // safety for the Google Tensor dispatcher's epoll state) even though we destroy cm_holder
  // explicitly below anyway.
  std::vector<TensorBufferHolder> input_holders(inputs.size());
  std::vector<LiteRtTensorBuffer> input_buffers(inputs.size());
  std::vector<TensorBufferHolder> output_holders(output_shapes.size());
  std::vector<LiteRtTensorBuffer> output_buffers(output_shapes.size());
  std::vector<size_t> output_element_counts(output_shapes.size());

  CachedPiece* piece = nullptr;
  auto cached = cached_pieces_.find(file_name);
  LiteRtStatus status = kLiteRtStatusOk;
  if (cached == cached_pieces_.end()) {
    std::unique_ptr<CachedPiece> created(new CachedPiece());
    status = LiteRtCreateModelFromFile(env_, path.c_str(), &created->model.model);
    if (status != kLiteRtStatusOk) return false;
    if (LiteRtCreateOptions(&created->options.options) != kLiteRtStatusOk) return false;
  // Confirmed empirically (2026-08-29): NPU-only rejects flash_step_probe_1152 with status 504 --
  // the exact same failure mode NpuUnetEngine.hpp documents for its RESHAPE-wrapped SDXL pieces
  // ("needs CPU available for the wrapper op"). Klein's pieces aren't RESHAPE-wrapped the same
  // way, but apparently still need a CPU fallback available for some op (likely one of the
  // internal reshape/collapse-to-3D ops FlashStep uses) -- so NPU|CPU it is, matching the
  // Kotlin/Options(Accelerator.NPU) path's *effective* behavior (that wrapper evidently allows
  // CPU fallback transparently) despite naming only NPU.
  status = LiteRtSetOptionsHardwareAccelerators(created->options.options,
                                                 kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu);
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("KleinSingleBlockEngine: %s: LiteRtSetOptionsHardwareAccelerators failed, status=%d\n",
              file_name.c_str(), status);
    return false;
  }
  status = LiteRtCreateCompiledModel(env_, created->model.model, created->options.options, &created->compiled.compiled_model);
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("KleinSingleBlockEngine: %s: LiteRtCreateCompiledModel failed, status=%d\n", file_name.c_str(), status);
    return false;
  }
    piece = created.get();
    cached_pieces_.emplace(file_name, std::move(created));
  } else {
    piece = cached->second.get();
  }
  LiteRtCompiledModel compiled_model = piece->compiled.compiled_model;
  const long long model_ms = 0;
  const long long compile_ms = elapsed_ms();

  for (size_t i = 0; i < inputs.size(); ++i) {
    const size_t num_elements = ElementCount(input_shapes[i]);
    if (inputs[i]->size() != num_elements) {
      NPU_LOGE("KleinSingleBlockEngine: %s: input %zu expected %zu elements, got %zu\n",
                file_name.c_str(), i, num_elements, inputs[i]->size());
      return false;
    }

    LiteRtTensorBufferRequirements requirements = nullptr;
    status = LiteRtGetCompiledModelInputBufferRequirements(compiled_model, /*signature_index=*/0,
                                                             /*input_index=*/i, &requirements);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("KleinSingleBlockEngine: %s: input buffer requirements (input %zu) failed, status=%d\n",
                file_name.c_str(), i, status);
      return false;
    }

    LiteRtRankedTensorType tensor_type = MakeFloat32TensorType(input_shapes[i]);
    status = LiteRtCreateManagedTensorBufferFromRequirements(env_, &tensor_type, requirements,
                                                               &input_holders[i].buffer);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("KleinSingleBlockEngine: %s: create input buffer (input %zu) failed, status=%d\n",
                file_name.c_str(), i, status);
      return false;
    }

    void* host_mem = nullptr;
    status = LiteRtLockTensorBuffer(input_holders[i].buffer, &host_mem, kLiteRtTensorBufferLockModeWrite);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("KleinSingleBlockEngine: %s: lock input buffer (input %zu) failed, status=%d\n",
                file_name.c_str(), i, status);
      return false;
    }
    std::memcpy(host_mem, inputs[i]->data(), num_elements * sizeof(float));
    LiteRtUnlockTensorBuffer(input_holders[i].buffer);
    input_buffers[i] = input_holders[i].buffer;
  }

  for (size_t i = 0; i < output_shapes.size(); ++i) {
    const size_t num_elements = ElementCount(output_shapes[i]);
    output_element_counts[i] = num_elements;

    LiteRtTensorBufferRequirements requirements = nullptr;
    status = LiteRtGetCompiledModelOutputBufferRequirements(compiled_model, /*signature_index=*/0,
                                                              /*output_index=*/i, &requirements);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("KleinSingleBlockEngine: %s: output buffer requirements (output %zu) failed, status=%d\n",
                file_name.c_str(), i, status);
      return false;
    }

    // Diagnostic-only: log every buffer type this piece's requirements actually support, and
    // which one gets chosen, to investigate the "read" phase's disproportionate cost (a possible
    // DMA-buf cache-sync cost on lock, vs. a plain host-memory buffer that wouldn't need one).
    int num_types = 0;
    LiteRtGetNumTensorBufferRequirementsSupportedBufferTypes(requirements, &num_types);
    std::string supported_types_str;
    for (int t = 0; t < num_types; ++t) {
      LiteRtTensorBufferType supported_type;
      LiteRtGetTensorBufferRequirementsSupportedTensorBufferType(requirements, t, &supported_type);
      supported_types_str += std::to_string(static_cast<int>(supported_type)) + " ";
    }

    LiteRtRankedTensorType tensor_type = MakeFloat32TensorType(output_shapes[i]);
    // Diagnostic experiment: LiteRtCreateManagedTensorBufferFromRequirements auto-picks Ahwb
    // (type 2) from the supported set [Ahwb, DmaBuf] every time, and the resulting
    // Lock(read)+memcpy+Unlock cost dominates total time (~66%, ~20x the Kotlin-profiled
    // equivalent). Force DmaBuf (type 4) explicitly instead, to see whether its CPU-read path is
    // cheaper on this hardware -- Ahwb typically requires a real cache-sync on CPU lock, which
    // DmaBuf's ION-style mapping may not.
    size_t requirements_buffer_size = 0;
    LiteRtGetTensorBufferRequirementsBufferSize(requirements, &requirements_buffer_size);
    status = LiteRtCreateManagedTensorBuffer(env_, kLiteRtTensorBufferTypeDmaBuf, &tensor_type,
                                              requirements_buffer_size, &output_holders[i].buffer);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("KleinSingleBlockEngine: %s: create output buffer (output %zu) failed, status=%d\n",
                file_name.c_str(), i, status);
      return false;
    }
    LiteRtTensorBufferType chosen_type = kLiteRtTensorBufferTypeUnknown;
    LiteRtGetTensorBufferType(output_holders[i].buffer, &chosen_type);
    NPU_LOGE("KleinSingleBlockEngine: %s: output %zu (%zu bytes) supported_types=[%s] chosen_type=%d\n",
              file_name.c_str(), i, num_elements * sizeof(float), supported_types_str.c_str(),
              static_cast<int>(chosen_type));
    output_buffers[i] = output_holders[i].buffer;
  }
  const long long buffers_ms = elapsed_ms();

  status = LiteRtRunCompiledModel(compiled_model, /*signature_index=*/0, input_buffers.size(),
                                   input_buffers.data(), output_buffers.size(), output_buffers.data());
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("KleinSingleBlockEngine: %s: LiteRtRunCompiledModel failed, status=%d\n", file_name.c_str(), status);
    return false;
  }
  const long long run_ms = elapsed_ms();

  out_results->clear();
  out_results->resize(output_shapes.size());
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    void* host_mem = nullptr;
    status = LiteRtLockTensorBuffer(output_holders[i].buffer, &host_mem, kLiteRtTensorBufferLockModeRead);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("KleinSingleBlockEngine: %s: lock output buffer (output %zu) failed, status=%d\n",
                file_name.c_str(), i, status);
      return false;
    }
    (*out_results)[i].resize(output_element_counts[i]);
    std::memcpy((*out_results)[i].data(), host_mem, output_element_counts[i] * sizeof(float));
    LiteRtUnlockTensorBuffer(output_holders[i].buffer);
  }
  const long long read_ms = elapsed_ms();

  NPU_LOGE("KleinSingleBlockEngine: phases %s: model=%lldms compile=%lldms buffers=%lldms run=%lldms read=%lldms total=%lldms\n",
           file_name.c_str(), model_ms, compile_ms - model_ms, buffers_ms - compile_ms, run_ms - buffers_ms,
           read_ms - run_ms, read_ms);

  return true;
}

bool KleinSingleBlockEngine::RunCachedDirect(
    const std::string& cache_key, const std::vector<LiteRtTensorBuffer>& input_buffers,
    const std::vector<LiteRtTensorBuffer>* output_buffers) {
  const auto found = cached_pieces_.find(cache_key);
  if (found == cached_pieces_.end()) return false;
  CachedPiece* piece = found->second.get();
  if (input_buffers.size() != piece->input_buffers.size()) return false;
  std::vector<LiteRtTensorBuffer> mutable_inputs = input_buffers;
  std::vector<LiteRtTensorBuffer> mutable_outputs =
      output_buffers ? *output_buffers : piece->output_buffers;
  if (mutable_outputs.size() != piece->output_buffers.size()) return false;
  const LiteRtStatus status = LiteRtRunCompiledModel(
      piece->compiled.compiled_model, 0, mutable_inputs.size(), mutable_inputs.data(),
      mutable_outputs.size(), mutable_outputs.data());
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("KleinSingleBlockEngine direct %s: run failed %d\n", cache_key.c_str(), status);
    return false;
  }
  return true;
}

KleinSingleBlockEngine::CachedPiece* KleinSingleBlockEngine::PrepareCachedPiece(
    const std::string& file_name, const std::vector<std::vector<int32_t>>& input_shapes,
    const std::vector<std::vector<int32_t>>& output_shapes, const std::string& cache_key) {
  const std::string& key = cache_key.empty() ? file_name : cache_key;
  const auto allocate_buffers = [&](CachedPiece* piece) -> bool {
    if (!piece->input_holders.empty()) return true;
    piece->input_holders.resize(input_shapes.size());
    piece->input_buffers.resize(input_shapes.size());
    piece->output_holders.resize(output_shapes.size());
    piece->output_buffers.resize(output_shapes.size());
    piece->output_sizes.resize(output_shapes.size());
    for (size_t i = 0; i < input_shapes.size(); ++i) {
      LiteRtTensorBufferRequirements requirements = nullptr;
      if (LiteRtGetCompiledModelInputBufferRequirements(piece->compiled.compiled_model, 0, i,
                                                        &requirements) != kLiteRtStatusOk) {
        return false;
      }
      size_t bytes = 0;
      LiteRtGetTensorBufferRequirementsBufferSize(requirements, &bytes);
      LiteRtRankedTensorType type = MakeFloat32TensorType(input_shapes[i]);
      if (LiteRtCreateManagedTensorBuffer(env_, kLiteRtTensorBufferTypeDmaBuf, &type, bytes,
                                          &piece->input_holders[i].buffer) != kLiteRtStatusOk) {
        return false;
      }
      piece->input_buffers[i] = piece->input_holders[i].buffer;
    }
    for (size_t i = 0; i < output_shapes.size(); ++i) {
      LiteRtTensorBufferRequirements requirements = nullptr;
      if (LiteRtGetCompiledModelOutputBufferRequirements(piece->compiled.compiled_model, 0, i,
                                                         &requirements) != kLiteRtStatusOk) {
        return false;
      }
      size_t bytes = 0;
      LiteRtGetTensorBufferRequirementsBufferSize(requirements, &bytes);
      LiteRtRankedTensorType type = MakeFloat32TensorType(output_shapes[i]);
      if (LiteRtCreateManagedTensorBuffer(env_, kLiteRtTensorBufferTypeDmaBuf, &type, bytes,
                                          &piece->output_holders[i].buffer) != kLiteRtStatusOk) {
        return false;
      }
      piece->output_buffers[i] = piece->output_holders[i].buffer;
      piece->output_sizes[i] = ElementCount(output_shapes[i]);
    }
    return true;
  };
  const auto found = cached_pieces_.find(key);
  if (found != cached_pieces_.end()) {
    CachedPiece* piece = found->second.get();
    if (piece->input_shapes != input_shapes || piece->output_shapes != output_shapes) return nullptr;
    return allocate_buffers(piece) ? piece : nullptr;
  }

  std::unique_ptr<CachedPiece> created(new CachedPiece());
  LiteRtStatus status = LiteRtCreateModelFromFile(
      env_, (model_dir_ + "/" + file_name).c_str(), &created->model.model);
  if (status != kLiteRtStatusOk || LiteRtCreateOptions(&created->options.options) != kLiteRtStatusOk ||
      LiteRtSetOptionsHardwareAccelerators(created->options.options,
                                            kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu) !=
          kLiteRtStatusOk ||
      LiteRtCreateCompiledModel(env_, created->model.model, created->options.options,
                                &created->compiled.compiled_model) != kLiteRtStatusOk) {
    return nullptr;
  }
  created->input_shapes = input_shapes;
  created->output_shapes = output_shapes;
  if (!allocate_buffers(created.get())) return nullptr;
  CachedPiece* result = created.get();
  cached_pieces_.emplace(key, std::move(created));
  return result;
}

bool KleinSingleBlockEngine::forward(const std::vector<float>& x, const std::vector<float>& pe,
                                       const std::vector<float>& mod_shift, const std::vector<float>& mod_scale,
                                       const std::vector<float>& mod_gate, std::vector<float>* out) {
  if (!IsLoaded()) {
    NPU_LOGE("KleinSingleBlockEngine::forward called before a successful Load()\n");
    return false;
  }
  if (x.size() != static_cast<size_t>(kTokenCount) * kHiddenSize ||
      pe.size() != static_cast<size_t>(kTokenCount) * kPeStride ||
      mod_shift.size() != kHiddenSize || mod_scale.size() != kHiddenSize || mod_gate.size() != kHiddenSize) {
    NPU_LOGE("KleinSingleBlockEngine::forward: unexpected input size(s)\n");
    return false;
  }

  const std::vector<int32_t> x_chunk_shape = {1, kChunk, kHiddenSize};
  const std::vector<int32_t> pe_chunk_shape = {1, 1, kChunk, kRopeComplexPairs, 2, 2};
  const std::vector<int32_t> mod_vec_shape = {1, 1, kHiddenSize};
  const std::vector<int32_t> qkv_shape = {1, kHeads, kChunk, kHeadDim};
  const std::vector<int32_t> mlp_shape = {1, kChunk, kMlpHiddenDim};
  const std::vector<int32_t> state_max_sum_shape = {1, kHeads, kChunk, 1};

  // Pass 1: QKV projection, one dispatch per query-chunk. q/k/v/mlp all kept as plain
  // std::vector<float> (native heap) -- no disk I/O anywhere, unlike the Kotlin diagnostic which
  // had to keep mlp on disk to fit the Dalvik-managed-heap ceiling (a JVM-specific constraint,
  // not a real one here -- see docs/flux2-klein-conversion.md's correction note).
  std::vector<float> q_chunks[kNumChunks];
  std::vector<float> k_chunks[kNumChunks];
  std::vector<float> v_chunks[kNumChunks];
  std::vector<float> mlp_chunks[kNumChunks];

  for (int c = 0; c < kNumChunks; ++c) {
    std::vector<float> x_chunk = Slice(x, static_cast<size_t>(c) * kChunk * kHiddenSize, kChunk * kHiddenSize);
    std::vector<float> pe_chunk = Slice(pe, static_cast<size_t>(c) * kChunk * kPeStride, kChunk * kPeStride);

    std::vector<std::vector<float>> result;
    bool ok = RunPiece(qkv_file_,
                        {&x_chunk, &pe_chunk, &mod_shift, &mod_scale},
                        {x_chunk_shape, pe_chunk_shape, mod_vec_shape, mod_vec_shape},
                        {qkv_shape, qkv_shape, qkv_shape, mlp_shape}, &result);
    if (!ok) {
      NPU_LOGE("KleinSingleBlockEngine::forward: qkv_proj chunk %d failed\n", c);
      return false;
    }
    q_chunks[c] = std::move(result[0]);
    k_chunks[c] = std::move(result[1]);
    v_chunks[c] = std::move(result[2]);
    mlp_chunks[c] = std::move(result[3]);
  }

  // Pass 2: flash attention -- one dispatch per (Q-chunk, K/V-chunk) pair, running-softmax state
  // threaded through host-side. ki=0 uses flash_step_init (no external running-state input --
  // avoids the sentinel-state AOT-compiler bug documented in flush_step_init_export.py/the doc);
  // flash_step (with the max()/correction merge) is used for ki=1..N-1.
  std::vector<float> attn_chunks[kNumChunks];
  for (int qi = 0; qi < kNumChunks; ++qi) {
    std::vector<std::vector<float>> init_result;
    bool ok = RunPiece("flash_step_init_probe_Google_Tensor_G5.tflite",
                        {&q_chunks[qi], &k_chunks[0], &v_chunks[0]},
                        {qkv_shape, qkv_shape, qkv_shape},
                        {state_max_sum_shape, state_max_sum_shape, qkv_shape}, &init_result);
    if (!ok) {
      NPU_LOGE("KleinSingleBlockEngine::forward: flash_step_init qi=%d failed\n", qi);
      return false;
    }
    std::vector<float> running_max = std::move(init_result[0]);
    std::vector<float> running_sum = std::move(init_result[1]);
    std::vector<float> running_out = std::move(init_result[2]);

    for (int ki = 1; ki < kNumChunks; ++ki) {
      std::vector<std::vector<float>> result;
      ok = RunPiece("flash_step_probe_1152_Google_Tensor_G5.tflite",
                    {&q_chunks[qi], &k_chunks[ki], &v_chunks[ki], &running_max, &running_sum, &running_out},
                    {qkv_shape, qkv_shape, qkv_shape, state_max_sum_shape, state_max_sum_shape, qkv_shape},
                    {state_max_sum_shape, state_max_sum_shape, qkv_shape}, &result);
      if (!ok) {
        NPU_LOGE("KleinSingleBlockEngine::forward: flash_step qi=%d ki=%d failed\n", qi, ki);
        return false;
      }
      running_max = std::move(result[0]);
      running_sum = std::move(result[1]);
      running_out = std::move(result[2]);
    }

    // Normalize (host-side, cheap) then transpose+reshape [1,H,chunk,D] -> [1,chunk,H*D] to match
    // _out's expected attn layout, exactly as runKleinChunkedBlockProbe's Kotlin version does.
    std::vector<float> attn(static_cast<size_t>(kChunk) * kHiddenSize);
    for (int h = 0; h < kHeads; ++h) {
      for (int t = 0; t < kChunk; ++t) {
        const float denom = running_sum[static_cast<size_t>(h) * kChunk + t];
        for (int d = 0; d < kHeadDim; ++d) {
          attn[static_cast<size_t>(t) * kHiddenSize + h * kHeadDim + d] =
              running_out[(static_cast<size_t>(h) * kChunk + t) * kHeadDim + d] / denom;
        }
      }
    }
    attn_chunks[qi] = std::move(attn);

    // q for this chunk is only ever needed within its own iteration; release it now. k/v stay
    // resident until the whole pass 2 loop finishes (every qi needs every ki).
    std::vector<float>().swap(q_chunks[qi]);
  }
  for (int i = 0; i < kNumChunks; ++i) {
    std::vector<float>().swap(k_chunks[i]);
    std::vector<float>().swap(v_chunks[i]);
  }

  // Pass 3: output projection, one dispatch per query-chunk.
  out->clear();
  out->resize(static_cast<size_t>(kTokenCount) * kHiddenSize);
  for (int c = 0; c < kNumChunks; ++c) {
    std::vector<float> x_chunk = Slice(x, static_cast<size_t>(c) * kChunk * kHiddenSize, kChunk * kHiddenSize);

    std::vector<std::vector<float>> result;
    bool ok = RunPiece(out_file_,
                        {&x_chunk, &attn_chunks[c], &mlp_chunks[c], &mod_gate},
                        {x_chunk_shape, x_chunk_shape, mlp_shape, mod_vec_shape}, {x_chunk_shape}, &result);
    if (!ok) {
      NPU_LOGE("KleinSingleBlockEngine::forward: out_proj chunk %d failed\n", c);
      return false;
    }
    std::memcpy(out->data() + static_cast<size_t>(c) * kChunk * kHiddenSize, result[0].data(),
                result[0].size() * sizeof(float));
  }

  return true;
}

bool KleinSingleBlockEngine::forwardZeroCopyPooled(
    const std::vector<float>& x, const std::vector<float>& pe,
    const std::vector<float>& mod_shift, const std::vector<float>& mod_scale,
    const std::vector<float>& mod_gate, std::vector<float>* out) {
  return forwardZeroCopyPooledImpl(x, pe, mod_shift, mod_scale, mod_gate, out,
                                   /*parallel_qkv=*/false, /*parallel_attention=*/false,
                                   /*parallel_output=*/false,
                                   /*probe_interblock=*/false, /*worker_count=*/1);
}

bool KleinSingleBlockEngine::forwardZeroCopyPooledParallelQkv(
    const std::vector<float>& x, const std::vector<float>& pe,
    const std::vector<float>& mod_shift, const std::vector<float>& mod_scale,
    const std::vector<float>& mod_gate, std::vector<float>* out) {
  return forwardZeroCopyPooledImpl(x, pe, mod_shift, mod_scale, mod_gate, out,
                                   /*parallel_qkv=*/true, /*parallel_attention=*/false,
                                   /*parallel_output=*/false,
                                   /*probe_interblock=*/false, /*worker_count=*/4);
}

bool KleinSingleBlockEngine::forwardZeroCopyPooledParallelQkvAttention(
    const std::vector<float>& x, const std::vector<float>& pe,
    const std::vector<float>& mod_shift, const std::vector<float>& mod_scale,
    const std::vector<float>& mod_gate, std::vector<float>* out) {
  return forwardZeroCopyPooledImpl(x, pe, mod_shift, mod_scale, mod_gate, out,
                                   /*parallel_qkv=*/true, /*parallel_attention=*/true,
                                   /*parallel_output=*/false,
                                   /*probe_interblock=*/false, /*worker_count=*/4);
}

bool KleinSingleBlockEngine::forwardZeroCopyPooledFullyParallel(
    const std::vector<float>& x, const std::vector<float>& pe,
    const std::vector<float>& mod_shift, const std::vector<float>& mod_scale,
    const std::vector<float>& mod_gate, std::vector<float>* out) {
  return forwardZeroCopyPooledImpl(x, pe, mod_shift, mod_scale, mod_gate, out,
                                   /*parallel_qkv=*/true, /*parallel_attention=*/true,
                                   /*parallel_output=*/true,
                                   /*probe_interblock=*/false, /*worker_count=*/4);
}

bool KleinSingleBlockEngine::forwardZeroCopyPooledWithWorkers(
    const std::vector<float>& x, const std::vector<float>& pe,
    const std::vector<float>& mod_shift, const std::vector<float>& mod_scale,
    const std::vector<float>& mod_gate, std::vector<float>* out, int worker_count) {
  if (worker_count < 1 || worker_count > kNumChunks) return false;
  return forwardZeroCopyPooledImpl(x, pe, mod_shift, mod_scale, mod_gate, out,
                                   /*parallel_qkv=*/worker_count > 1,
                                   /*parallel_attention=*/worker_count > 1,
                                   /*parallel_output=*/worker_count > 1,
                                   /*probe_interblock=*/false, worker_count);
}

bool KleinSingleBlockEngine::forwardZeroCopyPooledInterBlockProbe(
    const std::vector<float>& x, const std::vector<float>& pe,
    const std::vector<float>& mod_shift, const std::vector<float>& mod_scale,
    const std::vector<float>& mod_gate, std::vector<float>* out) {
  return forwardZeroCopyPooledImpl(x, pe, mod_shift, mod_scale, mod_gate, out,
                                   /*parallel_qkv=*/true, /*parallel_attention=*/true,
                                   /*parallel_output=*/false,
                                   /*probe_interblock=*/true, /*worker_count=*/4);
}

bool KleinSingleBlockEngine::forwardZeroCopyPooledImpl(
    const std::vector<float>& x, const std::vector<float>& pe,
    const std::vector<float>& mod_shift, const std::vector<float>& mod_scale,
    const std::vector<float>& mod_gate, std::vector<float>* out, bool parallel_qkv,
    bool parallel_attention, bool parallel_output, bool probe_interblock, int worker_count) {
  if (!IsLoaded() || x.size() != static_cast<size_t>(kTokenCount) * kHiddenSize ||
      pe.size() != static_cast<size_t>(kTokenCount) * kPeStride ||
      mod_shift.size() != kHiddenSize || mod_scale.size() != kHiddenSize ||
      mod_gate.size() != kHiddenSize) {
    return false;
  }

  const std::vector<int32_t> x_shape = {1, kChunk, kHiddenSize};
  const std::vector<int32_t> pe_shape = {1, 1, kChunk, kRopeComplexPairs, 2, 2};
  const std::vector<int32_t> mod_shape = {1, 1, kHiddenSize};
  const std::vector<int32_t> qkv_shape = {1, kHeads, kChunk, kHeadDim};
  const std::vector<int32_t> mlp_shape = {1, kChunk, kMlpHiddenDim};
  const std::vector<int32_t> state_shape = {1, kHeads, kChunk, 1};

  auto allocate_outputs = [&](CachedPiece* piece,
                              const std::vector<std::vector<int32_t>>& shapes,
                              BufferSet* set) -> bool {
    set->holders.resize(shapes.size());
    set->buffers.resize(shapes.size());
    set->sizes.resize(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
      LiteRtTensorBufferRequirements requirements = nullptr;
      if (LiteRtGetCompiledModelOutputBufferRequirements(piece->compiled.compiled_model, 0, i,
                                                         &requirements) != kLiteRtStatusOk) {
        return false;
      }
      size_t bytes = 0;
      LiteRtGetTensorBufferRequirementsBufferSize(requirements, &bytes);
      LiteRtRankedTensorType type = MakeFloat32TensorType(shapes[i]);
      if (LiteRtCreateManagedTensorBuffer(env_, kLiteRtTensorBufferTypeDmaBuf, &type, bytes,
                                          &set->holders[i].buffer) != kLiteRtStatusOk) {
        return false;
      }
      set->buffers[i] = set->holders[i].buffer;
      set->sizes[i] = ElementCount(shapes[i]);
    }
    return true;
  };
  auto write_input = [](CachedPiece* piece, size_t index,
                        const std::vector<float>& value) -> bool {
    void* memory = nullptr;
    if (LiteRtLockTensorBuffer(piece->input_holders[index].buffer, &memory,
                               kLiteRtTensorBufferLockModeWrite) != kLiteRtStatusOk) {
      return false;
    }
    std::memcpy(memory, value.data(), value.size() * sizeof(float));
    LiteRtUnlockTensorBuffer(piece->input_holders[index].buffer);
    return true;
  };

  // Keep Q/K/V and the corresponding MLP output for every query chunk in DmaBuf. MLP cannot
  // be regenerated later: its 85 MiB/chunk output is an input to the final projection.
  BufferSet qkv_sets[kNumChunks];
  CachedPiece* qkv_pieces[kNumChunks] = {};
  std::string qkv_keys[kNumChunks];
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    qkv_keys[chunk] = parallel_qkv ? "parallel_qkv_" + std::to_string(chunk) : "pool_qkv";
    qkv_pieces[chunk] = PrepareCachedPiece(
        qkv_file_,
        {x_shape, pe_shape, mod_shape, mod_shape},
        {qkv_shape, qkv_shape, qkv_shape, mlp_shape}, qkv_keys[chunk]);
    if (!qkv_pieces[chunk]) return false;
    std::vector<float> x_chunk = Slice(x, static_cast<size_t>(chunk) * kChunk * kHiddenSize,
                                       kChunk * kHiddenSize);
    std::vector<float> pe_chunk = Slice(pe, static_cast<size_t>(chunk) * kChunk * kPeStride,
                                        kChunk * kPeStride);
    if (!allocate_outputs(qkv_pieces[chunk], {qkv_shape, qkv_shape, qkv_shape, mlp_shape},
                          &qkv_sets[chunk]) ||
        !write_input(qkv_pieces[chunk], 0, x_chunk) ||
        !write_input(qkv_pieces[chunk], 1, pe_chunk) ||
        !write_input(qkv_pieces[chunk], 2, mod_shift) ||
        !write_input(qkv_pieces[chunk], 3, mod_scale)) {
      return false;
    }
  }
  if (parallel_qkv) {
    bool qkv_ok[kNumChunks] = {};
    for (int first = 0; first < kNumChunks; first += worker_count) {
      const int count = std::min(worker_count, kNumChunks - first);
      std::thread workers[kNumChunks];
      for (int worker = 0; worker < count; ++worker) {
        const int chunk = first + worker;
        workers[worker] = std::thread([&, chunk]() {
          qkv_ok[chunk] = RunCachedDirect(qkv_keys[chunk], qkv_pieces[chunk]->input_buffers,
                                          &qkv_sets[chunk].buffers);
        });
      }
      for (int worker = 0; worker < count; ++worker) workers[worker].join();
    }
    for (bool ok : qkv_ok) if (!ok) return false;
  } else {
    for (int chunk = 0; chunk < kNumChunks; ++chunk) {
      if (!RunCachedDirect(qkv_keys[chunk], qkv_pieces[chunk]->input_buffers,
                           &qkv_sets[chunk].buffers)) {
        return false;
      }
    }
  }

  CachedPiece* init_pieces[kNumChunks] = {};
  CachedPiece* flash_pieces[kNumChunks] = {};
  CachedPiece* finalize_pieces[kNumChunks] = {};
  std::string init_keys[kNumChunks];
  std::string flash_keys[kNumChunks];
  std::string finalize_keys[kNumChunks];
  BufferSet state_a[kNumChunks];
  BufferSet state_b[kNumChunks];
  BufferSet final_attention[kNumChunks];
  for (int query_chunk = 0; query_chunk < kNumChunks; ++query_chunk) {
    init_keys[query_chunk] = parallel_attention
        ? "parallel_init_" + std::to_string(query_chunk) : "pool_init";
    flash_keys[query_chunk] = parallel_attention
        ? "parallel_flash_" + std::to_string(query_chunk) : "pool_flash";
    finalize_keys[query_chunk] = parallel_attention
        ? "parallel_finalize_" + std::to_string(query_chunk) : "pool_finalize";
    init_pieces[query_chunk] = PrepareCachedPiece(
        "flash_step_init_probe_Google_Tensor_G5.tflite", {qkv_shape, qkv_shape, qkv_shape},
        {state_shape, state_shape, qkv_shape}, init_keys[query_chunk]);
    flash_pieces[query_chunk] = PrepareCachedPiece(
        "flash_step_probe_1152_Google_Tensor_G5.tflite",
        {qkv_shape, qkv_shape, qkv_shape, state_shape, state_shape, qkv_shape},
        {state_shape, state_shape, qkv_shape}, flash_keys[query_chunk]);
    finalize_pieces[query_chunk] = PrepareCachedPiece(
        "attn_finalize_probe_1152_Google_Tensor_G5.tflite", {state_shape, qkv_shape},
        {x_shape}, finalize_keys[query_chunk]);
    if (!init_pieces[query_chunk] || !flash_pieces[query_chunk] ||
        !finalize_pieces[query_chunk] ||
        !allocate_outputs(init_pieces[query_chunk], {state_shape, state_shape, qkv_shape},
                          &state_a[query_chunk]) ||
        !allocate_outputs(flash_pieces[query_chunk], {state_shape, state_shape, qkv_shape},
                          &state_b[query_chunk]) ||
        !allocate_outputs(finalize_pieces[query_chunk], {x_shape},
                          &final_attention[query_chunk])) {
      return false;
    }
  }
  auto run_attention_chain = [&](int query_chunk) -> bool {
    if (!RunCachedDirect(init_keys[query_chunk],
                         {qkv_sets[query_chunk].buffers[0], qkv_sets[0].buffers[1],
                          qkv_sets[0].buffers[2]},
                         &state_a[query_chunk].buffers)) {
      return false;
    }
    BufferSet* current = &state_a[query_chunk];
    for (int key_chunk = 1; key_chunk < kNumChunks; ++key_chunk) {
      BufferSet* next = current == &state_a[query_chunk] ? &state_b[query_chunk]
                                                          : &state_a[query_chunk];
      if (!RunCachedDirect(flash_keys[query_chunk],
                           {qkv_sets[query_chunk].buffers[0], qkv_sets[key_chunk].buffers[1],
                            qkv_sets[key_chunk].buffers[2], current->buffers[0],
                            current->buffers[1], current->buffers[2]},
                           &next->buffers)) {
        return false;
      }
      current = next;
    }
    return RunCachedDirect(finalize_keys[query_chunk],
                           {current->buffers[1], current->buffers[2]},
                           &final_attention[query_chunk].buffers);
  };
  if (parallel_attention) {
    bool attention_ok[kNumChunks] = {};
    for (int first = 0; first < kNumChunks; first += worker_count) {
      const int count = std::min(worker_count, kNumChunks - first);
      std::thread workers[kNumChunks];
      for (int worker = 0; worker < count; ++worker) {
        const int query_chunk = first + worker;
        workers[worker] = std::thread([&, query_chunk]() {
          attention_ok[query_chunk] = run_attention_chain(query_chunk);
        });
      }
      for (int worker = 0; worker < count; ++worker) workers[worker].join();
    }
    for (bool ok : attention_ok) if (!ok) return false;
  } else {
    for (int query_chunk = 0; query_chunk < kNumChunks; ++query_chunk) {
      if (!run_attention_chain(query_chunk)) return false;
    }
  }

  CachedPiece* out_pieces[kNumChunks] = {};
  std::string out_keys[kNumChunks];
  BufferSet output_sets[kNumChunks];
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    out_keys[chunk] = parallel_output ? "parallel_out_" + std::to_string(chunk) : "pool_out";
    out_pieces[chunk] = PrepareCachedPiece(
        out_file_,
        {x_shape, x_shape, mlp_shape, mod_shape}, {x_shape}, out_keys[chunk]);
    if (!out_pieces[chunk]) return false;
    if (probe_interblock &&
        !allocate_outputs(out_pieces[chunk], {x_shape}, &output_sets[chunk])) {
      return false;
    }
    std::vector<float> x_chunk = Slice(x, static_cast<size_t>(chunk) * kChunk * kHiddenSize,
                                       kChunk * kHiddenSize);
    if (!write_input(out_pieces[chunk], 0, x_chunk) ||
        !write_input(out_pieces[chunk], 3, mod_gate)) {
      return false;
    }
  }
  bool output_ok[kNumChunks] = {};
  auto run_output_chunk = [&](int chunk) -> bool {
    const std::vector<LiteRtTensorBuffer>* output_buffers =
        probe_interblock ? &output_sets[chunk].buffers : nullptr;
    return RunCachedDirect(out_keys[chunk],
                           {out_pieces[chunk]->input_buffers[0],
                            final_attention[chunk].buffers[0], qkv_sets[chunk].buffers[3],
                            out_pieces[chunk]->input_buffers[3]},
                           output_buffers);
  };
  if (parallel_output) {
    for (int first = 0; first < kNumChunks; first += worker_count) {
      const int count = std::min(worker_count, kNumChunks - first);
      std::thread workers[kNumChunks];
      for (int worker = 0; worker < count; ++worker) {
        const int chunk = first + worker;
        workers[worker] = std::thread([&, chunk]() {
          output_ok[chunk] = run_output_chunk(chunk);
        });
      }
      for (int worker = 0; worker < count; ++worker) workers[worker].join();
    }
  } else {
    for (int chunk = 0; chunk < kNumChunks; ++chunk) output_ok[chunk] = run_output_chunk(chunk);
  }
  for (bool ok : output_ok) if (!ok) return false;

  out->resize(static_cast<size_t>(kTokenCount) * kHiddenSize);
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    void* memory = nullptr;
    LiteRtTensorBuffer output_buffer = probe_interblock ? output_sets[chunk].buffers[0]
                                                        : out_pieces[chunk]->output_holders[0].buffer;
    if (LiteRtLockTensorBuffer(output_buffer, &memory,
                               kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk) {
      return false;
    }
    std::memcpy(out->data() + static_cast<size_t>(chunk) * kChunk * kHiddenSize, memory,
                out_pieces[chunk]->output_sizes[0] * sizeof(float));
    LiteRtUnlockTensorBuffer(output_buffer);
  }
  if (probe_interblock) {
    for (int chunk = 0; chunk < kNumChunks; ++chunk) {
      if (!RunCachedDirect(qkv_keys[chunk],
                           {output_sets[chunk].buffers[0], qkv_pieces[chunk]->input_buffers[1],
                            qkv_pieces[chunk]->input_buffers[2], qkv_pieces[chunk]->input_buffers[3]})) {
        NPU_LOGE("KleinSingleBlockEngine: inter-block DmaBuf handoff rejected at chunk %d\n",
                 chunk);
        return false;
      }
    }
    NPU_LOGE("KleinSingleBlockEngine: inter-block DmaBuf handoff accepted for all %d chunks\n",
             kNumChunks);
  }
  return true;
}

}  // namespace pockettavern
