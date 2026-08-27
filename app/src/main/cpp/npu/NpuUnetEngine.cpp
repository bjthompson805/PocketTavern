#include "npu/NpuUnetEngine.hpp"

#include <android/log.h>
#include <sys/stat.h>

#include <chrono>
#include <cstring>
#include <unordered_map>

#include <MNN/MNNDefine.h>

// MNN_ERROR (MNN/MNNDefine.h) logs under tag "MNNJNI" via __android_log_print when built with
// -DANDROID -- confirmed by reading the macro definition -- but produced NO visible logcat
// output at all when used from this file during on-device debugging (root cause not yet
// understood: this target doesn't inherit MNN's own private compile definitions since it's a
// separate CMake target that only links against MNN, not a MNN_USE_LOGCAT/-fvisibility
// difference issue based on the header read, but unconfirmed). Bypassing it entirely here with a
// direct, independently-verified-working __android_log_print call under the SAME
// "PocketTavernDiffusion" tag jni_diffusion.cpp's own LOGE already uses successfully, rather than
// spend more time on MNN_ERROR's specific failure mode in this translation unit.
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

bool FileExists(const std::string& path) {
  struct stat st{};
  return ::stat(path.c_str(), &st) == 0;
}

// Every role's fixed tensor shape (float32 throughout), captured directly from the compiled
// piece files by build_full_unet_wrapped.py's manifest generation -- see
// full_unet_pieces.json's "role_shapes". Keep in sync with that file if the piece set changes;
// do not hand-edit shapes here without regenerating from the same source.
const std::unordered_map<std::string, std::vector<int32_t>>& RoleShapes() {
  static const std::unordered_map<std::string, std::vector<int32_t>> kShapes = {
      {"t_emb", {1, 320}},
      {"text_embeds", {1, 1280}},
      {"time_ids", {1, 6}},
      {"emb", {1, 1280}},
      {"sample", {1, 4, 128, 128}},
      {"skip0", {1, 320, 128, 128}},
      {"skip1", {1, 320, 128, 128}},
      {"skip2", {1, 320, 128, 128}},
      {"skip3", {1, 320, 64, 64}},
      {"d1r0", {1, 640, 64, 64}},
      {"ehs", {1, 77, 2048}},
      {"skip4", {1, 640, 64, 64}},
      {"d1r1", {1, 640, 64, 64}},
      {"skip5", {1, 640, 64, 64}},
      {"skip6", {1, 640, 32, 32}},
      {"d2r0", {1, 1280, 32, 32}},
      {"d2a0_hidden", {1, 1024, 1280}},
      {"skip7", {1, 1280, 32, 32}},
      {"d2r1", {1, 1280, 32, 32}},
      {"d2a1_hidden", {1, 1024, 1280}},
      {"skip8", {1, 1280, 32, 32}},
      {"mid_r0", {1, 1280, 32, 32}},
      {"mid_hidden", {1, 1024, 1280}},
      {"mid_attn_out", {1, 1280, 32, 32}},
      {"mid_final", {1, 1280, 32, 32}},
      {"u0r0", {1, 1280, 32, 32}},
      {"u0a0_hidden", {1, 1024, 1280}},
      {"u0_after0", {1, 1280, 32, 32}},
      {"u0r1", {1, 1280, 32, 32}},
      {"u0a1_hidden", {1, 1024, 1280}},
      {"u0_after1", {1, 1280, 32, 32}},
      {"u0r2", {1, 1280, 32, 32}},
      {"u0a2_hidden", {1, 1024, 1280}},
      {"u0_after2", {1, 1280, 32, 32}},
      {"up0_out", {1, 1280, 64, 64}},
      {"u1r0", {1, 640, 64, 64}},
      {"u1_after0", {1, 640, 64, 64}},
      {"u1r1", {1, 640, 64, 64}},
      {"u1_after1", {1, 640, 64, 64}},
      {"u1r2", {1, 640, 64, 64}},
      {"u1_after2", {1, 640, 64, 64}},
      {"up1_out", {1, 640, 128, 128}},
      {"final_output", {1, 4, 128, 128}},
  };
  return kShapes;
}

size_t RoleElementCount(const std::vector<int32_t>& shape) {
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

// RAII wrappers -- this codebase builds with -fno-exceptions, so early `return false` on any
// LiteRtStatus failure must still release every LiteRT handle already created in this call.
// Destructors run on ordinary scope exit regardless of exceptions; only actual stack unwinding
// through a throw would skip them, which never happens here.
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

}  // namespace

const std::vector<NpuUnetEngine::PieceSpec>& NpuUnetEngine::Pieces() {
  // Transcribed from full_unet_pieces.json (build_full_unet_wrapped.py) -- same source of truth
  // already used for NpuDiagnostic.kt's UNET_PIECES. Already in dependency order.
  static const std::vector<PieceSpec> kPieces = {
      {"embed", "u_embed_wrapped.tflite", {"t_emb", "text_embeds", "time_ids"}, {"emb"}},
      {"conv_in", "u_conv_in_wrapped.tflite", {"sample"}, {"skip0"}},
      {"down0", "u_down0_wrapped.tflite", {"skip0", "emb"}, {"skip1", "skip2", "skip3"}},
      {"down1_resnet0", "u_down1_resnet0_wrapped.tflite", {"skip3", "emb"}, {"d1r0"}},
      {"down1_attn0", "u_down1_attn0_wrapped.tflite", {"d1r0", "ehs"}, {"skip4"}},
      {"down1_resnet1", "u_down1_resnet1_wrapped.tflite", {"skip4", "emb"}, {"d1r1"}},
      {"down1_attn1", "u_down1_attn1_wrapped.tflite", {"d1r1", "ehs"}, {"skip5"}},
      {"down1_downsample", "u_down1_downsample_wrapped.tflite", {"skip5"}, {"skip6"}},
      {"down2_resnet0", "u_down2_resnet0_wrapped.tflite", {"skip6", "emb"}, {"d2r0"}},
      {"down2_attn0_a", "u_down2_attn0_a_wrapped.tflite", {"d2r0", "ehs"}, {"d2a0_hidden"}},
      {"down2_attn0_b", "u_down2_attn0_b_wrapped.tflite", {"d2a0_hidden", "d2r0", "ehs"}, {"skip7"}},
      {"down2_resnet1", "u_down2_resnet1_wrapped.tflite", {"skip7", "emb"}, {"d2r1"}},
      {"down2_attn1_a", "u_down2_attn1_a_wrapped.tflite", {"d2r1", "ehs"}, {"d2a1_hidden"}},
      {"down2_attn1_b", "u_down2_attn1_b_wrapped.tflite", {"d2a1_hidden", "d2r1", "ehs"}, {"skip8"}},
      {"mid_resnet0", "u_mid_resnet0_wrapped.tflite", {"skip8", "emb"}, {"mid_r0"}},
      {"mid_attn0", "u_mid_attn0_wrapped.tflite", {"mid_r0", "ehs"}, {"mid_hidden"}},
      {"mid_attn1", "u_mid_attn1_wrapped.tflite", {"mid_hidden", "mid_r0", "ehs"}, {"mid_attn_out"}},
      {"mid_resnet1", "u_mid_resnet1_wrapped.tflite", {"mid_attn_out", "emb"}, {"mid_final"}},
      {"up0_resnet0", "u_up0_resnet0_wrapped.tflite", {"mid_final", "skip8", "emb"}, {"u0r0"}},
      {"up0_attn0_a", "u_up0_attn0_a_wrapped.tflite", {"u0r0", "ehs"}, {"u0a0_hidden"}},
      {"up0_attn0_b", "u_up0_attn0_b_wrapped.tflite", {"u0a0_hidden", "u0r0", "ehs"}, {"u0_after0"}},
      {"up0_resnet1", "u_up0_resnet1_wrapped.tflite", {"u0_after0", "skip7", "emb"}, {"u0r1"}},
      {"up0_attn1_a", "u_up0_attn1_a_wrapped.tflite", {"u0r1", "ehs"}, {"u0a1_hidden"}},
      {"up0_attn1_b", "u_up0_attn1_b_wrapped.tflite", {"u0a1_hidden", "u0r1", "ehs"}, {"u0_after1"}},
      {"up0_resnet2", "u_up0_resnet2_wrapped.tflite", {"u0_after1", "skip6", "emb"}, {"u0r2"}},
      {"up0_attn2_a", "u_up0_attn2_a_wrapped.tflite", {"u0r2", "ehs"}, {"u0a2_hidden"}},
      {"up0_attn2_b", "u_up0_attn2_b_wrapped.tflite", {"u0a2_hidden", "u0r2", "ehs"}, {"u0_after2"}},
      {"up0_upsample", "u_up0_upsample_wrapped.tflite", {"u0_after2"}, {"up0_out"}},
      {"up1_resnet0", "u_up1_resnet0_wrapped.tflite", {"up0_out", "skip5", "emb"}, {"u1r0"}},
      {"up1_attn0", "u_up1_attn0_wrapped.tflite", {"u1r0", "ehs"}, {"u1_after0"}},
      {"up1_resnet1", "u_up1_resnet1_wrapped.tflite", {"u1_after0", "skip4", "emb"}, {"u1r1"}},
      {"up1_attn1", "u_up1_attn1_wrapped.tflite", {"u1r1", "ehs"}, {"u1_after1"}},
      {"up1_resnet2", "u_up1_resnet2_wrapped.tflite", {"u1_after1", "skip3", "emb"}, {"u1r2"}},
      {"up1_attn2", "u_up1_attn2_wrapped.tflite", {"u1r2", "ehs"}, {"u1_after2"}},
      {"up1_upsample", "u_up1_upsample_wrapped.tflite", {"u1_after2"}, {"up1_out"}},
      {"up2", "u_up2_wrapped.tflite", {"up1_out", "emb", "skip0", "skip1", "skip2"}, {"final_output"}},
  };
  return kPieces;
}

NpuUnetEngine::NpuUnetEngine() = default;

NpuUnetEngine::~NpuUnetEngine() {
  if (env_) {
    LiteRtDestroyEnvironment(env_);
    env_ = nullptr;
  }
}

bool NpuUnetEngine::Load(std::string model_dir, const std::string& dispatch_lib_dir) {
  if (env_ != nullptr) {
    NPU_LOGE("NpuUnetEngine::Load called twice on the same instance\n");
    return false;
  }
  model_dir_ = std::move(model_dir);

  for (const auto& piece : Pieces()) {
    std::string path = model_dir_ + "/" + piece.file_name;
    if (!FileExists(path)) {
      NPU_LOGE("NpuUnetEngine::Load: missing piece file %s\n", path.c_str());
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
    NPU_LOGE("NpuUnetEngine::Load: LiteRtCreateEnvironment failed, status=%d\n", status);
    env_ = nullptr;
    return false;
  }
  return true;
}

bool NpuUnetEngine::RunPiece(const PieceSpec& piece,
                              const std::vector<const std::vector<float>*>& piece_inputs,
                              std::vector<std::vector<float>>* out_results) {
  const auto start = std::chrono::steady_clock::now();
  const auto elapsed_ms = [&start]() -> long long {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - start)
        .count();
  };
  const std::string path = model_dir_ + "/" + piece.file_name;
  const auto& shapes = RoleShapes();

  ModelHolder model_holder;
  LiteRtStatus status = LiteRtCreateModelFromFile(env_, path.c_str(), &model_holder.model);
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("NpuUnetEngine: %s: LiteRtCreateModelFromFile failed, status=%d\n", piece.name, status);
    return false;
  }
  const long long model_ms = elapsed_ms();

  // A non-null options object is required by this LiteRT C API build, despite the header
  // documenting it as optional. RESHAPE-wrapped pieces need CPU available for the wrapper op:
  // restricting this C API path to NPU alone rejects the first piece with status 504.
  OptionsHolder options_holder;
  status = LiteRtCreateOptions(&options_holder.options);
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("NpuUnetEngine: %s: LiteRtCreateOptions failed, status=%d\n", piece.name, status);
    return false;
  }
  status = LiteRtSetOptionsHardwareAccelerators(
      options_holder.options, kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu);
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("NpuUnetEngine: %s: LiteRtSetOptionsHardwareAccelerators failed, status=%d\n", piece.name, status);
    return false;
  }
  CompiledModelHolder cm_holder;
  status = LiteRtCreateCompiledModel(env_, model_holder.model, options_holder.options,
                                      &cm_holder.compiled_model);
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("NpuUnetEngine: %s: LiteRtCreateCompiledModel failed, status=%d\n", piece.name, status);
    return false;
  }
  const long long compile_ms = elapsed_ms();

  // Input buffers: allocate, write.
  std::vector<TensorBufferHolder> input_holders(piece.input_roles.size());
  std::vector<LiteRtTensorBuffer> input_buffers(piece.input_roles.size());
  for (size_t i = 0; i < piece.input_roles.size(); ++i) {
    const char* role = piece.input_roles[i];
    auto shape_it = shapes.find(role);
    if (shape_it == shapes.end()) {
      NPU_LOGE("NpuUnetEngine: %s: no known shape for role '%s'\n", piece.name, role);
      return false;
    }
    const std::vector<int32_t>& shape = shape_it->second;
    const size_t num_elements = RoleElementCount(shape);
    if (piece_inputs[i]->size() != num_elements) {
      NPU_LOGE("NpuUnetEngine: %s: role '%s' expected %zu elements, got %zu\n", piece.name, role,
                 num_elements, piece_inputs[i]->size());
      return false;
    }

    // Query the compiled model's OWN buffer requirements rather than assuming
    // num_elements*sizeof(float) is enough -- the NPU dispatch layer can need padding/alignment
    // beyond the raw tensor size, especially for small tensors (confirmed on-device: a [1,6]
    // tensor, 24 bytes of real data, needed a 64-byte buffer; a naively-sized 24-byte buffer
    // caused a real dispatch failure, not just a benign warning). The returned
    // LiteRtTensorBufferRequirements is owned by compiled_model, not by us -- do not destroy it.
    LiteRtTensorBufferRequirements requirements = nullptr;
    status = LiteRtGetCompiledModelInputBufferRequirements(cm_holder.compiled_model, /*signature_index=*/0,
                                                             /*input_index=*/i, &requirements);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("NpuUnetEngine: %s: LiteRtGetCompiledModelInputBufferRequirements (input '%s') failed, status=%d\n",
                 piece.name, role, status);
      return false;
    }

    LiteRtRankedTensorType tensor_type = MakeFloat32TensorType(shape);
    status = LiteRtCreateManagedTensorBufferFromRequirements(env_, &tensor_type, requirements,
                                                               &input_holders[i].buffer);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("NpuUnetEngine: %s: LiteRtCreateManagedTensorBufferFromRequirements (input '%s') failed, status=%d\n",
                 piece.name, role, status);
      return false;
    }

    void* host_mem = nullptr;
    status = LiteRtLockTensorBuffer(input_holders[i].buffer, &host_mem, kLiteRtTensorBufferLockModeWrite);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("NpuUnetEngine: %s: LiteRtLockTensorBuffer (input '%s') failed, status=%d\n", piece.name,
                 role, status);
      return false;
    }
    std::memcpy(host_mem, piece_inputs[i]->data(), num_elements * sizeof(float));
    LiteRtUnlockTensorBuffer(input_holders[i].buffer);

    input_buffers[i] = input_holders[i].buffer;
  }
  const long long inputs_ms = elapsed_ms();

  // Output buffers: allocate only (written by the run).
  std::vector<TensorBufferHolder> output_holders(piece.output_roles.size());
  std::vector<LiteRtTensorBuffer> output_buffers(piece.output_roles.size());
  std::vector<size_t> output_element_counts(piece.output_roles.size());
  for (size_t i = 0; i < piece.output_roles.size(); ++i) {
    const char* role = piece.output_roles[i];
    auto shape_it = shapes.find(role);
    if (shape_it == shapes.end()) {
      NPU_LOGE("NpuUnetEngine: %s: no known shape for role '%s'\n", piece.name, role);
      return false;
    }
    const std::vector<int32_t>& shape = shape_it->second;
    const size_t num_elements = RoleElementCount(shape);
    output_element_counts[i] = num_elements;

    LiteRtTensorBufferRequirements requirements = nullptr;
    status = LiteRtGetCompiledModelOutputBufferRequirements(cm_holder.compiled_model, /*signature_index=*/0,
                                                              /*output_index=*/i, &requirements);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("NpuUnetEngine: %s: LiteRtGetCompiledModelOutputBufferRequirements (output '%s') failed, status=%d\n",
                 piece.name, role, status);
      return false;
    }

    LiteRtRankedTensorType tensor_type = MakeFloat32TensorType(shape);
    status = LiteRtCreateManagedTensorBufferFromRequirements(env_, &tensor_type, requirements,
                                                               &output_holders[i].buffer);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("NpuUnetEngine: %s: LiteRtCreateManagedTensorBufferFromRequirements (output '%s') failed, status=%d\n",
                 piece.name, role, status);
      return false;
    }
    output_buffers[i] = output_holders[i].buffer;
  }
  const long long outputs_ms = elapsed_ms();

  status = LiteRtRunCompiledModel(cm_holder.compiled_model, /*signature_index=*/0, input_buffers.size(),
                                   input_buffers.data(), output_buffers.size(), output_buffers.data());
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("NpuUnetEngine: %s: LiteRtRunCompiledModel failed, status=%d\n", piece.name, status);
    return false;
  }
  const long long run_ms = elapsed_ms();

  out_results->clear();
  out_results->resize(piece.output_roles.size());
  for (size_t i = 0; i < piece.output_roles.size(); ++i) {
    void* host_mem = nullptr;
    status = LiteRtLockTensorBuffer(output_holders[i].buffer, &host_mem, kLiteRtTensorBufferLockModeRead);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("NpuUnetEngine: %s: LiteRtLockTensorBuffer (output '%s') failed, status=%d\n", piece.name,
                 piece.output_roles[i], status);
      return false;
    }
    (*out_results)[i].resize(output_element_counts[i]);
    std::memcpy((*out_results)[i].data(), host_mem, output_element_counts[i] * sizeof(float));
    LiteRtUnlockTensorBuffer(output_holders[i].buffer);
  }

  NPU_LOGE("NpuUnetEngine: phases %s: model=%lldms compile=%lldms inputs=%lldms outputs=%lldms run=%lldms read=%lldms\n",
           piece.name, model_ms, compile_ms - model_ms, inputs_ms - compile_ms,
           outputs_ms - inputs_ms, run_ms - outputs_ms, elapsed_ms() - run_ms);

  return true;
  // ModelHolder/OptionsHolder/CompiledModelHolder/TensorBufferHolder destructors release every
  // LiteRT handle created above, on this and every early-return path.
}

bool NpuUnetEngine::forward(const NpuUnetInputs& inputs, std::vector<float>* out_noise_pred) {
  if (!IsLoaded()) {
    NPU_LOGE("NpuUnetEngine::forward called before a successful Load()\n");
    return false;
  }

  std::unordered_map<std::string, std::vector<float>> role_map;
  role_map["sample"] = inputs.sample;
  role_map["t_emb"] = inputs.t_emb;
  role_map["text_embeds"] = inputs.text_embeds;
  role_map["time_ids"] = inputs.time_ids;
  role_map["ehs"] = inputs.encoder_hidden_states;

  const auto& pieces = Pieces();

  // Liveness cleanup: free a role's value right after its LAST consuming piece reads it, same
  // fix already applied on the Kotlin side (NpuDiagnostic.kt's runFullUnetSeparate) after
  // hitting a real 256MB Java heap OOM from an unbounded role map -- C++ has no GC to paper over
  // the same mistake, so this matters here too, arguably more.
  std::unordered_map<std::string, int> last_consumer_index;
  for (size_t i = 0; i < pieces.size(); ++i) {
    for (const char* role : pieces[i].input_roles) {
      last_consumer_index[role] = static_cast<int>(i);
    }
  }

  for (size_t i = 0; i < pieces.size(); ++i) {
    const PieceSpec& piece = pieces[i];
    std::vector<const std::vector<float>*> piece_inputs;
    piece_inputs.reserve(piece.input_roles.size());
    for (const char* role : piece.input_roles) {
      auto it = role_map.find(role);
      if (it == role_map.end()) {
        NPU_LOGE("NpuUnetEngine::forward: missing role '%s' for piece '%s'\n", role, piece.name);
        return false;
      }
      piece_inputs.push_back(&it->second);
    }

    std::vector<std::vector<float>> results;
    auto piece_start = std::chrono::steady_clock::now();
    if (!RunPiece(piece, piece_inputs, &results)) {
      NPU_LOGE("NpuUnetEngine::forward: piece '%s' failed\n", piece.name);
      return false;
    }
    auto piece_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - piece_start)
                        .count();
    NPU_LOGE("NpuUnetEngine: piece %s: %lldms\n", piece.name, static_cast<long long>(piece_ms));

    for (const char* role : piece.input_roles) {
      if (last_consumer_index[role] == static_cast<int>(i)) {
        role_map.erase(role);
      }
    }
    for (size_t j = 0; j < piece.output_roles.size(); ++j) {
      role_map[piece.output_roles[j]] = std::move(results[j]);
    }
  }

  auto final_it = role_map.find("final_output");
  if (final_it == role_map.end()) {
    NPU_LOGE("NpuUnetEngine::forward: final_output was never produced\n");
    return false;
  }
  *out_noise_pred = std::move(final_it->second);
  return true;
}

}  // namespace pockettavern
