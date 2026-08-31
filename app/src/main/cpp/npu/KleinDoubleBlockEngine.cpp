#include "npu/KleinDoubleBlockEngine.hpp"

#include <android/log.h>
#include <sys/stat.h>

#include <chrono>
#include <cstring>

#define NPU_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "PocketTavernDiffusion", __VA_ARGS__)

#include "litert/c/litert_any.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"

namespace pockettavern {
namespace {
constexpr int kTxtTokens = 512;
constexpr int kImgTokens = 4096;
constexpr int kImgChunk = 1024;
constexpr int kNumChunks = 5;
constexpr int kHidden = 3072;
constexpr int kHeads = 24;
constexpr int kHeadDim = 128;
constexpr int kPeStride = 256;

bool FileExists(const std::string& path) { struct stat st{}; return ::stat(path.c_str(), &st) == 0; }
size_t ElementCount(const std::vector<int32_t>& shape) { size_t n = 1; for (int32_t d : shape) n *= d; return n; }
LiteRtRankedTensorType FloatType(const std::vector<int32_t>& shape) {
  LiteRtRankedTensorType t{};
  t.element_type = kLiteRtElementTypeFloat32;
  t.layout.rank = static_cast<unsigned int>(shape.size());
  t.layout.has_strides = 0;
  for (size_t i = 0; i < shape.size(); ++i) t.layout.dimensions[i] = shape[i];
  return t;
}
std::vector<float> Slice(const std::vector<float>& source, size_t offset, size_t count) {
  return std::vector<float>(source.begin() + static_cast<long>(offset), source.begin() + static_cast<long>(offset + count));
}
struct ModelHolder { LiteRtModel value = nullptr; ~ModelHolder() { if (value) LiteRtDestroyModel(value); } };
struct OptionsHolder { LiteRtOptions value = nullptr; ~OptionsHolder() { if (value) LiteRtDestroyOptions(value); } };
struct CompiledModelHolder { LiteRtCompiledModel value = nullptr; ~CompiledModelHolder() { if (value) LiteRtDestroyCompiledModel(value); } };
struct BufferHolder { LiteRtTensorBuffer value = nullptr; ~BufferHolder() { if (value) LiteRtDestroyTensorBuffer(value); } };
// A separately-owned DmaBuf output set. These are deliberately not CachedPiece members: Q/K/V
// must remain distinct for all five chunks while one compiled qkv model is reused.
struct BufferSet {
  std::vector<BufferHolder> holders;
  std::vector<LiteRtTensorBuffer> buffers;
  std::vector<size_t> sizes;
};
}  // namespace

struct KleinDoubleBlockEngine::CachedPiece {
  ModelHolder model;
  OptionsHolder options;
  CompiledModelHolder compiled;
  std::vector<BufferHolder> input_holders;
  std::vector<BufferHolder> output_holders;
  std::vector<LiteRtTensorBuffer> input_buffers;
  std::vector<LiteRtTensorBuffer> output_buffers;
  std::vector<size_t> output_sizes;
  std::vector<std::vector<int32_t>> input_shapes;
  std::vector<std::vector<int32_t>> output_shapes;

  ~CachedPiece() {
    // The dispatcher can reuse an epoll fd while tensor buffers still reference it. Always tear
    // down the compiled model before the member buffer holders are destructed.
    if (compiled.value) {
      LiteRtDestroyCompiledModel(compiled.value);
      compiled.value = nullptr;
    }
  }
};

KleinDoubleBlockEngine::KleinDoubleBlockEngine() = default;
KleinDoubleBlockEngine::~KleinDoubleBlockEngine() {
  cached_pieces_.clear();
  if (env_) LiteRtDestroyEnvironment(env_);
}

bool KleinDoubleBlockEngine::Load(std::string model_dir, const std::string& dispatch_lib_dir,
                                  int block_index) {
  if (env_) return false;
  model_dir_ = std::move(model_dir);
  if (block_index < 0 || block_index >= 5) {
    NPU_LOGE("KleinDoubleBlockEngine::Load: invalid block index %d", block_index);
    return false;
  }
  const std::string prefix = "double" + std::to_string(block_index) + "_";
  // The fused qkv+norm+rope NPU artifact was earlier suspected of a real Tensor G5 numerical
  // defect (near-zero Q/K -- docs/FLUX2_KLEIN_HANDOFF.md "Critical current status"). That was a
  // false lead: the real cause was a host-side RoPE-generator indexing bug in jni_diffusion.cpp
  // (positions()'s `size_t o` reset to 0 every token instead of n*256, zeroing RoPE for every
  // token but the last). With that fixed, the fused artifact is confirmed correct on real
  // hardware to ~0.3-0.4% relative error, the same tolerance class as every other chained-dispatch
  // measurement in this project -- see docs/flux2-klein-conversion.md's "RoPE generator bug" update.
  // app/src/main/cpp/npu/klein_qk_norm_rope.hpp still exists (unused here) as a validated,
  // host-buildable native RMSNorm+RoPE implementation, in case a future shape/block ever needs it.
  img_qkv_file_ = prefix + "img_qkv_proj_probe_1024_Google_Tensor_G5.tflite";
  txt_qkv_file_ = prefix + "txt_qkv_proj_probe" +
      (block_index == 0 ? "" : "_512") + "_Google_Tensor_G5.tflite";
  img_out_file_ = prefix + "img_out_proj_probe_1024_Google_Tensor_G5.tflite";
  txt_out_file_ = prefix + "txt_out_proj_probe" +
      (block_index == 0 ? "" : "_512") + "_Google_Tensor_G5.tflite";
  const std::string kFiles[] = {
      img_qkv_file_, txt_qkv_file_, img_out_file_, txt_out_file_,
      "flash_step_init_probe_512_Google_Tensor_G5.tflite",
      "flash_step_init_probe_q1024_kv512_Google_Tensor_G5.tflite",
      "flash_step_probe_q512_kv1024_Google_Tensor_G5.tflite",
      "flash_step_probe_1024_Google_Tensor_G5.tflite",
      "attn_finalize_probe_512_Google_Tensor_G5.tflite",
      "attn_finalize_probe_1024_Google_Tensor_G5.tflite",
  };
  for (const std::string& file : kFiles) if (!FileExists(model_dir_ + "/" + file)) { NPU_LOGE("KleinDoubleBlockEngine: missing %s", file.c_str()); return false; }
  LiteRtAny value{}; value.type = kLiteRtAnyTypeString; value.str_value = dispatch_lib_dir.c_str();
  LiteRtEnvOption option{}; option.tag = kLiteRtEnvOptionTagDispatchLibraryDir; option.value = value;
  const LiteRtStatus status = LiteRtCreateEnvironment(1, &option, &env_);
  if (status != kLiteRtStatusOk) { NPU_LOGE("KleinDoubleBlockEngine: environment failed status=%d", status); env_ = nullptr; return false; }
  return true;
}

bool KleinDoubleBlockEngine::RunPiece(const std::string& file_name,
                                      const std::vector<const std::vector<float>*>& inputs,
                                      const std::vector<std::vector<int32_t>>& input_shapes,
                                      const std::vector<std::vector<int32_t>>& output_shapes,
                                      std::vector<std::vector<float>>* out_results,
                                      const std::string& cache_key) {
  const auto start = std::chrono::steady_clock::now();
  const auto elapsed = [&start]() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count(); };
  CachedPiece* piece = nullptr;
  const std::string& key = cache_key.empty() ? file_name : cache_key;
  auto found = cached_pieces_.find(key);
  if (found == cached_pieces_.end()) {
    std::unique_ptr<CachedPiece> created(new CachedPiece());
    LiteRtStatus status = LiteRtCreateModelFromFile(env_, (model_dir_ + "/" + file_name).c_str(), &created->model.value);
    if (status != kLiteRtStatusOk) { NPU_LOGE("KleinDoubleBlockEngine %s: open failed %d", file_name.c_str(), status); return false; }
    if (LiteRtCreateOptions(&created->options.value) != kLiteRtStatusOk ||
        LiteRtSetOptionsHardwareAccelerators(created->options.value, kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu) != kLiteRtStatusOk) return false;
    status = LiteRtCreateCompiledModel(env_, created->model.value, created->options.value, &created->compiled.value);
    if (status != kLiteRtStatusOk) { NPU_LOGE("KleinDoubleBlockEngine %s: compile failed %d", file_name.c_str(), status); return false; }
    piece = created.get();
    cached_pieces_.emplace(key, std::move(created));
  } else {
    piece = found->second.get();
  }
  LiteRtCompiledModel compiled_model = piece->compiled.value;
  const auto setup_ms = elapsed();
  LiteRtStatus status = kLiteRtStatusOk;

  // All calls to a given AOT artifact have fixed shapes. Allocate its DmaBuf-backed buffers once
  // and retain them with the persistent compiled model; subsequent calls only copy new input data
  // in, run, and copy output data out.
  if (piece->input_holders.empty()) {
    piece->input_shapes = input_shapes;
    piece->output_shapes = output_shapes;
    piece->input_holders.resize(inputs.size());
    piece->output_holders.resize(output_shapes.size());
    piece->input_buffers.resize(inputs.size());
    piece->output_buffers.resize(output_shapes.size());
    piece->output_sizes.resize(output_shapes.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      LiteRtTensorBufferRequirements req = nullptr;
      status = LiteRtGetCompiledModelInputBufferRequirements(compiled_model, 0, i, &req);
      if (status != kLiteRtStatusOk) return false;
      size_t bytes = 0; LiteRtGetTensorBufferRequirementsBufferSize(req, &bytes);
      LiteRtRankedTensorType type = FloatType(input_shapes[i]);
      status = LiteRtCreateManagedTensorBuffer(env_, kLiteRtTensorBufferTypeDmaBuf, &type, bytes, &piece->input_holders[i].value);
      if (status != kLiteRtStatusOk) return false;
      piece->input_buffers[i] = piece->input_holders[i].value;
    }
    for (size_t i = 0; i < output_shapes.size(); ++i) {
      piece->output_sizes[i] = ElementCount(output_shapes[i]);
      LiteRtTensorBufferRequirements req = nullptr;
      status = LiteRtGetCompiledModelOutputBufferRequirements(compiled_model, 0, i, &req);
      if (status != kLiteRtStatusOk) return false;
      size_t bytes = 0; LiteRtGetTensorBufferRequirementsBufferSize(req, &bytes);
      LiteRtRankedTensorType type = FloatType(output_shapes[i]);
      status = LiteRtCreateManagedTensorBuffer(env_, kLiteRtTensorBufferTypeDmaBuf, &type, bytes, &piece->output_holders[i].value);
      if (status != kLiteRtStatusOk) return false;
      piece->output_buffers[i] = piece->output_holders[i].value;
    }
  } else if (piece->input_shapes != input_shapes || piece->output_shapes != output_shapes) {
    NPU_LOGE("KleinDoubleBlockEngine %s: attempted buffer reuse with mismatched shapes", file_name.c_str());
    return false;
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    const size_t elements = ElementCount(input_shapes[i]);
    if (inputs[i]->size() != elements) { NPU_LOGE("KleinDoubleBlockEngine %s: bad input %zu", file_name.c_str(), i); return false; }
    void* memory = nullptr;
    if (LiteRtLockTensorBuffer(piece->input_holders[i].value, &memory, kLiteRtTensorBufferLockModeWrite) != kLiteRtStatusOk) return false;
    std::memcpy(memory, inputs[i]->data(), elements * sizeof(float));
    LiteRtUnlockTensorBuffer(piece->input_holders[i].value);
  }
  const auto buffers_ms = elapsed();
  status = LiteRtRunCompiledModel(compiled_model, 0, piece->input_buffers.size(), piece->input_buffers.data(), piece->output_buffers.size(), piece->output_buffers.data());
  if (status != kLiteRtStatusOk) { NPU_LOGE("KleinDoubleBlockEngine %s: run failed %d", file_name.c_str(), status); return false; }
  const auto run_ms = elapsed();
  out_results->assign(output_shapes.size(), {});
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    void* memory = nullptr;
    if (LiteRtLockTensorBuffer(piece->output_holders[i].value, &memory, kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk) return false;
    (*out_results)[i].resize(piece->output_sizes[i]);
    std::memcpy((*out_results)[i].data(), memory, piece->output_sizes[i] * sizeof(float));
    LiteRtUnlockTensorBuffer(piece->output_holders[i].value);
  }
  const auto read_ms = elapsed();
  NPU_LOGE("KleinDoubleBlockEngine: phases %s: setup=%lldms buffers=%lldms run=%lldms read=%lldms total=%lldms", file_name.c_str(), static_cast<long long>(setup_ms), static_cast<long long>(buffers_ms - setup_ms), static_cast<long long>(run_ms - buffers_ms), static_cast<long long>(read_ms - run_ms), static_cast<long long>(read_ms));
  return true;
}

bool KleinDoubleBlockEngine::DebugFirstQkv(
    const std::vector<float>& img, const std::vector<float>& txt,
    const std::vector<float>& pe, const std::vector<float>& pe_ctx,
    const std::vector<float>& img_mod1_shift, const std::vector<float>& img_mod1_scale,
    const std::vector<float>& txt_mod1_shift, const std::vector<float>& txt_mod1_scale,
    std::vector<std::vector<float>>* img_qkv, std::vector<std::vector<float>>* txt_qkv) {
  if (!env_ || !img_qkv || !txt_qkv) return false;
  const auto img_chunk = Slice(img, 0, kImgChunk * kHidden);
  const auto pe_chunk = Slice(pe, 0, kImgChunk * kPeStride);
  const std::vector<int32_t> img_shape = {1, kImgChunk, kHidden};
  const std::vector<int32_t> txt_shape = {1, kTxtTokens, kHidden};
  const std::vector<int32_t> img_pe_shape = {1, 1, kImgChunk, 64, 2, 2};
  const std::vector<int32_t> txt_pe_shape = {1, 1, kTxtTokens, 64, 2, 2};
  const std::vector<int32_t> mod_shape = {1, 1, kHidden};
  const std::vector<int32_t> img_qkv_shape = {1, kHeads, kImgChunk, kHeadDim};
  const std::vector<int32_t> txt_qkv_shape = {1, kHeads, kTxtTokens, kHeadDim};
  return RunPiece(img_qkv_file_, {&img_chunk, &pe_chunk, &img_mod1_shift, &img_mod1_scale},
                  {img_shape, img_pe_shape, mod_shape, mod_shape},
                  {img_qkv_shape, img_qkv_shape, img_qkv_shape}, img_qkv) &&
      RunPiece(txt_qkv_file_, {&txt, &pe_ctx, &txt_mod1_shift, &txt_mod1_scale},
               {txt_shape, txt_pe_shape, mod_shape, mod_shape},
               {txt_qkv_shape, txt_qkv_shape, txt_qkv_shape}, txt_qkv);
}

bool KleinDoubleBlockEngine::RunCachedDirect(const std::string& cache_key,
                                             const std::vector<LiteRtTensorBuffer>& input_buffers,
                                             const std::vector<LiteRtTensorBuffer>* output_buffers) {
  const auto found = cached_pieces_.find(cache_key);
  if (found == cached_pieces_.end()) return false;
  CachedPiece* piece = found->second.get();
  if (input_buffers.size() != piece->input_buffers.size()) return false;
  std::vector<LiteRtTensorBuffer> mutable_inputs = input_buffers;
  std::vector<LiteRtTensorBuffer> mutable_outputs = output_buffers ? *output_buffers : piece->output_buffers;
  if (mutable_outputs.size() != piece->output_buffers.size()) return false;
  const LiteRtStatus status = LiteRtRunCompiledModel(piece->compiled.value, 0, mutable_inputs.size(),
      mutable_inputs.data(), mutable_outputs.size(), mutable_outputs.data());
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("KleinDoubleBlockEngine direct %s: run failed %d", cache_key.c_str(), status);
    return false;
  }
  return true;
}

KleinDoubleBlockEngine::CachedPiece* KleinDoubleBlockEngine::PrepareCachedPiece(
    const std::string& file_name, const std::vector<std::vector<int32_t>>& input_shapes,
    const std::vector<std::vector<int32_t>>& output_shapes, const std::string& cache_key) {
  const std::string& key = cache_key.empty() ? file_name : cache_key;
  auto found = cached_pieces_.find(key);
  if (found != cached_pieces_.end()) {
    CachedPiece* piece = found->second.get();
    return piece->input_shapes == input_shapes && piece->output_shapes == output_shapes ? piece : nullptr;
  }
  std::unique_ptr<CachedPiece> created(new CachedPiece());
  LiteRtStatus status = LiteRtCreateModelFromFile(env_, (model_dir_ + "/" + file_name).c_str(), &created->model.value);
  if (status != kLiteRtStatusOk || LiteRtCreateOptions(&created->options.value) != kLiteRtStatusOk ||
      LiteRtSetOptionsHardwareAccelerators(created->options.value, kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu) != kLiteRtStatusOk ||
      LiteRtCreateCompiledModel(env_, created->model.value, created->options.value, &created->compiled.value) != kLiteRtStatusOk) return nullptr;
  created->input_shapes=input_shapes; created->output_shapes=output_shapes;
  created->input_holders.resize(input_shapes.size()); created->input_buffers.resize(input_shapes.size());
  created->output_holders.resize(output_shapes.size()); created->output_buffers.resize(output_shapes.size()); created->output_sizes.resize(output_shapes.size());
  for (size_t i=0;i<input_shapes.size();++i) { LiteRtTensorBufferRequirements req=nullptr; if (LiteRtGetCompiledModelInputBufferRequirements(created->compiled.value,0,i,&req)!=kLiteRtStatusOk) return nullptr; size_t bytes=0; LiteRtGetTensorBufferRequirementsBufferSize(req,&bytes); LiteRtRankedTensorType type=FloatType(input_shapes[i]); if (LiteRtCreateManagedTensorBuffer(env_,kLiteRtTensorBufferTypeDmaBuf,&type,bytes,&created->input_holders[i].value)!=kLiteRtStatusOk) return nullptr; created->input_buffers[i]=created->input_holders[i].value; }
  for (size_t i=0;i<output_shapes.size();++i) { LiteRtTensorBufferRequirements req=nullptr; if (LiteRtGetCompiledModelOutputBufferRequirements(created->compiled.value,0,i,&req)!=kLiteRtStatusOk) return nullptr; size_t bytes=0; LiteRtGetTensorBufferRequirementsBufferSize(req,&bytes); LiteRtRankedTensorType type=FloatType(output_shapes[i]); if (LiteRtCreateManagedTensorBuffer(env_,kLiteRtTensorBufferTypeDmaBuf,&type,bytes,&created->output_holders[i].value)!=kLiteRtStatusOk) return nullptr; created->output_buffers[i]=created->output_holders[i].value; created->output_sizes[i]=ElementCount(output_shapes[i]); }
  CachedPiece* result=created.get(); cached_pieces_.emplace(key,std::move(created)); return result;
}

bool KleinDoubleBlockEngine::forward(const std::vector<float>& img, const std::vector<float>& txt,
                                     const std::vector<float>& pe, const std::vector<float>& pe_ctx,
                                     const std::vector<float>& ims, const std::vector<float>& imsc, const std::vector<float>& img1g,
                                     const std::vector<float>& im2s, const std::vector<float>& im2sc, const std::vector<float>& img2g,
                                     const std::vector<float>& txs, const std::vector<float>& txsc, const std::vector<float>& txt1g,
                                     const std::vector<float>& tx2s, const std::vector<float>& tx2sc, const std::vector<float>& txt2g,
                                     std::vector<float>* img_out, std::vector<float>* txt_out) {
  if (!env_ || img.size() != kImgTokens * kHidden || txt.size() != kTxtTokens * kHidden || pe.size() != kImgTokens * kPeStride || pe_ctx.size() != kTxtTokens * kPeStride) return false;
  const std::vector<const std::vector<float>*> mods = {&ims,&imsc,&img1g,&im2s,&im2sc,&img2g,&txs,&txsc,&txt1g,&tx2s,&tx2sc,&txt2g};
  for (const auto* mod : mods) if (mod->size() != kHidden) return false;
  const int sizes[kNumChunks] = {kTxtTokens,kImgChunk,kImgChunk,kImgChunk,kImgChunk};
  std::vector<float> q[kNumChunks], k[kNumChunks], v[kNumChunks], attn[kNumChunks];
  const std::vector<int32_t> mod_shape = {1,1,kHidden};
  for (int c = 0; c < kNumChunks; ++c) {
    const bool is_txt = c == 0; const int tokens = sizes[c];
    std::vector<float> x = is_txt ? txt : Slice(img, static_cast<size_t>(c - 1) * kImgChunk * kHidden, kImgChunk * kHidden);
    std::vector<float> rope = is_txt ? pe_ctx : Slice(pe, static_cast<size_t>(c - 1) * kImgChunk * kPeStride, kImgChunk * kPeStride);
    const std::vector<int32_t> x_shape = {1,tokens,kHidden};
    const std::vector<int32_t> q_shape = {1,kHeads,tokens,kHeadDim};
    const std::vector<int32_t> rope_shape = {1,1,tokens,64,2,2};
    std::vector<std::vector<float>> result;
    if (!RunPiece(is_txt ? txt_qkv_file_ : img_qkv_file_, {&x,&rope,is_txt ? &txs : &ims,is_txt ? &txsc : &imsc}, {x_shape,rope_shape,mod_shape,mod_shape}, {q_shape,q_shape,q_shape}, &result)) return false;
    q[c]=std::move(result[0]); k[c]=std::move(result[1]); v[c]=std::move(result[2]);
  }
  for (int qi = 0; qi < kNumChunks; ++qi) {
    const int qt = sizes[qi]; const bool txt_q = qi == 0;
    const std::vector<int32_t> q_shape = {1,kHeads,qt,kHeadDim};
    const std::vector<int32_t> state_shape = {1,kHeads,qt,1};
    const std::vector<int32_t> txt_qkv_shape = {1,kHeads,kTxtTokens,kHeadDim};
    std::vector<std::vector<float>> result;
    if (!RunPiece(txt_q ? "flash_step_init_probe_512_Google_Tensor_G5.tflite" : "flash_step_init_probe_q1024_kv512_Google_Tensor_G5.tflite", {&q[qi],&k[0],&v[0]}, {q_shape,txt_qkv_shape,txt_qkv_shape}, {state_shape,state_shape,q_shape}, &result)) return false;
    std::vector<float> running_max=std::move(result[0]), running_sum=std::move(result[1]), running_out=std::move(result[2]);
    for (int ki=1; ki<kNumChunks; ++ki) {
      result.clear();
      const std::vector<int32_t> img_qkv_shape = {1,kHeads,kImgChunk,kHeadDim};
      if (!RunPiece(txt_q ? "flash_step_probe_q512_kv1024_Google_Tensor_G5.tflite" : "flash_step_probe_1024_Google_Tensor_G5.tflite", {&q[qi],&k[ki],&v[ki],&running_max,&running_sum,&running_out}, {q_shape,img_qkv_shape,img_qkv_shape,state_shape,state_shape,q_shape}, {state_shape,state_shape,q_shape}, &result)) return false;
      running_max=std::move(result[0]); running_sum=std::move(result[1]); running_out=std::move(result[2]);
    }
    attn[qi].resize(static_cast<size_t>(qt)*kHidden);
    for (int h=0; h<kHeads; ++h) for (int t=0; t<qt; ++t) for (int d=0; d<kHeadDim; ++d) attn[qi][(static_cast<size_t>(t)*kHeads+h)*kHeadDim+d] = running_out[(static_cast<size_t>(h)*qt+t)*kHeadDim+d] / running_sum[static_cast<size_t>(h)*qt+t];
    std::vector<float>().swap(q[qi]);
  }
  for (int c=0;c<kNumChunks;++c) { std::vector<float>().swap(k[c]); std::vector<float>().swap(v[c]); }
  img_out->assign(static_cast<size_t>(kImgTokens)*kHidden, 0.f); txt_out->clear();
  for (int c=0;c<kNumChunks;++c) {
    const bool is_txt=c==0; const int tokens=sizes[c];
    std::vector<float> x = is_txt ? txt : Slice(img, static_cast<size_t>(c-1)*kImgChunk*kHidden, kImgChunk*kHidden);
    const std::vector<int32_t> x_shape={1,tokens,kHidden}; std::vector<std::vector<float>> result;
    if (!RunPiece(is_txt ? txt_out_file_ : img_out_file_, {&x,&attn[c],is_txt ? &txt1g : &img1g,is_txt ? &tx2s : &im2s,is_txt ? &tx2sc : &im2sc,is_txt ? &txt2g : &img2g}, {x_shape,x_shape,mod_shape,mod_shape,mod_shape,mod_shape}, {x_shape}, &result)) return false;
    if (is_txt) *txt_out=std::move(result[0]); else std::memcpy(img_out->data()+static_cast<size_t>(c-1)*kImgChunk*kHidden,result[0].data(),result[0].size()*sizeof(float));
  }
  return true;
}

bool KleinDoubleBlockEngine::RunZeroCopyQkvToFlashProbe(
    const std::vector<float>& img, const std::vector<float>& pe,
    const std::vector<float>& img_mod1_shift, const std::vector<float>& img_mod1_scale,
    long long* direct_run_ms, float* max_abs_diff) {
  if (!env_ || img.size() != static_cast<size_t>(kImgTokens) * kHidden ||
      pe.size() != static_cast<size_t>(kImgTokens) * kPeStride ||
      img_mod1_shift.size() != kHidden || img_mod1_scale.size() != kHidden) return false;
  const std::vector<int32_t> x_shape = {1, kImgChunk, kHidden};
  const std::vector<int32_t> rope_shape = {1, 1, kImgChunk, 64, 2, 2};
  const std::vector<int32_t> qkv_shape = {1, kHeads, kImgChunk, kHeadDim};
  const std::vector<int32_t> state_shape = {1, kHeads, kImgChunk, 1};
  const std::vector<int32_t> mod_shape = {1, 1, kHidden};
  std::vector<float> x = Slice(img, 0, kImgChunk * kHidden);
  std::vector<float> rope = Slice(pe, 0, kImgChunk * kPeStride);
  std::vector<std::vector<float>> qkv;
  if (!RunPiece(img_qkv_file_,
                {&x, &rope, &img_mod1_shift, &img_mod1_scale},
                {x_shape, rope_shape, mod_shape, mod_shape}, {qkv_shape, qkv_shape, qkv_shape}, &qkv)) return false;

  // Compile/allocate the consumer once through the normal, reference path.
  std::vector<std::vector<float>> copied_result;
  std::vector<float> running_max(static_cast<size_t>(kHeads) * kImgChunk, -10.f);
  std::vector<float> running_sum(static_cast<size_t>(kHeads) * kImgChunk, 1.f);
  std::vector<float> running_out(static_cast<size_t>(kHeads) * kImgChunk * kHeadDim, 0.f);
  if (!RunPiece("flash_step_probe_1024_Google_Tensor_G5.tflite",
                {&qkv[0], &qkv[1], &qkv[2], &running_max, &running_sum, &running_out},
                {qkv_shape, qkv_shape, qkv_shape, state_shape, state_shape, qkv_shape},
                {state_shape, state_shape, qkv_shape}, &copied_result)) return false;

  CachedPiece* producer = cached_pieces_[img_qkv_file_].get();
  CachedPiece* consumer = cached_pieces_["flash_step_probe_1024_Google_Tensor_G5.tflite"].get();
  if (!producer || !consumer || producer->output_buffers.size() != 3 || consumer->output_buffers.size() != 3) return false;
  const auto start = std::chrono::steady_clock::now();
  std::vector<LiteRtTensorBuffer> direct_inputs = producer->output_buffers;
  direct_inputs.insert(direct_inputs.end(), consumer->input_buffers.begin() + 3, consumer->input_buffers.end());
  const LiteRtStatus status = LiteRtRunCompiledModel(consumer->compiled.value, 0,
      direct_inputs.size(), direct_inputs.data(),
      consumer->output_buffers.size(), consumer->output_buffers.data());
  *direct_run_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("KleinDoubleBlockEngine zero-copy qkv->flash: LiteRtRunCompiledModel failed %d", status);
    return false;
  }
  *max_abs_diff = 0.f;
  for (size_t output = 0; output < consumer->output_holders.size(); ++output) {
    void* memory = nullptr;
    if (LiteRtLockTensorBuffer(consumer->output_holders[output].value, &memory, kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk) return false;
    const float* values = static_cast<const float*>(memory);
    for (size_t i = 0; i < copied_result[output].size(); ++i) {
      const float diff = std::fabs(values[i] - copied_result[output][i]);
      if (diff > *max_abs_diff) *max_abs_diff = diff;
    }
    LiteRtUnlockTensorBuffer(consumer->output_holders[output].value);
  }
  return true;
}

bool KleinDoubleBlockEngine::forwardZeroCopy(
    const std::vector<float>& img, const std::vector<float>& txt, const std::vector<float>& pe,
    const std::vector<float>& pe_ctx, const std::vector<float>& img_mod1_shift,
    const std::vector<float>& img_mod1_scale, const std::vector<float>& img_mod1_gate,
    const std::vector<float>& img_mod2_shift, const std::vector<float>& img_mod2_scale,
    const std::vector<float>& img_mod2_gate, const std::vector<float>& txt_mod1_shift,
    const std::vector<float>& txt_mod1_scale, const std::vector<float>& txt_mod1_gate,
    const std::vector<float>& txt_mod2_shift, const std::vector<float>& txt_mod2_scale,
    const std::vector<float>& txt_mod2_gate, std::vector<float>* img_out, std::vector<float>* txt_out) {
  if (!env_) return false;
  const int sizes[kNumChunks] = {kTxtTokens, kImgChunk, kImgChunk, kImgChunk, kImgChunk};
  const std::vector<int32_t> mod_shape = {1, 1, kHidden};
  const std::string& qkv_txt = txt_qkv_file_;
  const std::string& qkv_img = img_qkv_file_;
  const std::string init_txt = "flash_step_init_probe_512_Google_Tensor_G5.tflite";
  const std::string init_img = "flash_step_init_probe_q1024_kv512_Google_Tensor_G5.tflite";
  const std::string flash_txt = "flash_step_probe_q512_kv1024_Google_Tensor_G5.tflite";
  const std::string flash_img = "flash_step_probe_1024_Google_Tensor_G5.tflite";
  auto write_inputs = [](CachedPiece* piece, const std::vector<const std::vector<float>*>& values) -> bool {
    for (size_t i=0;i<values.size();++i) { void* memory=nullptr; if (LiteRtLockTensorBuffer(piece->input_holders[i].value,&memory,kLiteRtTensorBufferLockModeWrite)!=kLiteRtStatusOk) return false; std::memcpy(memory,values[i]->data(),values[i]->size()*sizeof(float)); LiteRtUnlockTensorBuffer(piece->input_holders[i].value); } return true;
  };

  // Each producer gets a separate cache key, hence a distinct persistent output-DmaBuf set.
  for (int c = 0; c < kNumChunks; ++c) {
    const bool is_txt = c == 0; const int tokens = sizes[c];
    std::vector<float> x = is_txt ? txt : Slice(img, static_cast<size_t>(c - 1) * kImgChunk * kHidden, kImgChunk * kHidden);
    std::vector<float> rope = is_txt ? pe_ctx : Slice(pe, static_cast<size_t>(c - 1) * kImgChunk * kPeStride, kImgChunk * kPeStride);
    const std::vector<int32_t> x_shape = {1, tokens, kHidden};
    const std::vector<int32_t> rope_shape = {1, 1, tokens, 64, 2, 2};
    const std::vector<int32_t> q_shape = {1, kHeads, tokens, kHeadDim};
    const std::string key = std::string("zc_qkv_") + std::to_string(c);
    CachedPiece* piece=PrepareCachedPiece(is_txt ? qkv_txt : qkv_img,{x_shape,rope_shape,mod_shape,mod_shape},{q_shape,q_shape,q_shape},key);
    if (!piece || !write_inputs(piece,{&x,&rope,is_txt ? &txt_mod1_shift : &img_mod1_shift,is_txt ? &txt_mod1_scale : &img_mod1_scale}) || !RunCachedDirect(key,piece->input_buffers)) return false;
  }

  std::vector<float> attn[kNumChunks];
  for (int qi = 0; qi < kNumChunks; ++qi) {
    const bool txt_q = qi == 0; const int qt = sizes[qi];
    const std::vector<int32_t> q_shape = {1, kHeads, qt, kHeadDim};
    const std::vector<int32_t> state_shape = {1, kHeads, qt, 1};
    const std::vector<int32_t> txt_shape = {1, kHeads, kTxtTokens, kHeadDim};
    const std::vector<int32_t> img_shape = {1, kHeads, kImgChunk, kHeadDim};
    const std::string producer_key = std::string("zc_qkv_") + std::to_string(qi);
    CachedPiece* producer = cached_pieces_[producer_key].get();
    if (!producer) return false;
    const std::string init_key = std::string("zc_init_") + std::to_string(qi);
    CachedPiece* init_piece=PrepareCachedPiece(txt_q ? init_txt : init_img,{q_shape,txt_shape,txt_shape},{state_shape,state_shape,q_shape},init_key);
    if (!init_piece) return false;
    std::vector<LiteRtTensorBuffer> init_inputs = {producer->output_buffers[0], cached_pieces_["zc_qkv_0"]->output_buffers[1], cached_pieces_["zc_qkv_0"]->output_buffers[2]};
    if (!RunCachedDirect(init_key, init_inputs)) return false;
    CachedPiece* state_piece = init_piece;
    for (int ki = 1; ki < kNumChunks; ++ki) {
      const std::string step_key = std::string("zc_flash_") + std::to_string(qi) + "_" + std::to_string(ki & 1);
      CachedPiece* next_piece=PrepareCachedPiece(txt_q ? flash_txt : flash_img,{q_shape,img_shape,img_shape,state_shape,state_shape,q_shape},{state_shape,state_shape,q_shape},step_key);
      if (!next_piece) return false;
      std::vector<LiteRtTensorBuffer> step_inputs = {producer->output_buffers[0], cached_pieces_[std::string("zc_qkv_") + std::to_string(ki)]->output_buffers[1], cached_pieces_[std::string("zc_qkv_") + std::to_string(ki)]->output_buffers[2], state_piece->output_buffers[0], state_piece->output_buffers[1], state_piece->output_buffers[2]};
      if (!RunCachedDirect(step_key, step_inputs)) return false;
      state_piece = next_piece;
    }
    std::vector<float> sum(state_piece->output_sizes[1]), out(state_piece->output_sizes[2]);
    void* memory = nullptr;
    if (LiteRtLockTensorBuffer(state_piece->output_holders[1].value, &memory, kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk) return false;
    std::memcpy(sum.data(), memory, sum.size() * sizeof(float)); LiteRtUnlockTensorBuffer(state_piece->output_holders[1].value);
    if (LiteRtLockTensorBuffer(state_piece->output_holders[2].value, &memory, kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk) return false;
    std::memcpy(out.data(), memory, out.size() * sizeof(float)); LiteRtUnlockTensorBuffer(state_piece->output_holders[2].value);
    attn[qi].resize(static_cast<size_t>(qt) * kHidden);
    for (int h=0; h<kHeads; ++h) for (int t=0; t<qt; ++t) for (int d=0; d<kHeadDim; ++d) attn[qi][(static_cast<size_t>(t)*kHeads+h)*kHeadDim+d] = out[(static_cast<size_t>(h)*qt+t)*kHeadDim+d] / sum[static_cast<size_t>(h)*qt+t];
  }
  // Pass 3 remains the trusted host-fed projection until normalization/transpose moves on NPU.
  img_out->assign(static_cast<size_t>(kImgTokens)*kHidden, 0.f); txt_out->clear();
  for (int c=0;c<kNumChunks;++c) {
    const bool is_txt=c==0; const int tokens=sizes[c]; const std::vector<int32_t> x_shape={1,tokens,kHidden};
    std::vector<float> x = is_txt ? txt : Slice(img, static_cast<size_t>(c-1)*kImgChunk*kHidden,kImgChunk*kHidden);
    std::vector<std::vector<float>> result;
    if (!RunPiece(is_txt ? txt_out_file_ : img_out_file_, {&x,&attn[c],is_txt ? &txt_mod1_gate : &img_mod1_gate,is_txt ? &txt_mod2_shift : &img_mod2_shift,is_txt ? &txt_mod2_scale : &img_mod2_scale,is_txt ? &txt_mod2_gate : &img_mod2_gate}, {x_shape,x_shape,mod_shape,mod_shape,mod_shape,mod_shape}, {x_shape}, &result)) return false;
    if (is_txt) *txt_out=std::move(result[0]); else std::memcpy(img_out->data()+static_cast<size_t>(c-1)*kImgChunk*kHidden,result[0].data(),result[0].size()*sizeof(float));
  }
  return true;
}

bool KleinDoubleBlockEngine::forwardZeroCopyPooled(
    const std::vector<float>& img, const std::vector<float>& txt, const std::vector<float>& pe, const std::vector<float>& pe_ctx,
    const std::vector<float>& ims, const std::vector<float>& imsc, const std::vector<float>& img1g, const std::vector<float>& im2s,
    const std::vector<float>& im2sc, const std::vector<float>& img2g, const std::vector<float>& txs, const std::vector<float>& txsc,
    const std::vector<float>& txt1g, const std::vector<float>& tx2s, const std::vector<float>& tx2sc, const std::vector<float>& txt2g,
    std::vector<float>* img_out, std::vector<float>* txt_out) {
  if (!env_) return false;
  const int chunk_sizes[kNumChunks] = {kTxtTokens, kImgChunk, kImgChunk, kImgChunk, kImgChunk};
  const std::vector<int32_t> mod_shape = {1, 1, kHidden};

  auto allocate_outputs = [&](CachedPiece* piece,
                              const std::vector<std::vector<int32_t>>& shapes,
                              BufferSet* set) -> bool {
    set->holders.resize(shapes.size());
    set->buffers.resize(shapes.size());
    set->sizes.resize(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
      LiteRtTensorBufferRequirements req = nullptr;
      if (LiteRtGetCompiledModelOutputBufferRequirements(piece->compiled.value, 0, i, &req) != kLiteRtStatusOk) return false;
      size_t bytes = 0;
      LiteRtGetTensorBufferRequirementsBufferSize(req, &bytes);
      LiteRtRankedTensorType type = FloatType(shapes[i]);
      if (LiteRtCreateManagedTensorBuffer(env_, kLiteRtTensorBufferTypeDmaBuf, &type, bytes, &set->holders[i].value) != kLiteRtStatusOk) return false;
      set->buffers[i] = set->holders[i].value;
      set->sizes[i] = ElementCount(shapes[i]);
    }
    return true;
  };
  auto write_input = [](CachedPiece* piece, size_t index, const std::vector<float>& value) -> bool {
    void* memory = nullptr;
    if (LiteRtLockTensorBuffer(piece->input_holders[index].value, &memory, kLiteRtTensorBufferLockModeWrite) != kLiteRtStatusOk) return false;
    std::memcpy(memory, value.data(), value.size() * sizeof(float));
    LiteRtUnlockTensorBuffer(piece->input_holders[index].value);
    return true;
  };
  auto write_inputs = [](CachedPiece* piece, const std::vector<const std::vector<float>*>& values) -> bool {
    for (size_t i = 0; i < values.size(); ++i) {
      void* memory = nullptr;
      if (LiteRtLockTensorBuffer(piece->input_holders[i].value, &memory, kLiteRtTensorBufferLockModeWrite) != kLiteRtStatusOk) return false;
      std::memcpy(memory, values[i]->data(), values[i]->size() * sizeof(float));
      LiteRtUnlockTensorBuffer(piece->input_holders[i].value);
    }
    return true;
  };

  BufferSet qkv_sets[kNumChunks];
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    const bool is_text = chunk == 0;
    const int tokens = chunk_sizes[chunk];
    const std::string key = is_text ? "pool_qtxt" : "pool_qimg";
    std::vector<float> x_chunk = is_text ? txt : Slice(img, static_cast<size_t>(chunk - 1) * kImgChunk * kHidden, kImgChunk * kHidden);
    std::vector<float> pe_chunk = is_text ? pe_ctx : Slice(pe, static_cast<size_t>(chunk - 1) * kImgChunk * kPeStride, kImgChunk * kPeStride);
    const std::vector<int32_t> x_shape = {1, tokens, kHidden};
    const std::vector<int32_t> pe_shape = {1, 1, tokens, 64, 2, 2};
    const std::vector<int32_t> qkv_shape = {1, kHeads, tokens, kHeadDim};
    CachedPiece* piece = PrepareCachedPiece(is_text ? txt_qkv_file_ : img_qkv_file_, {x_shape, pe_shape, mod_shape, mod_shape}, {qkv_shape, qkv_shape, qkv_shape}, key);
    if (!piece || !allocate_outputs(piece, {qkv_shape, qkv_shape, qkv_shape}, &qkv_sets[chunk]) ||
        !write_inputs(piece, {&x_chunk, &pe_chunk, is_text ? &txs : &ims, is_text ? &txsc : &imsc}) ||
        !RunCachedDirect(key, piece->input_buffers, &qkv_sets[chunk].buffers)) return false;
  }

  BufferSet final_attention[kNumChunks];
  for (int query_chunk = 0; query_chunk < kNumChunks; ++query_chunk) {
    const bool is_text = query_chunk == 0;
    const int tokens = chunk_sizes[query_chunk];
    const std::vector<int32_t> qkv_shape = {1, kHeads, tokens, kHeadDim};
    const std::vector<int32_t> state_shape = {1, kHeads, tokens, 1};
    const std::vector<int32_t> text_qkv_shape = {1, kHeads, kTxtTokens, kHeadDim};
    const std::vector<int32_t> image_qkv_shape = {1, kHeads, kImgChunk, kHeadDim};
    const std::string init_key = is_text ? "pool_it" : "pool_ii";
    const std::string flash_key = is_text ? "pool_ft" : "pool_fi";
    CachedPiece* init = PrepareCachedPiece(is_text ? "flash_step_init_probe_512_Google_Tensor_G5.tflite" : "flash_step_init_probe_q1024_kv512_Google_Tensor_G5.tflite", {qkv_shape, text_qkv_shape, text_qkv_shape}, {state_shape, state_shape, qkv_shape}, init_key);
    CachedPiece* flash = PrepareCachedPiece(is_text ? "flash_step_probe_q512_kv1024_Google_Tensor_G5.tflite" : "flash_step_probe_1024_Google_Tensor_G5.tflite", {qkv_shape, image_qkv_shape, image_qkv_shape, state_shape, state_shape, qkv_shape}, {state_shape, state_shape, qkv_shape}, flash_key);
    BufferSet state_a, state_b;
    if (!init || !flash || !allocate_outputs(init, {state_shape, state_shape, qkv_shape}, &state_a) || !allocate_outputs(flash, {state_shape, state_shape, qkv_shape}, &state_b)) return false;
    if (!RunCachedDirect(init_key, {qkv_sets[query_chunk].buffers[0], qkv_sets[0].buffers[1], qkv_sets[0].buffers[2]}, &state_a.buffers)) return false;
    BufferSet* current = &state_a;
    for (int key_chunk = 1; key_chunk < kNumChunks; ++key_chunk) {
      BufferSet* next = current == &state_a ? &state_b : &state_a;
      if (!RunCachedDirect(flash_key, {qkv_sets[query_chunk].buffers[0], qkv_sets[key_chunk].buffers[1], qkv_sets[key_chunk].buffers[2], current->buffers[0], current->buffers[1], current->buffers[2]}, &next->buffers)) return false;
      current = next;
    }
    const std::vector<int32_t> attention_shape = {1, tokens, kHidden};
    const std::string finalize_key = is_text ? "pool_finalize_text" : "pool_finalize_img";
    CachedPiece* finalize = PrepareCachedPiece(
        is_text ? "attn_finalize_probe_512_Google_Tensor_G5.tflite" : "attn_finalize_probe_1024_Google_Tensor_G5.tflite",
        {state_shape, qkv_shape}, {attention_shape}, finalize_key);
    if (!finalize || !allocate_outputs(finalize, {attention_shape}, &final_attention[query_chunk]) ||
        !RunCachedDirect(finalize_key, {current->buffers[1], current->buffers[2]}, &final_attention[query_chunk].buffers)) return false;
  }

  img_out->assign(static_cast<size_t>(kImgTokens) * kHidden, 0.f);
  txt_out->clear();
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    const bool is_text = chunk == 0; const int tokens = chunk_sizes[chunk];
    std::vector<float> x_chunk = is_text ? txt : Slice(img, static_cast<size_t>(chunk - 1) * kImgChunk * kHidden, kImgChunk * kHidden);
    const std::vector<int32_t> x_shape = {1, tokens, kHidden};
    const std::string out_key = is_text ? "pool_out_text" : "pool_out_img";
    CachedPiece* out_piece = PrepareCachedPiece(
        is_text ? txt_out_file_ : img_out_file_,
        {x_shape, x_shape, mod_shape, mod_shape, mod_shape, mod_shape}, {x_shape}, out_key);
    if (!out_piece ||
        !write_input(out_piece, 0, x_chunk) ||
        !write_input(out_piece, 2, is_text ? txt1g : img1g) ||
        !write_input(out_piece, 3, is_text ? tx2s : im2s) ||
        !write_input(out_piece, 4, is_text ? tx2sc : im2sc) ||
        !write_input(out_piece, 5, is_text ? txt2g : img2g) ||
        !RunCachedDirect(out_key, {out_piece->input_buffers[0], final_attention[chunk].buffers[0], out_piece->input_buffers[2], out_piece->input_buffers[3], out_piece->input_buffers[4], out_piece->input_buffers[5]})) return false;
    void* memory = nullptr;
    if (LiteRtLockTensorBuffer(out_piece->output_holders[0].value, &memory, kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk) return false;
    if (is_text) {
      txt_out->resize(out_piece->output_sizes[0]);
      std::memcpy(txt_out->data(), memory, txt_out->size() * sizeof(float));
    } else {
      std::memcpy(img_out->data() + static_cast<size_t>(chunk - 1) * kImgChunk * kHidden, memory, out_piece->output_sizes[0] * sizeof(float));
    }
    LiteRtUnlockTensorBuffer(out_piece->output_holders[0].value);
  }
  return true;
}
}  // namespace pockettavern
