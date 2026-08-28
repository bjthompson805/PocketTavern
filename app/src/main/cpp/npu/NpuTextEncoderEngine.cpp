#include "npu/NpuTextEncoderEngine.hpp"

#include <android/log.h>
#include <sys/stat.h>

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

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

LiteRtRankedTensorType MakeTensorType(const std::vector<int32_t>& shape, LiteRtElementType elem_type) {
  LiteRtRankedTensorType t{};
  t.element_type = elem_type;
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

struct GenericInputTensor {
  const void* data;
  size_t byte_size;
  std::vector<int32_t> shape;
  LiteRtElementType elem_type;
};

struct GenericOutputTensor {
  std::vector<int32_t> shape;
  LiteRtElementType elem_type;
  size_t elem_count;
};

bool RunGenericPiece(
    LiteRtEnvironment env,
    const std::string& model_path,
    const char* piece_name,
    const std::vector<GenericInputTensor>& inputs,
    const std::vector<GenericOutputTensor>& outputs,
    std::vector<std::vector<uint8_t>>* out_raw_bytes) {
  const auto start = std::chrono::steady_clock::now();

  std::vector<TensorBufferHolder> input_holders(inputs.size());
  std::vector<LiteRtTensorBuffer> input_buffers(inputs.size());
  std::vector<TensorBufferHolder> output_holders(outputs.size());
  std::vector<LiteRtTensorBuffer> output_buffers(outputs.size());

  ModelHolder model_holder;
  LiteRtStatus status = LiteRtCreateModelFromFile(env, model_path.c_str(), &model_holder.model);
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("NpuTextEncoderEngine: %s: LiteRtCreateModelFromFile failed, status=%d\n", piece_name, status);
    return false;
  }

  OptionsHolder options_holder;
  status = LiteRtCreateOptions(&options_holder.options);
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("NpuTextEncoderEngine: %s: LiteRtCreateOptions failed, status=%d\n", piece_name, status);
    return false;
  }
  status = LiteRtSetOptionsHardwareAccelerators(
      options_holder.options, kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu);
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("NpuTextEncoderEngine: %s: LiteRtSetOptionsHardwareAccelerators failed, status=%d\n", piece_name, status);
    return false;
  }

  CompiledModelHolder cm_holder;
  status = LiteRtCreateCompiledModel(env, model_holder.model, options_holder.options,
                                      &cm_holder.compiled_model);
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("NpuTextEncoderEngine: %s: LiteRtCreateCompiledModel failed, status=%d\n", piece_name, status);
    return false;
  }

  // Bind inputs
  for (size_t i = 0; i < inputs.size(); ++i) {
    LiteRtTensorBufferRequirements reqs = nullptr;
    status = LiteRtGetCompiledModelInputBufferRequirements(cm_holder.compiled_model, 0, i, &reqs);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("NpuTextEncoderEngine: %s: input[%zu] reqs failed, status=%d\n", piece_name, i, status);
      return false;
    }
    LiteRtRankedTensorType tensor_type = MakeTensorType(inputs[i].shape, inputs[i].elem_type);
    status = LiteRtCreateManagedTensorBufferFromRequirements(env, &tensor_type, reqs, &input_holders[i].buffer);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("NpuTextEncoderEngine: %s: input[%zu] buffer alloc failed, status=%d\n", piece_name, i, status);
      return false;
    }
    void* host_mem = nullptr;
    status = LiteRtLockTensorBuffer(input_holders[i].buffer, &host_mem, kLiteRtTensorBufferLockModeWrite);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("NpuTextEncoderEngine: %s: input[%zu] lock failed, status=%d\n", piece_name, i, status);
      return false;
    }
    std::memcpy(host_mem, inputs[i].data, inputs[i].byte_size);
    LiteRtUnlockTensorBuffer(input_holders[i].buffer);
    input_buffers[i] = input_holders[i].buffer;
  }

  // Bind outputs
  for (size_t i = 0; i < outputs.size(); ++i) {
    LiteRtTensorBufferRequirements reqs = nullptr;
    status = LiteRtGetCompiledModelOutputBufferRequirements(cm_holder.compiled_model, 0, i, &reqs);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("NpuTextEncoderEngine: %s: output[%zu] reqs failed, status=%d\n", piece_name, i, status);
      return false;
    }
    LiteRtRankedTensorType tensor_type = MakeTensorType(outputs[i].shape, outputs[i].elem_type);
    status = LiteRtCreateManagedTensorBufferFromRequirements(env, &tensor_type, reqs, &output_holders[i].buffer);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("NpuTextEncoderEngine: %s: output[%zu] buffer alloc failed, status=%d\n", piece_name, i, status);
      return false;
    }
    output_buffers[i] = output_holders[i].buffer;
  }

  // Execute
  status = LiteRtRunCompiledModel(cm_holder.compiled_model, 0,
                                  input_buffers.size(), input_buffers.data(),
                                  output_buffers.size(), output_buffers.data());
  if (status != kLiteRtStatusOk) {
    NPU_LOGE("NpuTextEncoderEngine: %s: LiteRtRunCompiledModel failed, status=%d\n", piece_name, status);
    return false;
  }

  // Read outputs
  out_raw_bytes->resize(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    const size_t bytes = outputs[i].elem_count * (outputs[i].elem_type == kLiteRtElementTypeInt32 ? sizeof(int32_t) : sizeof(float));
    (*out_raw_bytes)[i].resize(bytes);
    const void* host_mem = nullptr;
    status = LiteRtLockTensorBuffer(output_holders[i].buffer, const_cast<void**>(&host_mem), kLiteRtTensorBufferLockModeRead);
    if (status != kLiteRtStatusOk) {
      NPU_LOGE("NpuTextEncoderEngine: %s: output[%zu] lock failed, status=%d\n", piece_name, i, status);
      return false;
    }
    std::memcpy((*out_raw_bytes)[i].data(), host_mem, bytes);
    LiteRtUnlockTensorBuffer(output_holders[i].buffer);
  }

  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - start).count();
  NPU_LOGE("NpuTextEncoderEngine: %s finished in %lld ms\n", piece_name, static_cast<long long>(elapsed_ms));
  return true;
}

}  // namespace

NpuTextEncoderEngine::NpuTextEncoderEngine() = default;

NpuTextEncoderEngine::~NpuTextEncoderEngine() {
  if (env_) {
    LiteRtDestroyEnvironment(env_);
    env_ = nullptr;
  }
}

bool NpuTextEncoderEngine::Load(std::string model_dir, const std::string& dispatch_lib_dir) {
  if (env_ != nullptr) {
    NPU_LOGE("NpuTextEncoderEngine::Load: already loaded\n");
    return false;
  }

  model_dir_ = std::move(model_dir);

  // If piece files live inside a text_encoder/ subdir, point there
  if (FileExists(model_dir_ + "/text_encoder/text_encoder_b2_wrapped.tflite")) {
    model_dir_ = model_dir_ + "/text_encoder";
  }

  // Verify all 9 piece files exist
  const char* kRequiredPieces[] = {
      "text_encoder_b2_wrapped.tflite",
      "te2_chunk0_embed_layers0_3_b2_wrapped.tflite",
      "te2_chunk1_layers4_7_b2_wrapped.tflite",
      "te2_chunk2_layers8_11_b2_wrapped.tflite",
      "te2_chunk3_layers12_15_b2_wrapped.tflite",
      "te2_chunk4_layers16_19_b2_wrapped.tflite",
      "te2_chunk5_layers20_23_b2_wrapped.tflite",
      "te2_chunk6_layers24_27_b2_wrapped.tflite",
      "te2_chunk7_layers28_31_head_b2_wrapped.tflite",
  };
  for (const char* piece_file : kRequiredPieces) {
    std::string path = model_dir_ + "/" + piece_file;
    if (!FileExists(path)) {
      NPU_LOGE("NpuTextEncoderEngine::Load: required piece missing: %s\n", path.c_str());
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
    NPU_LOGE("NpuTextEncoderEngine::Load: LiteRtCreateEnvironment failed, status=%d\n", status);
    env_ = nullptr;
    return false;
  }

  NPU_LOGE("NpuTextEncoderEngine: successfully loaded from %s\n", model_dir_.c_str());
  return true;
}

bool NpuTextEncoderEngine::RunTe1(const std::vector<int32_t>& input_ids, std::vector<float>* out_hidden) {
  if (input_ids.size() != 2 * 77) {
    NPU_LOGE("NpuTextEncoderEngine::RunTe1: expected 154 input_ids, got %zu\n", input_ids.size());
    return false;
  }
  std::string path = model_dir_ + "/text_encoder_b2_wrapped.tflite";
  std::vector<GenericInputTensor> inputs = {
      {input_ids.data(), input_ids.size() * sizeof(int32_t), {2, 77}, kLiteRtElementTypeInt32},
  };
  std::vector<GenericOutputTensor> outputs = {
      {{2, 77, 768}, kLiteRtElementTypeFloat32, 2 * 77 * 768},
  };
  std::vector<std::vector<uint8_t>> raw_out;
  if (!RunGenericPiece(env_, path, "te1_clip_l", inputs, outputs, &raw_out)) {
    return false;
  }
  out_hidden->resize(2 * 77 * 768);
  std::memcpy(out_hidden->data(), raw_out[0].data(), out_hidden->size() * sizeof(float));
  return true;
}

bool NpuTextEncoderEngine::RunTe2(const std::vector<int32_t>& input_ids,
                                  std::vector<float>* out_penultimate,
                                  std::vector<float>* out_text_embeds) {
  if (input_ids.size() != 2 * 77) {
    NPU_LOGE("NpuTextEncoderEngine::RunTe2: expected 154 input_ids, got %zu\n", input_ids.size());
    return false;
  }

  std::vector<float> h(2 * 77 * 1280);

  // Chunk 0: embed + layers 0..3
  {
    std::string path = model_dir_ + "/te2_chunk0_embed_layers0_3_b2_wrapped.tflite";
    std::vector<GenericInputTensor> inputs = {
        {input_ids.data(), input_ids.size() * sizeof(int32_t), {2, 77}, kLiteRtElementTypeInt32},
    };
    std::vector<GenericOutputTensor> outputs = {
        {{2, 77, 1280}, kLiteRtElementTypeFloat32, 2 * 77 * 1280},
    };
    std::vector<std::vector<uint8_t>> raw_out;
    if (!RunGenericPiece(env_, path, "te2_chunk0", inputs, outputs, &raw_out)) return false;
    std::memcpy(h.data(), raw_out[0].data(), h.size() * sizeof(float));
  }

  // Chunks 1..6: layers 4..27
  const char* kMiddleChunks[] = {
      "te2_chunk1_layers4_7_b2_wrapped.tflite",
      "te2_chunk2_layers8_11_b2_wrapped.tflite",
      "te2_chunk3_layers12_15_b2_wrapped.tflite",
      "te2_chunk4_layers16_19_b2_wrapped.tflite",
      "te2_chunk5_layers20_23_b2_wrapped.tflite",
      "te2_chunk6_layers24_27_b2_wrapped.tflite",
  };
  for (size_t c = 0; c < 6; ++c) {
    std::string path = model_dir_ + "/" + kMiddleChunks[c];
    char chunk_name[32];
    snprintf(chunk_name, sizeof(chunk_name), "te2_chunk%zu", c + 1);
    std::vector<GenericInputTensor> inputs = {
        {h.data(), h.size() * sizeof(float), {2, 77, 1280}, kLiteRtElementTypeFloat32},
    };
    std::vector<GenericOutputTensor> outputs = {
        {{2, 77, 1280}, kLiteRtElementTypeFloat32, 2 * 77 * 1280},
    };
    std::vector<std::vector<uint8_t>> raw_out;
    if (!RunGenericPiece(env_, path, chunk_name, inputs, outputs, &raw_out)) return false;
    std::memcpy(h.data(), raw_out[0].data(), h.size() * sizeof(float));
  }

  // Chunk 7: layers 28..31 + head
  {
    std::string path = model_dir_ + "/te2_chunk7_layers28_31_head_b2_wrapped.tflite";
    std::vector<GenericInputTensor> inputs = {
        {h.data(), h.size() * sizeof(float), {2, 77, 1280}, kLiteRtElementTypeFloat32},
        {input_ids.data(), input_ids.size() * sizeof(int32_t), {2, 77}, kLiteRtElementTypeInt32},
    };
    std::vector<GenericOutputTensor> outputs = {
        {{2, 77, 1280}, kLiteRtElementTypeFloat32, 2 * 77 * 1280},
        {{2, 1280}, kLiteRtElementTypeFloat32, 2 * 1280},
    };
    std::vector<std::vector<uint8_t>> raw_out;
    if (!RunGenericPiece(env_, path, "te2_chunk7", inputs, outputs, &raw_out)) return false;

    out_penultimate->resize(2 * 77 * 1280);
    std::memcpy(out_penultimate->data(), raw_out[0].data(), out_penultimate->size() * sizeof(float));

    out_text_embeds->resize(2 * 1280);
    std::memcpy(out_text_embeds->data(), raw_out[1].data(), out_text_embeds->size() * sizeof(float));
  }

  return true;
}

bool NpuTextEncoderEngine::encode(const std::vector<int32_t>& input_ids_1,
                                  const std::vector<int32_t>& input_ids_2,
                                  std::vector<float>* out_encoder_hidden_states,
                                  std::vector<float>* out_text_embeds) {
  if (!IsLoaded()) {
    NPU_LOGE("NpuTextEncoderEngine::encode: not loaded\n");
    return false;
  }

  std::vector<float> te1_hidden;
  if (!RunTe1(input_ids_1, &te1_hidden)) {
    NPU_LOGE("NpuTextEncoderEngine::encode: RunTe1 failed\n");
    return false;
  }

  std::vector<float> te2_penultimate;
  if (!RunTe2(input_ids_2, &te2_penultimate, out_text_embeds)) {
    NPU_LOGE("NpuTextEncoderEngine::encode: RunTe2 failed\n");
    return false;
  }

  // Concatenate along dim 2: [2, 77, 768] + [2, 77, 1280] -> [2, 77, 2048]
  out_encoder_hidden_states->resize(2 * 77 * 2048);
  for (int b = 0; b < 2; ++b) {
    for (int t = 0; t < 77; ++t) {
      const float* src1 = &te1_hidden[(b * 77 + t) * 768];
      const float* src2 = &te2_penultimate[(b * 77 + t) * 1280];
      float* dst = &(*out_encoder_hidden_states)[(b * 77 + t) * 2048];
      std::memcpy(dst, src1, 768 * sizeof(float));
      std::memcpy(dst + 768, src2, 1280 * sizeof(float));
    }
  }

  return true;
}

}  // namespace pockettavern
