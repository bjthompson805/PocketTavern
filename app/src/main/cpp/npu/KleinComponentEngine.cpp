#include "npu/KleinComponentEngine.hpp"

#include <android/log.h>

#include <cstring>
#include <memory>

#include "litert/c/litert_any.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"

namespace pockettavern {
namespace {
constexpr char kTag[] = "PocketTavernDiffusion";

size_t ElementCount(const std::vector<int32_t>& shape) {
  size_t count = 1;
  for (int32_t dim : shape) {
    if (dim <= 0) return 0;
    count *= static_cast<size_t>(dim);
  }
  return count;
}

LiteRtRankedTensorType FloatType(const std::vector<int32_t>& shape) {
  LiteRtRankedTensorType type{};
  type.element_type = kLiteRtElementTypeFloat32;
  type.layout.rank = static_cast<unsigned int>(shape.size());
  type.layout.has_strides = 0;
  for (size_t i = 0; i < shape.size(); ++i) type.layout.dimensions[i] = shape[i];
  return type;
}

struct ModelHolder { LiteRtModel value = nullptr; ~ModelHolder() { if (value) LiteRtDestroyModel(value); } };
struct OptionsHolder { LiteRtOptions value = nullptr; ~OptionsHolder() { if (value) LiteRtDestroyOptions(value); } };
struct CompiledHolder { LiteRtCompiledModel value = nullptr; ~CompiledHolder() { if (value) LiteRtDestroyCompiledModel(value); } };
struct BufferHolder { LiteRtTensorBuffer value = nullptr; ~BufferHolder() { if (value) LiteRtDestroyTensorBuffer(value); } };
}  // namespace

KleinComponentEngine::~KleinComponentEngine() {
  if (env_) LiteRtDestroyEnvironment(env_);
}

bool KleinComponentEngine::Load(const std::string& dispatch_lib_dir) {
  if (env_) return false;
  LiteRtEnvOption option{};
  option.tag = kLiteRtEnvOptionTagDispatchLibraryDir;
  LiteRtAny value{};
  value.type = kLiteRtAnyTypeString;
  value.str_value = dispatch_lib_dir.c_str();
  option.value = value;
  if (LiteRtCreateEnvironment(1, &option, &env_) != kLiteRtStatusOk) {
    __android_log_print(ANDROID_LOG_ERROR, kTag, "KleinComponentEngine: environment creation failed");
    env_ = nullptr;
    return false;
  }
  return true;
}

bool KleinComponentEngine::Run(const std::string& model_path,
                               const std::vector<const std::vector<float>*>& inputs,
                               const std::vector<std::vector<int32_t>>& input_shapes,
                               const std::vector<int32_t>& output_shape,
                               std::vector<float>* output) const {
  if (!output) return false;
  std::vector<std::vector<float>> outputs;
  if (!RunMulti(model_path, inputs, input_shapes, {output_shape}, &outputs) || outputs.size() != 1) return false;
  *output = std::move(outputs[0]);
  return true;
}

bool KleinComponentEngine::RunMulti(const std::string& model_path,
                                    const std::vector<const std::vector<float>*>& inputs,
                                    const std::vector<std::vector<int32_t>>& input_shapes,
                                    const std::vector<std::vector<int32_t>>& output_shapes,
                                    std::vector<std::vector<float>>* outputs) const {
  if (!env_ || !outputs || inputs.empty() || inputs.size() != input_shapes.size() || output_shapes.empty()) return false;
  for (const auto& shape : output_shapes) if (ElementCount(shape) == 0) return false;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i] || inputs[i]->size() != ElementCount(input_shapes[i])) return false;
  }

  ModelHolder model;
  if (LiteRtCreateModelFromFile(env_, model_path.c_str(), &model.value) != kLiteRtStatusOk) return false;
  OptionsHolder options;
  if (LiteRtCreateOptions(&options.value) != kLiteRtStatusOk ||
      LiteRtSetOptionsHardwareAccelerators(options.value,
          kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu) != kLiteRtStatusOk) return false;
  CompiledHolder compiled;
  if (LiteRtCreateCompiledModel(env_, model.value, options.value, &compiled.value) != kLiteRtStatusOk) return false;

  std::vector<BufferHolder> input_holders(inputs.size());
  std::vector<LiteRtTensorBuffer> input_buffers(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    LiteRtTensorBufferRequirements requirements = nullptr;
    if (LiteRtGetCompiledModelInputBufferRequirements(compiled.value, 0, i, &requirements) != kLiteRtStatusOk) return false;
    const LiteRtRankedTensorType type = FloatType(input_shapes[i]);
    if (LiteRtCreateManagedTensorBufferFromRequirements(env_, &type, requirements,
                                                         &input_holders[i].value) != kLiteRtStatusOk) return false;
    void* memory = nullptr;
    if (LiteRtLockTensorBuffer(input_holders[i].value, &memory, kLiteRtTensorBufferLockModeWrite) != kLiteRtStatusOk) return false;
    std::memcpy(memory, inputs[i]->data(), inputs[i]->size() * sizeof(float));
    LiteRtUnlockTensorBuffer(input_holders[i].value);
    input_buffers[i] = input_holders[i].value;
  }

  std::vector<BufferHolder> output_holders(output_shapes.size());
  std::vector<LiteRtTensorBuffer> output_buffers(output_shapes.size());
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    LiteRtTensorBufferRequirements requirements = nullptr;
    if (LiteRtGetCompiledModelOutputBufferRequirements(compiled.value, 0, i, &requirements) != kLiteRtStatusOk) return false;
    const LiteRtRankedTensorType type = FloatType(output_shapes[i]);
    if (LiteRtCreateManagedTensorBufferFromRequirements(env_, &type, requirements,
                                                         &output_holders[i].value) != kLiteRtStatusOk) return false;
    output_buffers[i] = output_holders[i].value;
  }
  if (LiteRtRunCompiledModel(compiled.value, 0, input_buffers.size(), input_buffers.data(),
                             output_buffers.size(), output_buffers.data()) != kLiteRtStatusOk) return false;
  outputs->resize(output_shapes.size());
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    (*outputs)[i].resize(ElementCount(output_shapes[i]));
    void* memory = nullptr;
    if (LiteRtLockTensorBuffer(output_holders[i].value, &memory, kLiteRtTensorBufferLockModeRead) != kLiteRtStatusOk) return false;
    std::memcpy((*outputs)[i].data(), memory, (*outputs)[i].size() * sizeof(float));
    LiteRtUnlockTensorBuffer(output_holders[i].value);
  }
  return true;
}

}  // namespace pockettavern
