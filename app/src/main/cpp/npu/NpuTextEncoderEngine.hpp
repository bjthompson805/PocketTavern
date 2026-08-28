#ifndef POCKETTAVERN_NPU_TEXT_ENCODER_ENGINE_HPP
#define POCKETTAVERN_NPU_TEXT_ENCODER_ENGINE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/c/litert_environment.h"

namespace pockettavern {

// NpuTextEncoderEngine: runs SDXL's text_encoder (CLIP-L) and text_encoder_2 (OpenCLIP-bigG)
// as LiteRT CompiledModels on the Google Tensor NPU.
//
// text_encoder is run as 1 piece (batch=2).
// text_encoder_2 is run as 8 chunks (batch=2).
//
// Output contract:
//   - out_encoder_hidden_states: [2, 77, 2048] float32 (concat of CLIP-L [2,77,768] and OpenCLIP [2,77,1280])
//   - out_text_embeds: [2, 1280] float32 (OpenCLIP-bigG pooled projection)
class NpuTextEncoderEngine {
 public:
  NpuTextEncoderEngine();
  ~NpuTextEncoderEngine();

  NpuTextEncoderEngine(const NpuTextEncoderEngine&) = delete;
  NpuTextEncoderEngine& operator=(const NpuTextEncoderEngine&) = delete;

  // Loads the environment and verifies all 9 compiled text encoder piece files exist under model_dir.
  // model_dir can either contain the piece files directly, or contain a text_encoder/ subdirectory.
  // dispatch_lib_dir is the directory containing libLiteRtDispatch_GoogleTensor.so.
  bool Load(std::string model_dir, const std::string& dispatch_lib_dir);

  bool IsLoaded() const { return env_ != nullptr; }

  // Encodes negative and positive prompts for SDXL (batch=2).
  // input_ids_1: [2, 77] int32 token IDs for CLIP-L (negative prompt row 0, positive prompt row 1)
  // input_ids_2: [2, 77] int32 token IDs for OpenCLIP-bigG (negative prompt row 0, positive prompt row 1)
  // Writes [2, 77, 2048] into out_encoder_hidden_states and [2, 1280] into out_text_embeds.
  bool encode(const std::vector<int32_t>& input_ids_1,
              const std::vector<int32_t>& input_ids_2,
              std::vector<float>* out_encoder_hidden_states,
              std::vector<float>* out_text_embeds);

 private:
  bool RunTe1(const std::vector<int32_t>& input_ids, std::vector<float>* out_hidden);
  bool RunTe2(const std::vector<int32_t>& input_ids,
              std::vector<float>* out_penultimate,
              std::vector<float>* out_text_embeds);

  std::string model_dir_;
  LiteRtEnvironment env_ = nullptr;
};

}  // namespace pockettavern

#endif  // POCKETTAVERN_NPU_TEXT_ENCODER_ENGINE_HPP
