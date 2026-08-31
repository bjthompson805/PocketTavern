// Memory-bounded FLUX.2 [klein] transformer runner.
//
// It intentionally owns one block engine at a time.  Every block engine is destroyed before
// advancing to the next block, so its compiled LiteRT models and all DmaBuf allocations are
// released.  This is the production-safe baseline while we establish the Tensor G5 compiled-model
// memory limit for all 25 blocks.  Inputs are the projected transformer tensors/modulations; the
// text encoder, input projections, time/modulation networks, final layer, and VAE are separate
// pipeline components and are not hidden behind this class.
#ifndef POCKETTAVERN_KLEIN_TRANSFORMER_ENGINE_HPP
#define POCKETTAVERN_KLEIN_TRANSFORMER_ENGINE_HPP

#include <array>
#include <string>
#include <vector>

namespace pockettavern {

struct KleinSingleModulation {
  std::vector<float> shift;
  std::vector<float> scale;
  std::vector<float> gate;
};

struct KleinDoubleModulation {
  KleinSingleModulation image_first;
  KleinSingleModulation image_second;
  KleinSingleModulation text_first;
  KleinSingleModulation text_second;
};

class KleinTransformerEngine {
 public:
  // model_dir is the flat directory produced by scripts/stage_klein_npu_artifacts.sh. Executes
  // 5 double blocks followed by 20 single blocks. img is [4096,3072], txt is
  // [512,3072], pe is [4096,256], and pe_ctx is [512,256], all flattened float32 tensors.
  // On success img contains the final 4096-token hidden state. txt is the final double-stack
  // state (single blocks operate on concat(txt,img) internally).  The caller owns final-layer
  // processing and removes the text tokens as required by the Flux architecture.
  bool Forward(const std::string& model_dir, const std::string& dispatch_lib_dir,
               const std::vector<float>& pe, const std::vector<float>& pe_ctx,
               const std::array<KleinDoubleModulation, 5>& double_modulations,
               const std::array<KleinSingleModulation, 20>& single_modulations,
               std::vector<float>* img, std::vector<float>* txt,
               int single_worker_count = 4) const;
};

}  // namespace pockettavern

#endif  // POCKETTAVERN_KLEIN_TRANSFORMER_ENGINE_HPP
