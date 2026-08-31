#ifndef POCKETTAVERN_KLEIN_COMPONENT_ENGINE_HPP
#define POCKETTAVERN_KLEIN_COMPONENT_ENGINE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "litert/c/litert_environment.h"

namespace pockettavern {

// Executes one fixed-shape float32 FLUX.2 [klein] AOT component.  The runner owns only its
// LiteRT environment; model and DmaBuf handles are scoped to Run(), so input/modulation/final
// components do not accumulate device allocations while a denoising step progresses.
class KleinComponentEngine {
 public:
  KleinComponentEngine() = default;
  ~KleinComponentEngine();

  KleinComponentEngine(const KleinComponentEngine&) = delete;
  KleinComponentEngine& operator=(const KleinComponentEngine&) = delete;

  bool Load(const std::string& dispatch_lib_dir);
  bool Run(const std::string& model_path,
           const std::vector<const std::vector<float>*>& inputs,
           const std::vector<std::vector<int32_t>>& input_shapes,
           const std::vector<int32_t>& output_shape,
           std::vector<float>* output) const;
  bool RunMulti(const std::string& model_path,
                const std::vector<const std::vector<float>*>& inputs,
                const std::vector<std::vector<int32_t>>& input_shapes,
                const std::vector<std::vector<int32_t>>& output_shapes,
                std::vector<std::vector<float>>* outputs) const;

 private:
  LiteRtEnvironment env_ = nullptr;
};

}  // namespace pockettavern

#endif  // POCKETTAVERN_KLEIN_COMPONENT_ENGINE_HPP
