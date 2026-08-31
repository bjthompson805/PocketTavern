// Production FLUX.2 [klein] end-to-end image generation: prompt -> on-device Qwen3-4B text
// encoding (CPU, fp16) -> 4-step NPU denoising -> staged NPU VAE decode -> PNG. Orchestrates the
// already-validated pieces (QwenTextEncoderEngine, KleinComponentEngine, KleinTransformerEngine)
// end to end, entirely in memory (no intermediate debug files) for a real prompt and seed.
//
// This deliberately does NOT share code with the diagnostic JNI entry points in jni_diffusion.cpp
// (nativeRunKleinOneStepReference / nativeDecodeKleinReferenceLatent) -- those stay untouched as
// validated reference tools; this engine re-implements the same schedule/RoPE math as a clean,
// callable orchestrator instead of risking a refactor of already-proven code.
#ifndef POCKETTAVERN_KLEIN_DIFFUSION_ENGINE_HPP
#define POCKETTAVERN_KLEIN_DIFFUSION_ENGINE_HPP

#include <cstdint>
#include <functional>
#include <string>

#include "npu/QwenTextEncoderEngine.hpp"

namespace pockettavern {

class KleinDiffusionEngine {
 public:
  KleinDiffusionEngine() = default;
  ~KleinDiffusionEngine() = default;

  KleinDiffusionEngine(const KleinDiffusionEngine&) = delete;
  KleinDiffusionEngine& operator=(const KleinDiffusionEngine&) = delete;

  // npu_model_dir: flat directory of *_Google_Tensor_G5.tflite files plus qk_norm_scales/
  // (see scripts/stage_klein_npu_artifacts.sh); dispatch_lib_dir: the app's own native library
  // directory (bundles the gated Tensor NPU dispatch .so, already part of the APK -- not
  // user-staged); qwen_config_path: the fp16 Qwen3-4B encoder's exported config.json. Loads the
  // (large, ~8GB) text encoder now so it's ready for Generate(); the NPU pieces are loaded lazily
  // per-component the same way the diagnostic path already does.
  bool Load(const std::string& npu_model_dir, const std::string& dispatch_lib_dir,
            const std::string& qwen_config_path, const std::string& mmap_cache_dir);

  // seed must be non-negative -- callers resolve a "random seed" request (e.g. seed < 0 in the
  // UI) to a concrete value before calling down into native code. progress_callback(percent) is
  // invoked synchronously, on the calling thread, at each of the pipeline's 13 phases (1 text
  // encode + 4 denoising steps + 8 VAE stages). Writes a real PNG to output_png_path on success.
  bool Generate(const std::string& prompt, uint32_t seed, const std::string& output_png_path,
                const std::function<void(int)>& progress_callback);

 private:
  std::string npu_model_dir_;
  std::string dispatch_lib_dir_;
  QwenTextEncoderEngine encoder_;
};

}  // namespace pockettavern

#endif  // POCKETTAVERN_KLEIN_DIFFUSION_ENGINE_HPP
