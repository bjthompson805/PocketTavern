#include "npu/KleinTransformerEngine.hpp"

#include <android/log.h>

#include "npu/KleinDoubleBlockEngine.hpp"
#include "npu/KleinSingleBlockEngine.hpp"

namespace pockettavern {

namespace {
constexpr size_t kHidden = 3072;
constexpr size_t kImageTokens = 4096;
constexpr size_t kTextTokens = 512;
constexpr size_t kPeStride = 256;

bool ValidModulation(const KleinSingleModulation& modulation) {
  return modulation.shift.size() == kHidden && modulation.scale.size() == kHidden &&
         modulation.gate.size() == kHidden;
}

void LogError(const char* message, int block) {
  __android_log_print(ANDROID_LOG_ERROR, "PocketTavernDiffusion", message, block);
}
}  // namespace

bool KleinTransformerEngine::Forward(
    const std::string& model_dir, const std::string& dispatch_lib_dir,
    const std::vector<float>& pe, const std::vector<float>& pe_ctx,
    const std::array<KleinDoubleModulation, 5>& double_modulations,
    const std::array<KleinSingleModulation, 20>& single_modulations,
    std::vector<float>* img, std::vector<float>* txt, int single_worker_count) const {
  if (!img || !txt || img->size() != kImageTokens * kHidden || txt->size() != kTextTokens * kHidden ||
      pe.size() != kImageTokens * kPeStride || pe_ctx.size() != kTextTokens * kPeStride ||
      single_worker_count < 1 || single_worker_count > 4) {
    return false;
  }
  for (const KleinDoubleModulation& modulation : double_modulations) {
    if (!ValidModulation(modulation.image_first) || !ValidModulation(modulation.image_second) ||
        !ValidModulation(modulation.text_first) || !ValidModulation(modulation.text_second)) return false;
  }
  for (const KleinSingleModulation& modulation : single_modulations) {
    if (!ValidModulation(modulation)) return false;
  }

  for (int block = 0; block < 5; ++block) {
    // Scope deliberately bounds both NPU compiled handles and DmaBufs to one block.
    KleinDoubleBlockEngine engine;
    if (!engine.Load(model_dir, dispatch_lib_dir, block)) {
      LogError("KleinTransformerEngine: failed to load double block %d", block);
      return false;
    }
    const KleinDoubleModulation& modulation = double_modulations[block];
    std::vector<float> next_img;
    std::vector<float> next_txt;
    if (!engine.forwardZeroCopyPooled(
            *img, *txt, pe, pe_ctx,
            modulation.image_first.shift, modulation.image_first.scale, modulation.image_first.gate,
            modulation.image_second.shift, modulation.image_second.scale, modulation.image_second.gate,
            modulation.text_first.shift, modulation.text_first.scale, modulation.text_first.gate,
            modulation.text_second.shift, modulation.text_second.scale, modulation.text_second.gate,
            &next_img, &next_txt)) {
      LogError("KleinTransformerEngine: double block %d failed", block);
      return false;
    }
    img->swap(next_img);
    txt->swap(next_txt);
  }

  std::vector<float> combined;
  combined.reserve((kTextTokens + kImageTokens) * kHidden);
  combined.insert(combined.end(), txt->begin(), txt->end());
  combined.insert(combined.end(), img->begin(), img->end());
  std::vector<float> combined_pe;
  combined_pe.reserve((kTextTokens + kImageTokens) * kPeStride);
  combined_pe.insert(combined_pe.end(), pe_ctx.begin(), pe_ctx.end());
  combined_pe.insert(combined_pe.end(), pe.begin(), pe.end());

  for (int block = 0; block < 20; ++block) {
    KleinSingleBlockEngine engine;
    if (!engine.Load(model_dir, dispatch_lib_dir, block)) {
      LogError("KleinTransformerEngine: failed to load single block %d", block);
      return false;
    }
    std::vector<float> next_combined;
    const KleinSingleModulation& modulation = single_modulations[block];
    if (!engine.forwardZeroCopyPooledWithWorkers(combined, combined_pe, modulation.shift,
                                                 modulation.scale, modulation.gate,
                                                 &next_combined, single_worker_count)) {
      LogError("KleinTransformerEngine: single block %d failed", block);
      return false;
    }
    combined.swap(next_combined);
  }

  txt->assign(combined.begin(), combined.begin() + kTextTokens * kHidden);
  img->assign(combined.begin() + kTextTokens * kHidden, combined.end());
  return true;
}

}  // namespace pockettavern
