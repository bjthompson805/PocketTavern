// Native performance/correctness smoke engine for FLUX.2 [klein] double_blocks.0.
// Uses the validated asymmetric chunked layout: txt@512 followed by four img@1024 chunks.
#ifndef POCKETTAVERN_KLEIN_DOUBLE_BLOCK_ENGINE_HPP
#define POCKETTAVERN_KLEIN_DOUBLE_BLOCK_ENGINE_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "litert/c/litert_environment.h"

namespace pockettavern {

class KleinDoubleBlockEngine {
 public:
  KleinDoubleBlockEngine();
  ~KleinDoubleBlockEngine();
  KleinDoubleBlockEngine(const KleinDoubleBlockEngine&) = delete;
  KleinDoubleBlockEngine& operator=(const KleinDoubleBlockEngine&) = delete;

  // block_index selects this double block's four learned projection artifacts.
  bool Load(std::string model_dir, const std::string& dispatch_lib_dir, int block_index = 0);
  bool IsLoaded() const { return env_ != nullptr; }

  // Runs double_blocks.0's complete 35-dispatch forward. Inputs and outputs are flattened
  // float32 tensors: img [4096,3072], txt [512,3072], pe/pe_ctx use stride 256.
  bool forward(const std::vector<float>& img, const std::vector<float>& txt,
               const std::vector<float>& pe, const std::vector<float>& pe_ctx,
               const std::vector<float>& img_mod1_shift, const std::vector<float>& img_mod1_scale,
               const std::vector<float>& img_mod1_gate, const std::vector<float>& img_mod2_shift,
               const std::vector<float>& img_mod2_scale, const std::vector<float>& img_mod2_gate,
               const std::vector<float>& txt_mod1_shift, const std::vector<float>& txt_mod1_scale,
               const std::vector<float>& txt_mod1_gate, const std::vector<float>& txt_mod2_shift,
               const std::vector<float>& txt_mod2_scale, const std::vector<float>& txt_mod2_gate,
               std::vector<float>* img_out, std::vector<float>* txt_out);
  bool forwardZeroCopy(const std::vector<float>& img, const std::vector<float>& txt,
                       const std::vector<float>& pe, const std::vector<float>& pe_ctx,
                       const std::vector<float>& img_mod1_shift, const std::vector<float>& img_mod1_scale,
                       const std::vector<float>& img_mod1_gate, const std::vector<float>& img_mod2_shift,
                       const std::vector<float>& img_mod2_scale, const std::vector<float>& img_mod2_gate,
                       const std::vector<float>& txt_mod1_shift, const std::vector<float>& txt_mod1_scale,
                       const std::vector<float>& txt_mod1_gate, const std::vector<float>& txt_mod2_shift,
                       const std::vector<float>& txt_mod2_scale, const std::vector<float>& txt_mod2_gate,
                       std::vector<float>* img_out, std::vector<float>* txt_out);
  bool forwardZeroCopyPooled(const std::vector<float>& img, const std::vector<float>& txt,
                             const std::vector<float>& pe, const std::vector<float>& pe_ctx,
                             const std::vector<float>& img_mod1_shift, const std::vector<float>& img_mod1_scale,
                             const std::vector<float>& img_mod1_gate, const std::vector<float>& img_mod2_shift,
                             const std::vector<float>& img_mod2_scale, const std::vector<float>& img_mod2_gate,
                             const std::vector<float>& txt_mod1_shift, const std::vector<float>& txt_mod1_scale,
                             const std::vector<float>& txt_mod1_gate, const std::vector<float>& txt_mod2_shift,
                             const std::vector<float>& txt_mod2_scale, const std::vector<float>& txt_mod2_gate,
                             std::vector<float>* img_out, std::vector<float>* txt_out);

  // Narrow zero-copy compatibility probe: feed qkv_proj's output DmaBufs straight into
  // flash_step_init and compare that result with the existing host-copy path.
  bool RunZeroCopyQkvToFlashProbe(const std::vector<float>& img, const std::vector<float>& pe,
                                  const std::vector<float>& img_mod1_shift,
                                  const std::vector<float>& img_mod1_scale,
                                  long long* direct_run_ms, float* max_abs_diff);

  // Diagnostic-only: run block 0's text projection and first image streaming projection through
  // the same RunPiece path used by forward(), returning Q, K, and V for each.
  bool DebugFirstQkv(const std::vector<float>& img, const std::vector<float>& txt,
                     const std::vector<float>& pe, const std::vector<float>& pe_ctx,
                     const std::vector<float>& img_mod1_shift, const std::vector<float>& img_mod1_scale,
                     const std::vector<float>& txt_mod1_shift, const std::vector<float>& txt_mod1_scale,
                     std::vector<std::vector<float>>* img_qkv,
                     std::vector<std::vector<float>>* txt_qkv);

 private:
  struct CachedPiece;
  bool RunPiece(const std::string& file_name,
                const std::vector<const std::vector<float>*>& inputs,
                const std::vector<std::vector<int32_t>>& input_shapes,
                const std::vector<std::vector<int32_t>>& output_shapes,
                std::vector<std::vector<float>>* out_results,
                const std::string& cache_key = "");
  bool RunCachedDirect(const std::string& cache_key,
                       const std::vector<LiteRtTensorBuffer>& input_buffers,
                       const std::vector<LiteRtTensorBuffer>* output_buffers = nullptr);
  CachedPiece* PrepareCachedPiece(const std::string& file_name,
                                  const std::vector<std::vector<int32_t>>& input_shapes,
                                  const std::vector<std::vector<int32_t>>& output_shapes,
                                  const std::string& cache_key);

  std::string model_dir_;
  std::string img_qkv_file_;
  std::string txt_qkv_file_;
  std::string img_out_file_;
  std::string txt_out_file_;
  LiteRtEnvironment env_ = nullptr;
  // Eight chunked pieces total ~500 MiB and were already proven safe to retain in the Kotlin
  // chain. Keeping them open removes repeated dispatch-runtime compilation from 35 calls.
  std::map<std::string, std::unique_ptr<CachedPiece>> cached_pieces_;
};

}  // namespace pockettavern

#endif  // POCKETTAVERN_KLEIN_DOUBLE_BLOCK_ENGINE_HPP
