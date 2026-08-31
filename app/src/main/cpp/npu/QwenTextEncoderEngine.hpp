#ifndef POCKETTAVERN_QWEN_TEXT_ENCODER_ENGINE_HPP
#define POCKETTAVERN_QWEN_TEXT_ENCODER_ENGINE_HPP

#include <memory>
#include <string>
#include <vector>

namespace MNN {
namespace Transformer {
class Llm;
}
}  // namespace MNN

namespace pockettavern {

// QwenTextEncoderEngine: runs the on-device MNN export of FLUX.2 [klein]'s Qwen3-4B text encoder
// (see docs/FLUX2_KLEIN_PHASE1_TEXT_ENCODER_PLAN.md) to produce the [1,512,7680] conditioning
// tensor Klein's txt_in expects for an arbitrary prompt -- the on-device replacement for the
// checked-in context.bin the fixed-prompt reference pipeline uses today.
//
// Uses MNN's vendored LLM inference engine (transformers/llm/engine) on CPU. The model must be
// exported with `--hidden_states_layers 9,18,27` (see llmexport.py) so its second module output,
// named "hidden_states", is the concatenation of those three post-block hidden states -- exactly
// matching scripts/export_klein_qwen_reference.py's PyTorch reference, validated desktop-side to
// ~1.1-1.3% relative error on the real-prompt-content region across several prompts (see
// docs/flux2-klein-conversion.md, 2026-08-30 entry).
class QwenTextEncoderEngine {
 public:
  QwenTextEncoderEngine();
  ~QwenTextEncoderEngine();

  QwenTextEncoderEngine(const QwenTextEncoderEngine&) = delete;
  QwenTextEncoderEngine& operator=(const QwenTextEncoderEngine&) = delete;

  // config_path points at the exported model's config.json (llm.mnn / llm.mnn.weight /
  // tokenizer.mtok alongside it), e.g. from `--export mnn --hqq` in llmexport.py.
  //
  // mmap_cache_dir enables MNN's use_mmap weight loading (see llm_->set_config() call in the
  // .cpp): the fp16 weight file (~8GB) is materialized once into a file-backed mmap allocation
  // under this directory instead of anonymous heap memory. That matters specifically because of
  // how Android 17's Memory Limiter kills a process -- it enforces memory.high on the anon+swap
  // cgroup counter, and anonymous pages that don't fit have nowhere to go but swap; once swap is
  // full the process is killed (confirmed via logcat: "MemoryLimiter: onLimitExceeded ...
  // type=memory.high" followed by death ~66s later). File-backed mmap pages are clean/reclaimable
  // instead -- the kernel can drop and re-read them from disk under pressure without touching
  // swap, which is what keeps the full 8GB from ever needing to be resident at once. Must be a
  // writable, app-private directory (e.g. Context.cacheDir); pass empty to fall back to the
  // model's plain (fully-resident) load.
  //
  // This directory's contents are always wiped and rematerialized fresh on every Load() call --
  // NOT reused across calls despite MNN's own use_cached_mmap option nominally supporting that.
  // Root-caused on-device: MNN's cached-mmap shard files are keyed by a plain ordinal allocation
  // counter, not tensor identity, and several CPU executor constructors only participate in that
  // counter on the first ("cold") materializing pass; a later load that trusts an existing cache
  // takes a different code path that skips those allocations, so the counter desyncs and shard
  // file N -- still byte-identical on disk -- ends up mmap'd into the wrong tensor. This silently
  // produced numerically-plausible-but-wrong encoder output with no error anywhere in the
  // pipeline. See the .cpp for the full trace. Only a fresh materialize-and-read within the same
  // Load() call has been confirmed correct.
  bool Load(const std::string& config_path, const std::string& mmap_cache_dir);

  bool IsLoaded() const { return llm_ != nullptr; }

  // Releases the ~8GB weight mapping. Callers that only need one Encode() call per Load() (e.g.
  // KleinDiffusionEngine::Generate(), which must free this before its own NPU/VAE phases start
  // allocating -- see that call site's comment) should call this immediately after Encode()
  // returns rather than relying on the kernel to reclaim the mmap'd pages under pressure on its
  // own: confirmed on-device that reclaim can lose that race against fresh NPU allocations,
  // breaching Android 17 Memory Limiter's memory.high ceiling before eviction catches up.
  void Unload();

  // Encodes prompt into out_hidden_states, a 512*7680 float32 buffer (row-major [seq, hidden]),
  // matching context.bin's layout. Tokens beyond 512 are truncated; the tail is right-padded with
  // the tokenizer's pad token, and the padding region is masked out of every real token's
  // attention (MNN's own naive built-in mask is causal-only and not padding-aware, so this class
  // builds the combined causal+padding mask itself -- see the desktop validation script for the
  // reference implementation this mirrors).
  bool Encode(const std::string& prompt, std::vector<float>* out_hidden_states);

  static constexpr int kSeqLen = 512;
  static constexpr int kHiddenPerLayer = 2560;
  static constexpr int kNumTapLayers = 3;
  static constexpr int kHidden = kHiddenPerLayer * kNumTapLayers;  // 7680

 private:
  std::unique_ptr<MNN::Transformer::Llm> llm_;
};

}  // namespace pockettavern

#endif  // POCKETTAVERN_QWEN_TEXT_ENCODER_ENGINE_HPP
