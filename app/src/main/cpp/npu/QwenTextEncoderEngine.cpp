#include "npu/QwenTextEncoderEngine.hpp"

#include <cstdio>
#include <dirent.h>
#include <limits>

#include "llm/llm.hpp"

namespace pockettavern {

namespace {
using MNN::Express::_Input;
using MNN::Express::NCHW;
using MNN::Express::VARP;

// Non-recursive: MNN's mmap cache is a flat directory of *.static shard files plus its own sync
// marker, no subdirectories.
void WipeDirectoryContents(const std::string& dir) {
  DIR* d = opendir(dir.c_str());
  if (!d) return;
  struct dirent* entry;
  while ((entry = readdir(d)) != nullptr) {
    const std::string name = entry->d_name;
    if (name == "." || name == "..") continue;
    remove((dir + "/" + name).c_str());
  }
  closedir(d);
}

// The pad token id for Qwen3's tokenizer (<|endoftext|>). Fixed for this specific text encoder
// checkpoint, matching what scripts/export_klein_qwen_reference.py's HF tokenizer pads with.
constexpr int kPadTokenId = 151643;

// Qwen3's official chat template rendered for a single user turn with add_generation_prompt=True,
// enable_thinking=False, and no system message -- exactly what
// scripts/export_klein_qwen_reference.py feeds the reference encoder. This wrapper text is fixed
// regardless of prompt content, so it's built directly here rather than going through
// MNN::Transformer::Llm::apply_chat_template(), which has no way to force enable_thinking=False
// (Qwen3's template defaults to thinking-mode ON when that variable is left unset, which would
// silently diverge from the validated reference).
std::string BuildChatPrompt(const std::string& prompt) {
  return "<|im_start|>user\n" + prompt +
         "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

}  // namespace

QwenTextEncoderEngine::QwenTextEncoderEngine() = default;
QwenTextEncoderEngine::~QwenTextEncoderEngine() = default;

bool QwenTextEncoderEngine::Load(const std::string& config_path, const std::string& mmap_cache_dir) {
  llm_.reset(MNN::Transformer::Llm::createLLM(config_path));
  if (!llm_) {
    return false;
  }
  if (!mmap_cache_dir.empty()) {
    // See the doc comment on Load() in the header for why this is the fix for the Memory
    // Limiter OOM-kill, not just a load-time optimization.
    //
    // Always wipe and force a fresh materialize-then-read pass, never trust an existing cache
    // from a prior Load() call. Root-caused on-device: MNN's cached-mmap machinery
    // (MmapAllocator in source/core/BufferAllocator.cpp) names each weight's shard file by a
    // plain ordinal allocation counter, not by tensor identity. Several CPU executor
    // constructors allocate+release a scratch STATIC buffer mid-construction on the first
    // ("cold", materializing) pass, which participates in that counter; on a later load that
    // finds the cache already complete, those same constructors skip that scratch allocation
    // entirely (see useCachedMmap-gated early returns in e.g. Convolution1x1Strassen.cpp,
    // KleidiAIConvolution.cpp), so the counter advances on a different sequence and shard file
    // N -- still byte-identical on disk -- ends up mmap'd into a *different* tensor than the one
    // that wrote it. No content/identity check exists anywhere in that path, so this reliably
    // produced numerically-plausible-but-wrong encoder output (a structurally coherent-looking
    // but content-garbled "mosaic" image, confirmed across multiple prompts/seeds) with no error
    // anywhere in the pipeline. Only a fresh materialize-and-read within the same Load() call has
    // ever produced correct output on this device -- so that's the only path this ever takes.
    WipeDirectoryContents(mmap_cache_dir);
    llm_->set_config("{\"use_mmap\": true, \"use_cached_mmap\": true, \"tmp_path\": \"" +
                      mmap_cache_dir + "\"}");
  }
  if (!llm_->load()) {
    llm_.reset();
    return false;
  }
  return true;
}

void QwenTextEncoderEngine::Unload() { llm_.reset(); }

bool QwenTextEncoderEngine::Encode(const std::string& prompt, std::vector<float>* out_hidden_states) {
  if (!llm_ || !out_hidden_states) {
    return false;
  }

  std::vector<int> input_ids = llm_->tokenizer_encode(BuildChatPrompt(prompt));
  if (input_ids.empty()) {
    return false;
  }
  if (static_cast<int>(input_ids.size()) > kSeqLen) {
    input_ids.resize(kSeqLen);
  }
  const int real_len = static_cast<int>(input_ids.size());
  input_ids.resize(kSeqLen, kPadTokenId);

  MNN::Express::ExecutorScope executor_scope(llm_->getExecutor());

  // forwardRaw() is a low-level entry point: unlike Llm::forwardVec() (the path response()/
  // generate() use), it does not update the shared KVMeta itself -- the caller must, or the
  // KV-cache manager's size bookkeeping never learns this call is adding kSeqLen tokens.
  llm_->setKVCacheInfo(kSeqLen, 0);

  VARP hidden_state = llm_->embedding(input_ids);
  VARP position_ids = llm_->gen_position_ids(kSeqLen);

  // Combined causal + padding mask: token i may attend to token j only if j <= i (causal) and
  // j < real_len (not padding). MNN's own gen_attention_mask() is causal-only (and, for the CPU
  // backend, defaults to a bare scalar meaning "no mask tensor at all") and would let the real
  // prompt tokens attend into the padding region, which the exported model was never validated
  // against -- see docs/flux2-klein-conversion.md's 2026-08-30 entry.
  VARP attention_mask = _Input({1, 1, kSeqLen, kSeqLen}, NCHW, ::halide_type_of<float>());
  {
    auto* mask_ptr = attention_mask->writeMap<float>();
    const float kMasked = std::numeric_limits<float>::lowest();
    for (int i = 0; i < kSeqLen; ++i) {
      for (int j = 0; j < kSeqLen; ++j) {
        mask_ptr[i * kSeqLen + j] = (j <= i && j < real_len) ? 0.0f : kMasked;
      }
    }
  }

  std::vector<VARP> outputs = llm_->forwardRaw(hidden_state, attention_mask, position_ids);
  const int hidden_states_index = llm_->getOutputIndex("hidden_states");
  if (hidden_states_index < 0 || hidden_states_index >= static_cast<int>(outputs.size())) {
    return false;
  }
  VARP hidden_states_out = outputs[hidden_states_index];
  auto info = hidden_states_out->getInfo();
  if (!info || info->size != static_cast<int>(kSeqLen) * kHidden) {
    return false;
  }
  const float* data = hidden_states_out->readMap<float>();
  if (!data) {
    return false;
  }
  out_hidden_states->assign(data, data + static_cast<size_t>(kSeqLen) * kHidden);
  return true;
}

}  // namespace pockettavern
