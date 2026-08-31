# FLUX.2 [klein] production integration — Phase 1: on-device Qwen3-4B text encoder

## Context

The Klein NPU transformer pipeline is now proven correct end-to-end on real hardware (fox image,
92s/step), but it only works with a single hardcoded prompt because there is no on-device text
encoder — the working run reads a checked-in `context.bin` produced offline by PyTorch. The user
wants Klein to become a real, selectable on-device image-gen option, which requires arbitrary
prompts. That requires five things in sequence: (1) a real on-device Qwen3-4B text encoder,
(2) a generalized denoising loop (not the fixed 4-step reference), (3) a production engine/JNI
layer chaining encoder → transformer → VAE, (4) a download/staging flow for the ~12GB NPU artifact
set plus the encoder weights, and (5) a model-selection UI entry. Items 2-5 cannot be usefully
built or tested without item 1, so **this plan scopes and implements item 1 only**; items 2-5 are
recorded as the follow-up roadmap (to be added to `docs/FLUX2_KLEIN_HANDOFF.md`) rather than
implemented now.

Research this session found that item 1 is much smaller than initially feared: PocketTavern
already vendors MNN's LLM inference engine (`app/src/main/cpp/MNN/transformers/llm/engine/`),
already compiles it into `libMNN.so` (`MNN_BUILD_LLM ON` in `app/src/main/cpp/CMakeLists.txt`),
and it is simply unwired (no JNI entry point uses it yet — the app's actual chat feature uses a
separate closed-source llama.cpp AAR, `GgufEngine`, which has no hidden-state access and is not
relevant here). MNN's Python export tooling (`app/src/main/cpp/MNN/transformers/llm/export/llmexport.py`)
already has native Qwen3 support (`regist_qwen3`) and an existing `hidden_states` output-tap
mechanism (used today for speculative-decoding draft models) — so exposing layers 9/18/27
concatenated as a named export output is a targeted modification to existing tooling, not new
infrastructure. This makes Phase 1 "convert + wire an existing on-device LLM runtime," comparable
in size to the SDXL NPU text-encoder work already in this codebase, not a from-scratch effort like
the Klein transformer NPU conversion was.

## Scope of this phase

Produce, on real Pixel hardware, a `context.bin`-equivalent tensor (`[1,512,7680]`, layers 9/18/27
hidden states concatenated) for an **arbitrary user-supplied prompt**, and confirm it is
numerically close to the existing PyTorch reference path
(`scripts/export_klein_qwen_reference.py`) for the same prompt. This is a diagnostic-hook
deliverable (mirrors how the Klein transformer work itself proceeded — validate correctness via a
native diagnostic entry point before touching production UI), not yet wired into the app's
generation UI.

## Implementation steps

1. **Export Qwen3-4B to MNN with the right hidden-state taps.**
   - Use `~/Downloads/flux2_klein_qwen/` (official Qwen3-4B shards + tokenizer, already verified
     present per `docs/flux2-klein-conversion.md`).
   - Modify `app/src/main/cpp/MNN/transformers/llm/export/llmexport.py`'s existing
     `hidden_states` output-tap path (currently taps the single final pre-lm_head hidden state) to
     instead tap decoder layers 9, 18, and 27, concatenate them on the feature dim, and export that
     as a named MNN output (`hidden_states_9_18_27` or similar), matching
     `scripts/export_klein_qwen_reference.py`'s existing concatenation order exactly.
   - Export with the chat-template/padding-to-512 behavior matching the existing reference script
     (reuse its prompt-formatting logic rather than re-deriving it).
   - Quantize for on-device size/speed (MNN's standard export quant flags, e.g. int4 or int8 —
     check what `llmexport.py` defaults to and what other MNN LLM exports in this ecosystem use)
     — 4B params in fp16 is ~8GB, too large to ship/run as-is.
   - Validate the exported MNN model on desktop (MNN's Python/CLI tooling if available, or a small
     desktop harness) against `export_klein_qwen_reference.py`'s PyTorch output for a couple of
     test prompts before touching Android at all — this is the same "prove correctness off-device
     first" discipline used throughout the Klein transformer work.

2. **Native JNI wrapper.**
   - New `app/src/main/cpp/npu/QwenTextEncoderEngine.{hpp,cpp}` (naming mirrors
     `NpuTextEncoderEngine`), wrapping `MNN::Transformer::Llm::createLLM` + `forwardRaw` +
     `getOutputIndex`/`getOutputs` to run the tokenize → forward → extract-named-output flow.
   - Add a diagnostic JNI entry point (pattern-matched to
     `Java_com_pockettavern_app_util_NpuDiagnostic_nativeRunKleinOneStepReference` in
     `jni_diffusion.cpp`) that takes a prompt string, runs the encoder, and dumps the resulting
     `[1,512,7680]` tensor to a `.bin` file the same way existing debug dumps work, so it can be
     pulled and compared against the PyTorch reference with a numpy diff script (matching this
     project's established validation pattern).
   - Add the new source files to `app/src/main/cpp/CMakeLists.txt`'s `pockettavern_diffusion`
     target (or wherever `NpuTextEncoderEngine.cpp` is currently listed).

3. **Validate on real hardware.**
   - Stage the exported MNN model onto the device (same `adb push` + `run-as cat` staging
     convention used for the Klein NPU artifacts).
   - Run the new diagnostic hook for 2-3 varied prompts (not just the standing fox prompt), pull
     the resulting `.bin`, and diff against `export_klein_qwen_reference.py`'s PyTorch output for
     the same prompts. Target the same tolerance class already accepted elsewhere in this project
     (~0.3-0.5% relative error).
   - Note on-device timing (prefill-only forward pass over ≤512 tokens, not autoregressive
     generation — expect this to be fast relative to the 90s/transformer-step budget, but measure
     rather than assume, per this project's standing "profile, don't assume" lesson).

4. **Document.**
   - Save this plan itself into the repo as `docs/FLUX2_KLEIN_PHASE1_TEXT_ENCODER_PLAN.md` (copied
     from this plan file) so it survives as a durable handoff artifact alongside the other
     `docs/flux2-klein-*` files, rather than only living in the ephemeral plan-mode file. Do this
     as the first documentation step, before implementation starts.
   - Append a dated section to `docs/flux2-klein-conversion.md` with the investigation/implementation
     trail (matching existing style in that doc).
   - Update `docs/FLUX2_KLEIN_HANDOFF.md`: mark the "Qwen text encoder still needs a real on-device
     execution path" line as resolved, and add a new "Remaining work for production UI integration"
     section listing items 2-5 above (generalized scheduler, production engine/JNI, download
     manifest, model-selection UI) as the explicit next-session roadmap, so the next session can
     pick up without re-deriving scope.

## What this phase deliberately does NOT include

- No changes to `ImageGenBackendType`, `SdxlModelManager`, `MnnDiffusionBackend`, or any UI file —
  per the existing handoff doc's standing instruction, Klein stays out of the production UI/download
  path until the full non-fixed-prompt pipeline is validated, and item 1 alone isn't that.
- No generalized denoising-schedule work (still uses the existing fixed 4-step reference path for
  the transformer/VAE side — only the text-conditioning input becomes dynamic).
- No download-manifest or model-selection UI work.

## Verification

- Desktop: exported MNN model's hidden-state output matches PyTorch reference within ~0.3-0.5%
  relative error for multiple prompts.
- Device: same comparison run through the real JNI/MNN on-device path, pulled and diffed the same
  way, for multiple prompts (not just the standing fixed prompt) — proving the encoder generalizes
  beyond the one prompt the rest of the pipeline has been validated against.
- Optional stretch (only if time permits after the above): feed one on-device-encoded prompt's
  output into the existing `nativeRunKleinOneStepReference`/`nativeDecodeKleinReferenceLatent` path
  in place of the fixed `context.bin`, to confirm a genuinely new prompt produces a coherent (not
  gray/degenerate) image end-to-end — this would be the first real proof the whole chain works
  beyond the one memorized prompt, though full scheduler/UI work is still deferred to later phases.
