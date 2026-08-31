# FLUX.2 [klein] NPU implementation handoff

Updated 2026-08-30. This doc is the entry point — the full
history/design rationale lives in `docs/flux2-klein-conversion.md` (read that for context on *why*
things are built this way; this doc is just "what's true right now and what to do next").

## IN PROGRESS — Phase 1: on-device Qwen3-4B text encoder (2026-08-30)

Full plan: `docs/FLUX2_KLEIN_PHASE1_TEXT_ENCODER_PLAN.md`. Goal: replace the fixed-prompt checked-in
`context.bin` (mentioned throughout the rest of this doc) with a real on-device text encoder so
Klein can take arbitrary prompts — the last piece before Klein can be considered for production UI
integration.

**Done and validated:**
- `llmexport.py` extended with `--hidden_states_layers` (reuses the existing dflash hidden-state-tap
  mechanism) so MNN's Qwen3-4B export can produce the same layers-9/18/27-concatenated
  `[1,512,7680]` tensor as `scripts/export_klein_qwen_reference.py`'s PyTorch reference.
- Desktop-validated MNN's Qwen3 forward against the PyTorch reference for 3 varied prompts:
  real-content-region relative error ~1.1-1.3% (same tolerance class used elsewhere in this
  project); padding-region divergence explained as shared large-magnitude "attention sink" noise
  in both implementations, not a one-sided bug.
- Quantized MNN export (`--export mnn --hqq`, 2.26GB int4 weights) built and staged onto the Pixel.
- `app/src/main/cpp/npu/QwenTextEncoderEngine.{hpp,cpp}` (wraps MNN's own `MNN::Transformer::Llm`
  engine, already compiled into `libMNN.so` but previously unwired) + diagnostic JNI entry point
  (`nativeRunQwenTextEncoder` / `NpuDiagnostic.runNativeQwenTextEncoder` /
  `run_native_qwen_text_encoder` debug intent extra) all build cleanly.

**Crash fixed.** The `OUT_OF_MEMORY` was never a real memory-size problem — root cause was
`QwenTextEncoderEngine::Encode` calling `Llm::forwardRaw()` (a low-level entry point) without first
calling `llm_->setKVCacheInfo(kSeqLen, 0)`, which `forwardVec()` (what `response()`/`generate()` use)
does internally. Without it, `mMeta->add` stays 0, `CPUAttention` inserts zero tokens into its KV
cache, and a resulting zero-byte buffer allocation silently propagates as `OUT_OF_MEMORY`. One-line
fix, confirmed on real hardware: a full 36-layer, 512-token forward pass now completes.

**Quantization accuracy investigated and resolved with a decision: ship fp16.** int4 (`--hqq` and
plain) and int8 exports (including `--sym`, `--quant_block 32`, `--sym --hqq` variants) are all
numerically unusable on-device (~27-55% relative error vs. the PyTorch reference, measured on the
real `hidden_states` output, not a logit proxy). Investigated whether this was a code/export bug
(per `MNN/skills/general-debug/SKILL.md` §2 methodology) before accepting it as a quantization
limit: ruled out embedding-table export corruption (dequantized rows directly, cosine ~0.99998 vs.
HF weights across the whole vocab); confirmed the huge per-element outlier is a real "massive
activation" phenomenon present in HF's own reference too, not invented noise; but even excluding
all outlier channels, ~27% error remains from broad noise across ordinary channels — a genuine
checkpoint-specific quantization-accuracy problem (this Qwen3-4B checkpoint has extreme per-channel
weight outliers, e.g. a `k_norm` gamma of 44.0 in layer 0), not a bug. Full investigation trail in
`docs/flux2-klein-conversion.md`'s "2026-08-30 continued further still" entry. **Decision (made by
the user 2026-08-30): ship fp16** (`--quant_bit 16`, 8GB weight file, ~348s on-device forward pass,
~3.5% relative error, already validated on real hardware) despite the size, rather than invest
further in mixed-precision int8/int4 tuning. That quantization work is deferred indefinitely.
**Do not integrate this into production UI yet** — same standing rule as below. NPU deployment of
the fp16 text encoder is the explicit next phase once the fp16 CPU path is fully wired up end-to-end
(per the user's standing instruction: get it working on CPU first, NPU later).

## RESOLVED — the "Q/K NPU numerical failure" below was a host-side RoPE bug, not a Tensor G5 defect (2026-08-30)

**The entire "Critical current status" section below is now superseded and its root-cause guess
was wrong.** The real bug: `jni_diffusion.cpp`'s RoPE-position generator (the `positions()` lambda
inside `nativeRunKleinOneStepReference`, used for both `pe` and `pe_ctx`) declared its per-token
write offset as `size_t o=0;` *inside* the per-token loop instead of `o=n*256`. Every token's 256
floats of RoPE data overwrote `pe[0..255]`; only the last token in the sequence ever had valid
RoPE data, and every other token got an all-zero rotation matrix (which zeros out `apply_rope`'s
output for that token). This corrupted RoPE for the entire pipeline — double blocks *and* single
blocks, since both consume the same `pe`/`pe_ctx` buffers — and is what actually produced
near-zero Q/K and near-uniform-gray images, not a Tensor G5 compiler/hardware defect.

Found via a from-scratch CPU-only investigation (see `docs/flux2-klein-conversion.md`'s "RoPE
generator bug" update for the full trail): a native RMSNorm+RoPE implementation
(`app/src/main/cpp/npu/klein_qk_norm_rope.hpp`) was built and validated bit-close against PyTorch
off-device, then wired in to replace the fused NPU artifact's norm+rope tail — and *still* produced
garbage on real hardware despite passing every offline check. Chasing that contradiction (comparing
device-measured raw Q against PyTorch's raw projection matched; the isolated norm+rope function fed
the same real raw Q also matched; only the full on-device chain didn't) led to the actual `pe`
buffer, which turned out to be almost entirely zero. Fixed with a one-line change (`o=n*256`
instead of `o=0`, `app/src/main/cpp/jni_diffusion.cpp`'s `positions()` lambda).

**After the fix, the ORIGINAL fused qkv+norm+rope NPU artifact was re-tested directly (not the
native-RMSNorm workaround) and is confirmed correct**: `meanAbsDiff` vs. PyTorch is ~0.3-0.4%
relative error for double_blocks.0's Q/K, the same tolerance class as every other chained-dispatch
measurement already accepted elsewhere in this project. `KleinDoubleBlockEngine` was reverted back
to the fused artifacts (the native-RMSNorm split is unnecessary and was reverted); the zero-copy
performance paths (`forwardZeroCopy`, `forwardZeroCopyPooled`, `RunZeroCopyQkvToFlashProbe`), which
had been temporarily disabled while the split was in place, are restored to their original form.
`klein_qk_norm_rope.hpp` and its host-buildable self-test (`scripts/klein_qk_norm_rope_selftest.cpp`)
are left in the tree, unused, in case a future shape/block ever hits a genuine NPU defect.

**Full pipeline confirmed working end-to-end on real Tensor G5 hardware after the fix**: all 25
transformer blocks + final layer completed in 92.1s (step 0), all 4 diffusion steps completed
cleanly (~90s each), and the direct staged VAE decode (19.3s) produced a real, coherent 1024×1024
image (a fox in a moonlit forest, matching the project's standing reference prompt) — not gray, not
degenerate. This is the first confirmed **complete on-device FLUX.2 [klein] text-to-image
generation** in this project, using the fixed-Qwen-context reference path
(`nativeRunKleinOneStepReference` / `nativeDecodeKleinReferenceLatent`).

**What's still true from before**: the ~92s/step full-pipeline runtime (~6.1 min for 4 steps, plus
~20s VAE decode) has not changed — this fix was a correctness fix, not a performance one. The
"Immediate next steps" list further down (re-measure double-stream C++ perf, check input-buffer
DmaBuf, persistent-model reuse, etc.) is still the right place to look for speed. The Qwen text
encoder still needs a real on-device execution path (this run used the checked-in fixed-prompt
`context.bin`) before this becomes a real prompt-to-image feature. **Still do not integrate into
PocketTavern's download UI or `NpuUnetEngine`** until the full non-fixed-prompt pipeline (real Qwen
+ scheduler beyond the fixed 4-step reference) is validated end-to-end.

## Critical current status — Q/K NPU numerical failure (2026-08-29) — SUPERSEDED, see above

This supersedes the older status narrative below. The full fixed-prompt pipeline and direct VAE decode
run on the Pixel, but images are nearly uniform gray because the first double transformer block is
already numerically wrong. Do not tune scheduler/VAE/image quality until this is repaired.

### Proven facts

- Corrected four-step reference: denoiser steps are ~89–94 s; direct staged VAE decode is ~20 s.
- Input/component boundaries are sound: `img_in`, `txt_in`, `time_in`, and modulation outputs match
  PyTorch closely. Native RoPE positions match upstream `EmbedND` to max `3.02e-6`.
- Host-copy and zero-copy double-block paths have the same large error: DmaBuf is not the cause.
- Real step-0 QKV traces prove the fused artifact fault. V matches (mean error ~`0.011–0.013`,
  magnitude `3.4–4.7`); Q/K are effectively zero (native mean abs ~`0.0007`, PyTorch ~`0.75`).
  The NPU reports success: the failing portion is fused Q/K RMSNorm and/or RoPE.

### Repair chosen and artifacts

Keep input normalization + learned QKV matrix on NPU. Move only per-head Q/K RMSNorm and RoPE to
native C++ before existing flash-attention artifacts. This preserves the expensive projection while
avoiding the faulty compiled tail.

`/home/brandont/code/litert-torch/scratch/klein_double0_qkv_export.py` now has `--raw-qkv`: three
inputs (`x [1,T,3072]`, `mod_shift [1,1,3072]`, `mod_scale [1,1,3072]`), separate Q/K/V outputs
`[1,24,T,128]`, and no PE input. It also fixes a text-exporter bug: `--stream txt` now uses text,
not image, learned weights.

Block-0 raw artifacts are locally validated against PyTorch (max error `5.15e-5`) and compiled:

- `/home/brandont/code/litert-torch/scratch/models/flux2_klein_probe/double0_img_raw_qkv_1024_noflags_aot/double0_img_raw_qkv_1024_Google_Tensor_G5.tflite`
- `/home/brandont/code/litert-torch/scratch/models/flux2_klein_probe/double0_txt_raw_qkv_512_noflags_aot/double0_txt_raw_qkv_512_Google_Tensor_G5.tflite`

They are not yet staged/pushed or wired into C++.

### Exact next work

1. Extract Q/K per-head RMSNorm scales (`[128]`) for every double/single block from safetensors;
   do not omit learned scales.
2. Add C++ in-place Q/K RMSNorm + RoPE helper for `[heads,tokens,128]`, matching upstream epsilon
   and the validated `[1,1,tokens,64,2,2]` PE layout.
3. Wire raw artifacts into double block 0, test `debug_double0_img/txt.bin` against PyTorch, then
   extend all 5 double blocks.
4. Apply the split to `KleinSingleBlockEngine` too; its QKV artifact also fuses this risky tail.
5. Restage/deploy and rerun four-step reference only after per-block comparison passes.

Device trace files can be pulled with `adb exec-out run-as com.pockettavern.app cat
files/flux2_klein_npu/<name> > /tmp/<name>`: `debug_img_in.bin`, `debug_txt_in.bin`,
`debug_mod_img_0.bin`, `_1`, `debug_mod_txt_0.bin`, `_1`, `debug_double0_{img,txt}_{q,k,v}.bin`,
and `debug_double0_{img,txt}.bin`.

## Where things stand

FLUX.2 [klein] 4B's transformer blocks run on the Pixel's Tensor G5 NPU via a chunked/flash-attention
redesign (SDPA can't run as one NPU dispatch at full 4608-token resolution — see the main doc's
"performance cliff" and "chunked attention" sections for why). Both `single_blocks.0` and
`double_blocks.0` are validated correct on real hardware. The remaining work is confirming/optimizing
real C++ engine performance before scaling to the other 22 blocks.

**User's explicit goal, still active**: "We need to confirm performance, since 26.5 minutes is too
long." A Kotlin diagnostic harness measured ~18.2s/block (single-stream) / ~6.7s/block (double-stream),
naively extrapolating to ~26.5 min/image — too slow. The production engine will be C++ (mirroring
SDXL's already-shipped `NpuUnetEngine`), not Kotlin, so a real C++ number was needed.

## What was just found and fixed

Built `app/src/main/cpp/npu/KleinSingleBlockEngine.{hpp,cpp}` — a throwaway (not production-wired)
native engine for one single-stream block, using the LiteRT C API directly (same pattern as
`NpuUnetEngine.cpp`, which is the reference to copy from for API conventions). Wired through:
- `app/src/main/cpp/jni_diffusion.cpp` — new JNI function `nativeRunKleinSingleBlockSmoke`
- `app/src/main/cpp/CMakeLists.txt` — added `npu/KleinSingleBlockEngine.cpp` to the
  `pockettavern_diffusion` target's sources
- `app/src/main/kotlin/com/pockettavern/app/util/NpuDiagnostic.kt` — `runNativeKleinSingleBlockSmoke`
- `app/src/main/kotlin/com/pockettavern/app/MainActivity.kt` — debug-only intent hook
  `run_native_klein_single_block_smoke`

**First real run: 34.9s/block — worse than Kotlin's 18.2s**, despite correct output
(`maxAbsDiff=1.930710`, matching Kotlin exactly) and despite using native (non-Dalvik) heap. Added
per-phase timing (`model`/`compile`/`buffers`/`run`/`read` — model-open, `LiteRtCreateCompiledModel`,
buffer alloc, `LiteRtRunCompiledModel`, and `LiteRtLockTensorBuffer(read)+memcpy+Unlock` respectively).
Found: **`read` was 66% of total (21.2s of 32s)**, while `run` (actual NPU compute, 6.1s total) matched
Kotlin-profiled expectations closely — the NPU itself was never the bottleneck.

Added diagnostic logging of each output tensor's `supported_types` (via
`LiteRtGetTensorBufferRequirementsSupportedTensorBufferType`) and `chosen_type` (via
`LiteRtGetTensorBufferType` after creation). Found: **every output tensor, across every piece type,
offers only `[Ahwb(2), DmaBuf(4)]`** (no `HostMemory` option exists for these NPU-dispatched tensors at
all), and `LiteRtCreateManagedTensorBufferFromRequirements` always auto-picks `Ahwb`.

**Root cause + fix**: switched output-buffer creation from
`LiteRtCreateManagedTensorBufferFromRequirements` to `LiteRtCreateManagedTensorBuffer` with
`kLiteRtTensorBufferTypeDmaBuf` forced explicitly (buffer size taken from
`LiteRtGetTensorBufferRequirementsBufferSize`). This is the current state of
`KleinSingleBlockEngine.cpp`'s output-buffer-creation loop — **input buffers were NOT changed and
still use the auto-picking `FromRequirements` call** (untested whether that matters, see below).

**Result: read phase 21.2s → 2.9s (~7x faster), total 34.9s → 16.3s**, correctness unchanged
(`maxAbsDiff=1.930710`, bit-for-bit same as before the fix). Ahwb's CPU-lock path evidently triggers a
real cache-sync cost on this hardware that DmaBuf's mapping avoids.

**Revised confirmed-input projection**: 16.3s/block (single-stream, real C++ measurement) × 19
remaining single-stream blocks + 6.7s/block (double-stream — **this number is carried over from the
Kotlin diagnostic / pre-DmaBuf-fix C++ state, NOT yet re-measured with the fix**) × 4 double-stream
blocks, × 4 Klein steps ≈ **22.5 min/image**. This supersedes the ~26.5min naive-Kotlin figure and an
earlier ~9.7min *projected* figure that didn't know about the Ahwb/DmaBuf penalty at all.

## Immediate next steps, in priority order

1. **Re-measure double-stream through the same C++ engine pattern with DmaBuf forced from the start.**
   No `KleinDoubleBlockEngine` exists yet in C++ — the 6.7s/block double-stream figure is from the
   Kotlin diagnostic (`runKleinDoubleChunkedBlockProbe` in `NpuDiagnostic.kt`), pre-dating this whole
   C++/DmaBuf investigation. Build a C++ equivalent (mirror `KleinSingleBlockEngine`'s structure —
   4 pieces per chunk instead of single-stream's 3, asymmetric img/txt chunk sizes per the main doc's
   "double-stream" sections) with `DmaBuf` forced from the start, and get a real number.

2. **Check whether input buffers need the same DmaBuf fix.** Only output buffers were changed in the
   fix above. `run` is now the dominant cost (5.9s/block) after the read-phase fix — worth checking
   whether input buffer type also affects `LiteRtRunCompiledModel`'s own cost (e.g. if the NPU has to
   do its own internal cache-sync reading from an Ahwb-backed input, that could show up inside `run_ms`
   rather than `read_ms`).

3. **Investigate `compile`+`buffers` phase overhead** (~4.1s/block combined, now proportionally bigger
   post-fix). The double-stream Kotlin diagnostic already proved persistent-model-reuse is a real,
   low-risk win (create once, reuse across dispatches, vs. create→run→destroy per call — see the main
   doc's "tested persistent-model reuse" update). `KleinSingleBlockEngine` currently does **not** test
   this pattern at all (each call to `RunPiece` does a fresh create→run→destroy). Worth trying in C++
   too, now that read is no longer masking it.

4. **Once both stream types have real DmaBuf-forced (and possibly reuse-optimized) C++ numbers,**
   update `docs/flux2-klein-conversion.md`'s projection section with the real figures (replace the
   "carried over" double-stream number).

5. **Then resume scaling to the remaining 22 blocks** (19 single-stream + 4 double-stream) using the
   C++ engine pattern, not the Kotlin diagnostic path — see the main doc's step 8 for the AOT-export
   pipeline (already generalized and proven on `single_blocks.1`; ~2-3 min/single-stream block, ~3
   min/double-stream block to AOT-compile, real measured).

## Practical gotchas hit while running this on-device (save yourself the debugging time)

- **Package name is `com.pockettavern.app`, not `com.pockettavern.app.debug`** — the debug build variant
  does NOT append a suffix. `adb shell pm list packages | grep pockettavern` to confirm if unsure.
- **`adb shell am start` on an already-running instance of the activity does NOT re-deliver the intent
  extras into a fresh `onCreate`** — the debug hooks in `MainActivity.kt` only fire in `onCreate`. Always
  `adb shell am force-stop com.pockettavern.app` before `am start` with a new diagnostic extra, or the
  hook silently doesn't run.
- Build with `./gradlew :app:assembleDebug -q`, install with
  `adb install -r app/build/outputs/apk/debug/app-debug.apk`.
- Launch example: `adb shell am start -n com.pockettavern.app/com.pockettavern.app.MainActivity --ez
  run_native_klein_single_block_smoke true`
- Watch results: `adb logcat -v time | grep -E "NATIVE_KLEIN_SINGLE_BLOCK_SMOKE|KleinSingleBlockEngine"`
  — final result line is tagged `NpuDiagnostic`/`NATIVE_KLEIN_SINGLE_BLOCK_SMOKE`, per-piece phase
  timing is tagged `PocketTavernDiffusion`/`KleinSingleBlockEngine: phases`.
- `LiteRtCreateCompiledModel` requires **both** `kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu` in
  `LiteRtSetOptionsHardwareAccelerators` — NPU-only flags cause `status=504` on at least
  `flash_step_probe_1152` (confirmed empirically this session; don't re-try NPU-only as an
  "optimization," it just fails).
- `LiteRtDestroyCompiledModel()` must be called explicitly **before** `TensorBufferHolder` destructors
  run (see `NpuUnetEngine::RunPiece`'s comment and `KleinSingleBlockEngine::RunPiece`'s mirrored
  ordering) — a documented Google Tensor dispatcher epoll/fd-reuse bug otherwise.
- Build is `-fno-exceptions` (matches the vendored MNN build) — use RAII holder structs + `return
  false` on error paths, never `throw`/`try`/`catch`.

## Files touched this session (all currently uncommitted — check `git status`/`git diff` before
committing anything)

- `docs/flux2-klein-conversion.md` — updated with the correction + this finding (read its last few
  `### Update:` sections for full narrative)
- `app/src/main/cpp/npu/KleinSingleBlockEngine.hpp` (new)
- `app/src/main/cpp/npu/KleinSingleBlockEngine.cpp` (new)
- `app/src/main/cpp/jni_diffusion.cpp` (added JNI entry point)
- `app/src/main/cpp/CMakeLists.txt` (added new source file)
- `app/src/main/kotlin/com/pockettavern/app/util/NpuDiagnostic.kt` (added smoke-test wrapper)
- `app/src/main/kotlin/com/pockettavern/app/MainActivity.kt` (added debug intent hook)
- `app/src/main/AndroidManifest.xml` (`android:largeHeap="true"` — from an earlier Kotlin-diagnostic
  memory fix, unrelated to the C++ work but still uncommitted from the same session)

Nothing has been committed. The user has not asked for a commit — check with them before committing,
per this repo's usual workflow.
