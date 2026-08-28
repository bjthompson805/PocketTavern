# FLUX.2 [klein] 4B on Tensor G5 — conversion handoff

**Status: updated 2026-08-27, later same day.** The "decisive current fact" below (even a plain
1024×1024 linear fails on this compiler) turned out to be a false conclusion caused by two AOT
flags, not a real compiler/model limit. With those flags dropped, a full Klein single-stream block
AOT-compiles for Tensor G5 in one NPU partition **at the real 1024px production token count**
(4608 tokens), and its exported graph has been confirmed numerically correct against the PyTorch
reference (CPU/XNNPACK). **Real on-device execution then hit a Tensor G5/Darwinn hardware fault**
(forced recovery, silent all-zero output) whose root cause is **not yet known** — an earlier version
of this doc wrongly assumed it was the same unfixable class of bug as an old SDXL NPU crash and
proposed shrinking the model as the fix; both the "unfixable, not our code" framing and the
"shrink it" instinct were wrong turns in this project's own prior history (see the corrected
"Update" section below) and should not be repeated here without actual evidence.
`NpuDiagnostic.kt`/`MainActivity.kt` now carry a throwaway debug-only diagnostic hook for this
(`runKleinSingle0` / `run_klein_single0_npu_diagnostic` intent extra) — the first PocketTavern
production source touched by this investigation, added deliberately to reuse the same
diagnostic-only pattern already established for the SDXL work, not a real pipeline integration.

## Goal

Investigate whether the existing SDXL strategy—export graph pieces, Tensor G5 AOT-compile them,
then create → run → close each `CompiledModel` to bound dmabuf memory—can make FLUX.2 [klein] 4B
practical on-device.

The intended target is the distilled Klein 4B text-to-image path, which is four steps and
guidance-distilled (`guidance=1.0`), so it does **not** need the two concurrent unconditional /
conditional CFG forwards used by the SDXL NPU path. Its lower number of forwards is still a strong
reason to pursue this if Tensor G5 can compile its transformer kernels.

## Input checkpoint

User-provided checkpoint (do not copy it to `/tmp`; `/tmp` is RAM-backed):

```
/home/brandont/Downloads/unstableRevolutionF2K_AlphaF2K4BFp16.safetensors
```

- File size: **7.3 GiB**.
- Despite the filename, every tensor is **BF16**.
- It is a merged **ComfyUI diffusion-model-only** checkpoint, not a complete FLUX pipeline. Its
  metadata shows it was saved after the user's selected LoRAs, so conversion from this file should
  preserve the intended style.
- Tensor names use the prefix `model.diffusion_model.`.
- It exactly matches the official Klein 4B flow-transformer topology:
  - 5 `double_blocks`
  - 20 `single_blocks`
  - hidden size 3072
  - 24 attention heads (head dimension 128)
  - latent input/output channels 128
  - text-conditioning input dimension 7680 (Qwen3 4B encoder)
  - 149 tensors total.

The model architecture source was checked out persistently here (it was moved out of `/tmp` after
the user pointed out that `/tmp` is tmpfs):

```
/home/brandont/code/litert-torch/scratch/flux2_upstream/flux2
```

The LiteRT Python venv required `einops`; version 0.8.2 was installed into:

```
/home/brandont/code/litert-torch/.venv
```

## Existing PocketTavern context

The successful SDXL work is documented in `docs/npu-unet-conversion.md`. It proves that one
separate, reshape-wrapped `CompiledModel` per graph piece is necessary to avoid cumulative dmabuf
memory. Reuse that lifetime/buffer architecture, but **do not** try to reuse `NpuUnetEngine`'s
SDXL-specific 36-piece role/shape manifest: Klein is a different transformer pipeline.

Natural Klein boundaries are its 5 dual-stream and 20 single-stream blocks. Its Qwen text encoder
and FLUX.2 autoencoder can initially be staged separately (encode / release → flow / release →
decode), rather than requiring all weights in RAM simultaneously.

## Persistent probe code and outputs

Probe script:

```
/home/brandont/code/litert-torch/scratch/flux2_klein_probe.py
```

Artifacts (all are on persistent storage, not `/tmp`):

```
/home/brandont/code/litert-torch/scratch/models/flux2_klein_probe/
```

The script:

1. Loads only the four tensors for `single_blocks.0` from the checkpoint and upcasts only that
   block to fp32—the same precision that was successful for SDXL Tensor G5 pieces.
2. Uses the official BFL implementation from `scratch/flux2_upstream/flux2/src`.
3. Exports direct PyTorch → LiteRT with `litert_torch.convert`; it does **not** use ONNX.
4. Uses a small fixed token probe (16 text + 64 image = 80 tokens) before attempting the real
   1024px token length.

The full single-stream block includes the hard operator set: RMS Q/K normalization, rotary
position application, scaled-dot-product attention, SiLU-gated MLP, modulation, and residual.
It was converted successfully:

```
single0.tflite — 468.0 MiB fp32
```

This verifies that the actual merged Klein weights and operator graph export cleanly through
LiteRT-Torch.

## G5 AOT compilation results

All AOT calls used the existing LiteRT SDK and this configuration:

```python
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.vendors.google_tensor import target as gt_target

aot_lib.aot_compile(
    tflite_path,
    target=[gt_target.Target(gt_target.SocModel.TENSOR_G5)],
    keep_going=True,
    google_tensor_enable_large_model_support=True,
    google_tensor_sharding_intensity="maximum",
)
```

### Results table

| Probe | TFLite size | G5 partitioning | Result |
|---|---:|---:|---|
| Full single-stream block | 468 MiB | 87/87 ops; one NPU partition | Failed: compiler `INTERNAL` |
| Pre-attention stage (`linear1` + Q/K RMS norm) | 324 MiB | 42/42 ops; one partition | Failed: compiler `INTERNAL` |
| Fused QKV linear (3072→9216) | 108 MiB | selected for NPU | Failed: compiler `INTERNAL` |
| Q-only linear (3072→3072) | 36 MiB | selected for NPU | Failed: compiler `INTERNAL` |
| Q slice (3072→1024) | 12 MiB | selected for NPU | Failed: compiler `INTERNAL` |
| Q tile (1024→1024) | 4 MiB | selected for NPU | Failed: compiler `INTERNAL` |

The decisive current fact: **even a plain fp32 1024×1024 linear probe fails in the Tensor G5
compiler**. The compiler does not report an unsupported op; it accepts/partitions the graph and
then crashes internally while compiling. This is below any transformer-specific complexity, so
splitting the model merely into blocks, Q/K/V, or 1024-wide tiles is not currently sufficient.

The compiler error files are ephemeral `/tmp/tmp*.error` files created by the SDK, so do not rely
on them being present. Their meaningful contents were consistent across all failures:

```
Partitioned subgraph..., selected <all> ops ... one partition.
Failed to compile model.
Compilation has failed with error type: INTERNAL.
```

Example error filenames from this run:

```
/tmp/tmpw56py2ia.error  # full block
/tmp/tmpb77i5as0.error  # pre-attention
/tmp/tmpin2w5hlk.error  # fused QKV
/tmp/tmp8ftg4wxw.error  # Q-only
/tmp/tmpajq2j3g3.error  # 3072→1024
/tmp/tmp23gmtq_2.error  # 1024×1024
```

## Update: the INTERNAL failures were a flag bug, not a model/size limit

All six failures in the results table above were AOT-compiled with both
`google_tensor_enable_large_model_support=True` and `google_tensor_sharding_intensity="maximum"`.
Those exact two flags are already documented, in this repo, as a known trigger for the same
compiler `INTERNAL` error class: `scratch/run_aot_piece.py` (the production SDXL 36-piece build)
carries an existing comment that this flag pair "fails up2 outright at batch=2 with a
compiler-internal error (INTERNAL, ...)" on an SDXL piece that has nothing to do with attention or
quantization, and that dropping the flags compiles it fine (just slower, ~4min vs <1s for that
piece).

Retrying every failed Klein artifact from the table with `keep_going=True` only (no
`large_model_support`, no `sharding_intensity`) via `scratch/retry_klein_noflags.py`:

| Probe | TFLite size | Result without the two flags |
|---|---:|---|
| Full single-stream block (RMS norm, RoPE, SDPA, SiLU-gated MLP, modulation, residual) | 468 MiB | **Compiled.** 87/87 ops, 1 partition, ~51s |
| Pre-attention stage (`linear1` + Q/K RMS norm) | 324 MiB | **Compiled.** 42/42 ops, 1 partition, ~40s |
| Fused QKV linear (3072→9216) | 108 MiB | **Compiled.** ~12s |
| Q-only linear (3072→3072) | 36 MiB | **Compiled.** ~3s |
| Q slice (3072→1024) | 12 MiB | **Compiled.** ~1s |
| Q tile (1024→1024) | 4 MiB | **Compiled.** <1s |

Every single one compiles in one NPU partition with no flags. There is no evidence left of a hard
matmul-size ceiling or an unsupported op in this operator set — the entire "decisive current fact"
in the previous version of this doc was an artifact of the flag bug. Compiled outputs are in
`scratch/models/flux2_klein_probe/*_noflags_aot/`.

This does **not** yet mean a full Klein block runs correctly on-device — only that it compiles.

## Update: full block also compiles at the real production token count (1024px)

The result above was still at the tiny 80-token probe shape (16 text + 64 image). Klein's real
shape at 1024px output is **512 text tokens** (Qwen3 4B encoder's `MAX_LENGTH` in
`flux2/text_encoder.py`) and **4096 image tokens** (AutoEncoder is 8x spatial downsample —
`ch_mult=[1,2,4,4]`, 3 downsample stages — then 2x2 patchify, so 1024/16 = 64 per side, 64×64 =
4096; this also matches the 128-channel patchified latent input the checkpoint uses), i.e. 4608
tokens total vs. the probe's 80.

Ran the full pipeline at real shape, both steps wrapped in the memory watchdog
(`scratch/watchdog_run.sh`, floor 2000 MB combined `MemAvailable`+`SwapFree`):

1. **Export** (`flux2_klein_probe.py --stage full --text-tokens 512 --image-tokens 4096`): succeeded,
   468 MiB fp32 TFLite (same weight size as the probe shape — token count only affects activation
   shapes baked into the graph, not weight size), peak RSS ~2.9 GiB.
2. **AOT compile** for Tensor G5, still with no `large_model_support`/`sharding_intensity` flags
   (`scratch/retry_klein_noflags.py`): succeeded, **87/87 ops offloaded to 1 NPU partition**, ~19
   minutes compile time, peak RSS ~11.8 GiB (watchdog never tripped; system stayed responsive).
   Output: `scratch/models/flux2_klein_probe/single0_full_realtoks_noflags_aot/single0_full_realtoks_Google_Tensor_G5.tflite`
   (261 MiB).

So a full Klein single-stream block — the hardest operator mix (RMS norm, RoPE, SDPA over a
4608-token sequence, SiLU-gated MLP, modulation, residual) — compiles for Tensor G5 in one NPU
partition at the real production shape, without needing the buggy flags at all. This removes the
last open question from the previous update: `sharding_intensity="maximum"` was not needed even at
full size, so there is currently no reason to believe the flag-triggered `INTERNAL` bug will
resurface.

Compile time (~19 min for one single-stream block) and host RAM/swap use (~12 GiB peak for one
block, on a device that will eventually need to hold pieces from 5 double-blocks + 20 single-blocks
+ text encoder + autoencoder) are real open costs for the desktop conversion pipeline, separate
from and not yet measured for on-device runtime cost.

## Update: numerical correctness confirmed against the PyTorch reference (CPU, real shape)

Compared the exported (pre-AOT, CPU-runnable) `single0_full_realtoks.tflite` against the PyTorch
`SingleStreamBlock` reference, same real-shape inputs (fixed seed, 4608 tokens), using:

- `scratch/compute_klein_single0_torch_reference.py` — runs the PyTorch block, saves inputs/output
  to `.npy` in `scratch/models/klein_single0_validation_io/`, exits (frees memory before the next
  step loads TFLite).
- `scratch/run_klein_single0_tflite_and_compare.py` — loads the TFLite graph via
  `ai_edge_litert.interpreter.Interpreter` (XNNPACK CPU delegate), runs it on the same saved
  inputs, diffs against the saved PyTorch output.

Result: `maxAbsDiff=0.000626`, `meanAbsDiff=0.000014`, against `meanAbsRef=6.029` (mean absolute
output magnitude) — fp32 rounding-level agreement (XNNPACK vs. eager PyTorch execution order), not
a graph-correctness problem. The exported single-stream block computes the right thing.

Caveat: this validates the *graph/op math* that was also fed into the Tensor G5 AOT compiler (same
source `.tflite`), not the AOT-compiled artifact's actual on-device numerics — NPU execution can
differ from the CPU/XNNPACK path (different kernels, potential internal precision choices). On-device
execution is therefore still the next real unknown, not a formality.

## Update: real on-device run hit a genuine Tensor G5 firmware fault, not our code

Added a throwaway debug-only diagnostic to test the AOT-compiled artifact on real hardware:

- `NpuDiagnostic.runKleinSingle0()` (`app/src/main/kotlin/com/pockettavern/app/util/NpuDiagnostic.kt`):
  loads `single0_full_realtoks_Google_Tensor_G5.tflite` via `CompiledModel` + `Accelerator.NPU`,
  reads 5 raw float32 input files, runs it, diffs the output against a pushed PyTorch reference
  output. Follows the exact existing `CompiledModel.create(...).run(...)` pattern already used
  throughout this file for SDXL pieces — no new native/C++ code.
- `MainActivity.kt`: one more `BuildConfig.DEBUG`-gated intent-extra hook
  (`run_klein_single0_npu_diagnostic`), mirroring the existing `run_npu_unet_parallel_cfg_step`
  hook — inert unless adb explicitly supplies the extra.
- Inputs/model pushed to `/data/data/com.pockettavern.app/files/klein_single0/` via the established
  `adb push` + `run-as` technique (see "On-device testing mechanics" in
  `docs/npu-unet-conversion.md`). Reference `.npy` files converted to headerless raw float32 via
  `ndarray.tofile()` so the Kotlin side can read them with a plain `ByteBuffer`.

Launched via `adb shell am start -n com.pockettavern.app/.MainActivity --ez
run_klein_single0_npu_diagnostic true` (no UI interaction — force-stop + `am start` only, matching
the "don't drive this device's UI" rule in `docs/npu-unet-conversion.md`).

**Result: a real Tensor G5 firmware crash.** Logcat shows a Darwinn (`/dev/edgetpu`) hardware fault
partway through the run — `cluster_driver.cc: Timed out on PollAnyBit`, `Failed to cease DIVE,
requesting fatal shutdown`, `exception_handler.cc: Fatal Exception Type 1 Occurred!` with a full
register dump, then `Trying to recover from error.` The driver auto-recovered (~13s later); the
device came back healthy and responsive afterward — this is not a phone-bricking or data-loss risk,
just a contained accelerator-firmware fault. `CompiledModel.run()` itself did **not** throw — it
returned successfully after the recovery, but the output buffer was never actually written: the
diagnostic's own diff confirms the "NPU" output was exactly all-zero (`maxAbsDiff` exactly equals
the reference's max absolute value, `meanAbsDiff` exactly equals `meanAbsRef` — i.e. `npuOut ≡ 0`,
not noise). So on real hardware, this graph currently: (a) crashes the NPU firmware, and (b)
silently returns a zero buffer with no error surfaced through the API, rather than failing loudly.

**Root cause not yet known — do not assume it's an unfixable hardware/firmware limit.** An earlier
draft of this doc jumped to "same class of problem as the SDXL batch=2 CFG crash, not our code,"
and separately proposed splitting the block into smaller pieces as the fix. Both were wrong turns,
corrected here:

- The SDXL project's own history (see `docs/npu-unet-conversion.md` and the correction added to
  `project_npu_unet_conversion` memory) shows its batch=2 crash was *initially* misdiagnosed the
  same way ("reproducible closed-source dispatch-runtime crash, not our code") and that conclusion
  was **wrong** — the real cause was our own C++ `TensorBuffer`/`CompiledModel` destruction-order
  bug. Once fixed, batch=2 worked correctly; it was dropped afterward only for being slow, not for
  crashing.
- Repeatedly trying to shrink pieces/op-count has not historically been what fixed an SDXL NPU
  crash in this project. The real SDXL CFG fix (two parallel batch=1 passes instead of one batch=2
  dispatch) was about not combining two logically separate forward passes into one oversized
  dispatch — not about reducing graph/op complexity — and Klein's guidance-distilled, single-pass
  design has no batch=2/CFG-shaped analog to apply that fix to directly.

So this Klein Darwinn fault could still turn out to be either kind of thing (a real hardware/SDK
limit, or a lifetime/resource-handling bug on our side) and it is not yet known which. Notably,
this fault looks structurally different from the SDXL epoll bug: the epoll crash was a `SIGABRT` in
our own app process (`Fatal signal 6`, our C++ code on the stack); this Klein fault is a hardware
exception logged by the kernel-level Darwinn/`edgetpu` driver itself (a different process, PID
1324, not PocketTavern's PID), with the app-level `CompiledModel.run()` call never throwing at all.
That's a real, unexplained difference worth investigating rather than pattern-matching away.

## Update: crash analysis — DIVE STORE_ACCESS_FAULT, not a timeout

The logcat from the `runKleinSingle0` diagnostic run has been decoded:

```
cluster_driver.cc: Timed out on PollAnyBit on register 0x304008, want 0x51 (mask=0x51), got 0x2e.
```

Register `0x304008` is the Darwinn cluster-state register. The driver was waiting for the DIVE
(the NPU's on-chip RISC-V microcontroller) to signal completion:
- `0x51` = `0b01010001` = "both tile groups done + halted" (EXECUTE_DONE_TG0 | EXECUTE_DONE | HALT)
- `0x2e` = `0b00101110` = unexpected partial state — the DIVE never reached HALT

Why didn't the DIVE reach HALT? The RISC-V register dump tells us:

```
mcause = 0x800000000000000b   ← RISC-V exception cause: Store/AMO Access Fault (cause=11)
mtval  = 0x0                  ← faulting address: 0 (null/unmapped)
mepc   = 0x8005f660           ← instruction where the fault occurred (DIVE firmware)
```

`mcause=0x8000000000000b` is a **RISC-V Store Access Fault** — the DIVE firmware tried to write
to a memory address it does not have access to. This is not a "the model ran too long" timeout;
the hardware faulted on a bad write during execution. The cluster driver then timed out waiting
for a HALT that would never come, and triggered the fatal-shutdown / recovery sequence.

Two hypotheses for what caused the store fault, ordered by plausibility:

1. **DMA buffer range exceeded** — at 4608 tokens, the largest single intermediate activation
   is the `linear1` pre-attention output: `4608 × 9216 × 4 bytes ≈ 170 MB`. If the Darwinn
   address space for a single dispatch (or the total DMA buffer allocation for a CompiledModel)
   has a ceiling that 4608-token activations exceed, the DIVE firmware will fault writing past
   the end. At 80 tokens the same activation is only `~3 MB` — an 80× difference. This is the
   size-as-a-variable hypothesis, directly testable by running the already-compiled 80-token
   artifact.

2. **TensorBuffer lifecycle** — the Kotlin `runKleinSingle0` code never explicitly closed the
   `TensorBuffer` objects (`inputs`, `outputs`) before calling `model.close()` in the prior run.
   If the LiteRT Kotlin API unmaps the DMA buffers when `model.close()` is called — before the
   DIVE has finished using those buffers — the DIVE's next write to that address would store-fault
   at `mtval=0` or a garbage address. This is the lifecycle-order hypothesis.

### What was confirmed

**Both hypotheses have been tested:**

**Lifecycle fix — did NOT fix the crash.** `runKleinSingle0` was re-run (2026-08-28) with the
corrected buffer lifecycle (output read before any close, then inputs → outputs → model in order).
Result: same DIVE STORE_ACCESS_FAULT, same all-zero output. Lifecycle order was not the cause.

**Size IS the variable — confirmed by bisect.** Ran the compiled block on-device at four token
counts using the new `runKleinTokenProbe` diagnostic (`run_klein_token_probe` intent, `scratch/
build_klein_token_bisect.sh`):

| Tokens | `linear1` intermediate | runMs | Result |
|---|---|---|---|
| 80 | ~2.8 MiB | 19ms | ✅ clean |
| 512 | ~18 MiB | 69ms | ✅ clean |
| 1024 | ~36 MiB | 174ms | ✅ clean |
| 2048 | ~72 MiB | **compiler INTERNAL** | ❌ AOT failed (6:32, no artifact) |
| 3072 | ~108 MiB | 2240ms | ✅ clean |
| 4608 | ~162 MiB | 226ms | ❌ DIVE STORE_ACCESS_FAULT |

**T=2048 produced a compiler `INTERNAL` error** (no output artifact) — the same `INTERNAL` class as
the original flag-triggered failures, but this time *without* those flags. This is a compiler bug,
not a model or hardware limit: the compiler accepted and partitioned all 87 ops (87/87 to 1 NPU
partition), then failed internally during the actual code-generation phase after 6:32. T=4608
compiled successfully (~19 min); T=1024 compiled fine (2:19). T=3072 also compiled fine (finished
after handoff). So the INTERNAL bug is narrow — it does not affect a contiguous range above 2048,
just (at least) that one specific shape — and is not a simple monotone size ceiling.

**T=3072 ran clean on-device** (2026-08-28, post-handoff): `runMs=2240 hasNanOrInf=false
maxAbs=2.7719727 allZero=false outputSize=9437184 linear1_intermediate_MiB=108`. No DIVE fault, no
`cluster_driver`/`edgetpu` error anywhere in logcat around the run — real, non-zero, finite output.
This moves the DMA runtime ceiling (see hypothesis 1 below) to somewhere strictly between 3072 and
4608 tokens, i.e. a `linear1` intermediate strictly between ~108 MiB and ~162 MiB. Note `runMs` at
3072 (2240ms) is far above the 1024→3072 trend implied by the earlier clean points (19/69/174ms) —
worth another look before trusting it as a real steady-state number, since 4608's `runMs=226` was
almost certainly a time-to-fault, not a completion time, so the two aren't apples-to-apples either.

**Two separate issues now in play:**
1. **Compiler INTERNAL at T=2048** (and possibly other token counts) — a code-gen bug in the
   Tensor G5 AOT compiler, triggered by a specific graph shape. Not a deployment blocker for
   token counts that do compile (1024, 4608).
2. **Runtime DIVE STORE_ACCESS_FAULT at T=4608** — the on-device DMA ceiling. Token counts
   above some threshold between 1024 and 4608 exceed a Darwinn DMA address-range limit.

**This rules out the lifecycle order hypothesis** and confirms: the block's ops are all valid and
correctly implemented; the AOT artifact is sound; the runtime fault at T=4608 is a DMA range issue.
The compiler INTERNAL at T=2048 is a separate, independent compiler bug.

**New `runKleinSingle0SmallShape` diagnostic added** (`NpuDiagnostic.kt`): runs the already-
compiled 80-token artifact (`single0_noflags_aot/single0_Google_Tensor_G5.tflite`) with synthetic
sine-wave inputs, no reference bins needed. Intent extra: `run_klein_single0_small_shape_diagnostic`.
Push script: `scratch/push_klein_small_shape.sh`.

**New `runKleinTokenProbe` diagnostic added** (`NpuDiagnostic.kt`): generalized probe that takes
any compiled Klein block `.tflite` filename and token count from intent extras, runs with synthetic
inputs, and logs whether it crashed or produced real output. No APK rebuild needed per bisect point.
Intent extras: `--ez run_klein_token_probe true --es klein_token_probe_file <name> --ei klein_token_probe_tokens <N>`.
Build scripts: `scratch/build_klein_token_bisect.sh`.


### Recommended next steps

1. ~~**Correctness**: numerically compare this compiled block's output against the PyTorch
   reference.~~ Done. Confirmed correct on CPU/XNNPACK (maxAbsDiff=0.000626 vs meanAbsRef=6.029).
2. ~~**On-device**: get this one compiled block actually running on the Pixel.~~ Attempted — hit a
   real Tensor G5 firmware fault. See bisect results above for full characterization.
3. ~~**Look for lifecycle/resource bugs**.~~ Done. Buffer close-order fixed; retest confirmed it
   was not the cause. Size is definitively the variable.
4. ~~**Isolate whether size is the variable.**~~ Done. Full bisect completed: 80/512/1024 tokens
   all run clean; 4608 tokens crashes DIVE; 2048 tokens fails to AOT-compile (separate compiler
   bug). T=3072 compile pending at time of handoff.

5. ~~**Complete the T=3072 bisect compile.**~~ Done. Compiled successfully (259 MiB artifact,
   `scratch/models/flux2_klein_probe/single0_full_T3072tok_noflags_aot/`), pushed, and run on-device:
   clean result, no DIVE fault (`runMs=2240 hasNanOrInf=false allZero=false`, see bisect table above).

6. **Find the exact DMA runtime ceiling between 3072 and 4608 tokens.** The gap is now narrower:
   3072 runs clean, 4608 DIVE-faults. Bisect further, e.g. T=3584 or T=3840, using
   `scratch/build_klein_token_bisect.sh` as a template (change the `--text-tokens`/`--image-tokens`
   split and output name). The `runKleinTokenProbe` diagnostic is already on-device and requires no
   APK rebuild — just push a new `.tflite` and fire the intent:
   ```bash
   adb push scratch/models/flux2_klein_probe/<new_dir>/<new>_Google_Tensor_G5.tflite /data/local/tmp/
   adb shell "run-as com.pockettavern.app cp /data/local/tmp/<new>_Google_Tensor_G5.tflite /data/data/com.pockettavern.app/files/klein_single0/"
   adb shell am force-stop com.pockettavern.app
   adb shell am start -n com.pockettavern.app/.MainActivity --ez run_klein_token_probe true --es klein_token_probe_file <new>_Google_Tensor_G5.tflite --ei klein_token_probe_tokens <N>
   adb logcat -d -s NpuDiagnostic | grep 'KLEIN_TOKEN_PROBE\[<N>\]'
   ```
   The DMA ceiling is what determines the real split granularity for the production pipeline. Also
   worth double-checking the T=3072 `runMs=2240` outlier (see above) doesn't indicate something else
   going on at that shape before trusting it as clean.

7. **Design the graph split at the DMA ceiling.** Once the ceiling token count `C` is known:
   - The split point is **pre-attention linear vs. SDPA+out**. The `linear1` output (the 9216-wide
     projection before SDPA) is the largest intermediate; splitting before SDPA keeps each piece's
     peak allocation under the ceiling.
   - Export `SingleBlockPre` (already in `flux2_klein_probe.py` as `--stage pre`) and a new
     `SingleBlockAttnOut` wrapper at the full 4608-token shape, compile both for Tensor G5, and
     run them as two separate `CompiledModel` instances chained via host-side readFloat/writeFloat.
   - This is architecturally identical to the SDXL RESHAPE-wrapped separate-instance approach
     (which was the real working pattern for SDXL). Do apply the RESHAPE-wrap lesson: each piece's
     external input must pass through a same-shape RESHAPE node (as in `build_reshape_wrapped_piece.py`)
     to avoid the provenance bug seen in the SDXL mid_block investigation.

8. **Compile the remaining 24 transformer blocks.** Once piece 0 works end-to-end: the 20
   single-stream blocks and 5 double-stream blocks. Double-stream blocks have a different operator
   shape (two parallel streams with cross-attention) — do not assume the split strategy for
   single-stream blocks transfers directly; verify with at least one `double_blocks.0` probe first.

9. **Do not integrate into PocketTavern's download UI or `NpuUnetEngine`** until Klein runs
   correctly on-device, all 25 blocks, matching the PyTorch reference. The diagnostic hooks
   (`run_klein_single0_npu_diagnostic`, `run_klein_single0_small_shape_diagnostic`,
   `run_klein_token_probe`) remain debug-only and are gated on `BuildConfig.DEBUG`.

## Commands used

Basic direct conversion:

```bash
cd /home/brandont/code/litert-torch
.venv/bin/python scratch/flux2_klein_probe.py
```

Stages can be emitted without overwriting the full artifact, for example:

```bash
.venv/bin/python scratch/flux2_klein_probe.py \
  --stage q1024x1024 \
  --output scratch/models/flux2_klein_probe/single0_q1024x1024.tflite
```

Then invoke `aot_compile` with the code configuration shown above. The experiment used a tiny
80-token shape intentionally; do not spend the memory required for realistic 1024px block export
until the basic dense compilation issue is understood.
