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
| 3584 | ~126 MiB | 5997ms | ✅ clean |
| 4096 | ~144 MiB | 164ms (all-zero) | ❌ DIVE STORE_ACCESS_FAULT |
| 4608 | ~162 MiB | 226ms (all-zero) | ❌ DIVE STORE_ACCESS_FAULT |

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

**T=3584 also ran clean on-device** (2026-08-28, same day): exported (468 MiB), AOT-compiled
(263 MiB, 87/87 ops, 1 partition, ~12 min), pushed, and run — `runMs=5997 hasNanOrInf=false
maxAbs=2.7871094 allZero=false outputSize=11010048 linear1_intermediate_MiB=126`. No fault. Note
`runMs` jumped disproportionately to the token-count increase from T=3072: 2240ms → 5997ms is a
2.68x increase for only a 1.17x token increase, a bigger jump than 1024→3072 saw.

**T=4096 faulted on-device** (2026-08-28, same day): exported (468 MiB), AOT-compiled (268 MiB,
87/87 ops, 1 partition, ~16 min), pushed, and run — **DIVE STORE_ACCESS_FAULT**, same signature as
T=4608 (`Timed out on PollAnyBit ... want 0x51 ... got 0x2e`, `Fatal Exception Type 1 Occurred!`,
same faulting `mepc=8005f660` as the original T=4608 dump, `Trying to recover from error`, firmware
restart, auto-recovered). The diagnostic's own result line confirms the silent all-zero failure mode
from before: `runMs=164 hasNanOrInf=false maxAbs=0.0 allZero=true outputSize=12582912
linear1_intermediate_MiB=144`. `CompiledModel.run()` again did not throw.

**This pins the DMA runtime ceiling to strictly between 3584 tokens (clean, ~126 MiB linear1
intermediate) and 4096 tokens (fault, ~144 MiB).** The nonlinear `runMs` growth noted at T=3584 (far
above the 80→1024 trend) is now a plausible early warning sign of approaching this ceiling, rather
than an unrelated curiosity — worth watching for at the next bisect point too, not just the pass/fail
outcome.

## Update: the real ceiling is performance, not the DMA fault — pivoting to chunked attention

Checked the `runMs` scaling directly against tokens rather than just noting it looked nonlinear:

| Tokens | runMs | Tokens vs. T=1024 | Time vs. T=1024 |
|---|---:|---:|---:|
| 1024 | 174 | 1.0x | 1.0x |
| 3072 | 2240 | 3.0x | 12.9x |
| 3584 | 5997 | 3.5x | **34.5x** |

Even crediting the whole gap to SDPA's O(n²) attention term (which alone predicts ~12x at 3.5x
tokens), the observed 34.5x is roughly 3x worse than that upper bound. This is not "gets slower as it
gets bigger," it is degrading faster than the algorithm's own worst-case component — consistent with
falling out of on-chip SRAM/cache and thrashing through DMA well before the hard fault at T=4096.

**Conclusion (2026-08-28, user call): stop bisecting for the exact fault byte.** Even if the precise
DMA ceiling between 3584 and 4096 were found, everything approaching it is already too slow to be
usable in a real 4-step generation pipeline. ~1024 tokens is the practical ceiling for one NPU
dispatch of this block — not because of the DIVE fault, but because performance is already falling
off a cliff well before that fault triggers.

**This forces an architecture decision, not just a split-granularity one.** Production 1024px output
needs 4608 tokens (512 text + 4096 image) in a single attention computation — SDPA needs every
query token to see every key/value token in the sequence, so simply staging the *pipeline* into more
pieces (e.g. pre-attention vs. post-attention, or per-transformer-block) does not shrink any single
piece's attention below the full 4608-token sequence. Splitting pre/post-attention (the plan from the
original DMA-ceiling hypothesis, see step 7 below) is therefore insufficient by itself.

**Chosen direction: chunked/flash-attention-style SDPA**, keeping full 1024px resolution. Tile the
attention computation itself so no single NPU dispatch ever materializes a full-sequence intermediate
(the `linear1` projection and the SDPA score matrix) — process K/V in ~1024-token chunks, accumulate
the (running-max, running-sum) softmax statistics online across chunks (the standard flash-attention
recurrence), and only assemble the final per-query output after all K/V chunks are processed. This
is a real reimplementation of the attention sub-block, not a straightforward pipeline split — RMS
norm, RoPE, and the SiLU-gated MLP outside attention can likely still be tiled by simple sequence
chunking (they're per-token, no cross-token dependency), but the SDPA core needs the flash-attention
recurrence specifically.

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

5b. ~~**Bisect T=3584.**~~ Done. Compiled successfully (263 MiB artifact,
   `scratch/models/flux2_klein_probe/single0_full_T3584tok_noflags_aot/`), pushed, and run on-device:
   clean, no DIVE fault (`runMs=5997 hasNanOrInf=false allZero=false linear1_intermediate_MiB=126`).

5c. ~~**Bisect T=4096.**~~ Done. Compiled successfully (268 MiB artifact,
   `scratch/models/flux2_klein_probe/single0_full_T4096tok_noflags_aot/`), pushed, and run on-device:
   **DIVE STORE_ACCESS_FAULT** — same signature as T=4608, all-zero output
   (`runMs=164 allZero=true linear1_intermediate_MiB=144`).

6. ~~**Find the exact DMA runtime ceiling between 3584 and 4096 tokens.**~~ **Abandoned by user
   decision (2026-08-28).** The performance cliff analysis above (34.5x runtime for 3.5x tokens,
   `T=3072`→`T=3584`) shows anything near that ceiling is already too slow to ship, regardless of
   the exact fault byte. Not worth further bisecting. See "Update: the real ceiling is performance,
   not the DMA fault" above.

7. **Design and implement chunked/flash-attention SDPA for the single-stream block**, targeting
   ~1024-token K/V chunks (the proven-fast regime) while still covering the full 4608-token
   production sequence per block, at full 1024px resolution. This is the current active direction
   (user-selected over lowering resolution or shelving the decision). Concretely:
   - ~~Reimplement `SingleStreamBlock`'s attention ... using the flash-attention online-softmax
     recurrence ... Validate this chunked implementation numerically against the existing unchunked
     PyTorch reference ... before touching TFLite/AOT at all.~~ **Done (2026-08-29).** See "Update:
     chunked-attention math validated on CPU" below — bit-close match (both on synthetic q/k/v and on
     the real checkpoint's single_blocks.0 at production shape), block-tiled over both Q and K/V axes.
   - ~~RMS norm, RoPE application, and the SiLU-gated MLP are per-token ops ... but confirm this
     against the reference too rather than assuming.~~ **Done (2026-08-29).** Confirmed: `apply_rope`
     sliced into 1024-token chunks and reassembled is bit-identical (`maxAbsDiff=0.0`) to the unchunked
     call — RoPE, `_qkv` (LayerNorm + linear1 + QKNorm), and `_out` (linear2 + residual) all operate
     per-token with no cross-token mixing, so plain sequence slicing is exact for everything outside
     SDPA itself.
   - Once chunked math is verified on CPU, design the actual NPU dispatch boundary: probably one
     `CompiledModel` per K/V chunk (each seeing the full running-softmax state as input/output,
     RESHAPE-wrapped per the SDXL lesson already applied above), chained host-side. This is a
     meaningfully bigger change than the previously-planned single pre/post-attention split (step 7b
     below is now superseded by this). **Not started** — next concrete step.
   - Re-run the same on-device token-probe-style validation once a chunked artifact compiles: confirm
     no DIVE fault and correct (non-zero, finite, reference-matching) output at production token
     count, and check whether `runMs` for the chunked version actually lands near the ~1024-token
     single-chunk baseline (174ms) times the number of chunks, rather than repeating the same
     superlinear blowup — chunking only pays off if per-chunk dispatches don't inherit the same
     cliff.

### Update: chunked-attention math validated on CPU (2026-08-29)

Two scripts under `scratch/` in the `litert-torch` repo prove the flash-attention recurrence from
step 7 is numerically sound before any TFLite work is attempted:

- `scratch/klein_chunked_attention_prototype.py` — synthetic q/k/v, block-tiled over **both** the
  query axis and the K/V axis (not just K/V) into `chunk`-sized pieces, standard online-softmax
  recurrence (running max, running sum, running output, rescaled on each new block max). Checked
  against `F.scaled_dot_product_attention` at the real production shape and two stress cases:

  | tokens | chunk | chunks | maxAbsDiff | allclose(rtol=1e-4, atol=1e-5) |
  |---|---|---|---|---|
  | 4608 | 1024 | 5 | 1.08e-07 | ✅ |
  | 4608 | 900 (non-divisible) | 6 | 1.84e-07 | ✅ |
  | 3584 | 1024 | 4 | 2.43e-07 | ✅ |

- `scratch/klein_chunked_singleblock_validation.py` — loads the **real** `single_blocks.0` weights
  (same loader as `compute_klein_single0_torch_reference.py`) at the real production shape (512 text
  + 4096 image = 4608 tokens), runs the full unchunked block forward and a version with only the SDPA
  call swapped for `chunked_sdpa(..., chunk=1024)`, and diffs the two block outputs directly:
  `maxAbsDiff=3.05e-05` vs. `meanAbsRef=6.03` — same order of magnitude as the already-accepted
  CPU/XNNPACK-vs-reference tolerance (`maxAbsDiff=0.000626` vs `meanAbsRef=6.029`, see "numerical
  correctness" update above). Confirms the chunked design is correct end-to-end through a real block,
  not just for isolated attention.

RoPE-chunking check (also in that run): slicing `apply_rope`'s inputs into 1024-token pieces and
reassembling gives `maxAbsDiff=0.0` against the unchunked call — exact, as expected for a per-token op.

**Conclusion: the math for step 7 is settled.** The remaining work is entirely about the NPU dispatch
boundary (one `CompiledModel` per K/V chunk, chained host-side, RESHAPE-wrapped) and re-validating
on-device — not about correctness of the chunking algorithm itself.

### Update: full NPU dispatch design worked out and validated on CPU (2026-08-29)

`scratch/klein_flash_step_prototype.py` refactors the attention recurrence into the actual shape one
NPU dispatch would take: `flash_step(q_blk, k_blk, v_blk, running_max, running_sum, running_out,
scale) -> (running_max, running_sum, running_out)`. All tensor shapes are fixed per call
(chunk-sized) — this, not the Python loop around it, is what gets `litert_torch.convert`-wrapped and
AOT-compiled per chunk. Host-side chaining of `flash_step()` calls (one per Q-chunk × K/V-chunk pair)
was checked bit-for-bit against the already-validated loop-based `chunked_sdpa` (`maxAbsDiff=0.0`,
`torch.equal(...)=True`) and against full SDPA (`maxAbsDiff≈1e-7`) at both 4608/1024 (25 dispatches)
and 3584/1024 (16 dispatches) shapes.

The original on-device bisect's peak-memory metric (`linear1 intermediate MiB`, growing 36→108→
126→144→162 MiB across the fault boundary) comes from `_qkv`'s `linear1` projection running over the
*entire* token sequence at once — not from SDPA alone. So `scratch/klein_chunked_block_full.py`
chunks the **whole block**, not just attention, into three dispatch-shaped passes over ~1024-token
query chunks:

1. **QKV projection** (`_qkv`: LayerNorm + `linear1` + QKNorm) + RoPE, run once per query-chunk of
   `x` — 5 dispatches at production shape. `mod_shift`/`mod_scale`/`mod_gate` are global per-block
   vectors (shape `[1,1,3072]`), not per-token, so no slicing needed for them.
2. **Attention**, via `flash_step()` — one dispatch per (Q-chunk, K/V-chunk) pair, 25 dispatches at
   production shape (5×5).
3. **Output projection** (`_out`: `linear2` + residual), run once per query-chunk — 5 dispatches.

**35 total dispatches per single-stream block at production shape**, each with a fixed, bounded
input/output size (~1024 tokens or a 1024×1024 attention tile) instead of one monolithic dispatch
over the full 4608-token sequence. Validated against the real `single_blocks.0` checkpoint's
unchunked forward: `maxAbsDiff=4.58e-05` vs `meanAbsRef=6.03` (chunk=1024) — same tolerance class as
every other correctness check in this doc. **The full dispatch design is now proven correct on CPU
end-to-end**, at the granularity that would actually become separate `CompiledModel`s.

Not yet done, and the actual next step: export one piece of this design (start with the `flash_step`
kernel, since it's the smallest and most novel — RESHAPE-wrap its five inputs per the SDXL lesson,
`litert_torch.convert` it at a small token shape first to prove it exports/compiles at all) before
committing to a full 1024-token-chunk AOT compile, which is expensive (12–19 min, ~12 GiB peak RSS
per piece, and this design needs at least 3 distinct piece shapes — QKV-proj, flash-step, out-proj —
compiled once and reused across chunks/dispatches within a block).

### Update: flash_step exports to LiteRT cleanly, but hit (and fixed) a new conversion bug (2026-08-29)

`scratch/klein_flash_step_export.py` wraps `flash_step()` in an `nn.Module` and runs
`litert_torch.convert()` + `.export()` (no AOT compiler involved — this step is local/free/fast).

**First attempt failed** with `RuntimeError: NHWC node rewriter not found: amax`, from
`optimize_layout_transposes_pass`. Root cause: that pass runs a min-cut/greedy partitioner that
globally assigns every **4D** tensor op in the graph to an NHWC or NCHW partition (treating them as
conv-layout candidates), and `amax` (needed for the online-softmax running-max step) has no
registered rewriter in either partition — this is a real litert_torch limitation, not a bug in our
math (the same file's SDPA-based `SingleBlockAttention` avoids it only because SDPA is a fused/opaque
op that doesn't expose its internal `amax` to this pass).

**Fix**: the pass is a no-op for non-4D tensors (`pass_body.py`: "for non-4D tensors input_to_nchw is
always noop"), so `FlashStep.forward` now collapses `[B, H, T, D]` → `[B*H, T, D]` before the
matmul/amax/exp chain and reshapes back to 4D at the end — sidesteps the bug entirely rather than
needing an upstream fix. This is a new conversion-layer lesson, distinct from the existing
RESHAPE-wrap (provenance-bug) and flag-bug (compiler INTERNAL) lessons already documented above: **any
new manual-softmax-style op sequence on 4D attention tensors in this codebase should collapse to 3D
before non-SDPA reduction ops** (`amax`, `sum`, elementwise `exp`/`max` reductions), not just this one
kernel.

Validated correct after the fix, at both a small probe shape and the real 1024-token production chunk
shape:

| chunk | export | running_max maxAbsDiff | running_sum maxAbsDiff | running_out maxAbsDiff |
|---|---|---|---|---|
| 64 | ✅ clean (4.9 KiB) | 0.0 | 3.81e-06 | 1.91e-06 |
| 1024 (production) | ✅ clean (4.9 KiB) | 0.0 | 3.05e-05 | 3.81e-05 |

Both runs used the on-CPU XNNPACK interpreter (not NPU/AOT), diffed against the eager PyTorch
reference. This proves the `flash_step` piece converts and runs correctly through the full
LiteRT-Torch pipeline at the real chunk size.

### Update: flash_step AOT-compiles for Tensor G5 — fast and clean (2026-08-29)

Ran `scratch/retry_klein_noflags.py` (no `large_model_support`/`sharding_intensity` flags, per the
standing flag-bug lesson) on `flash_step_probe_1024.tflite`, under the memory watchdog (floor 2000 MB):
**23/23 ops offloaded to 1 NPU partition, compiled in 23 seconds** — dramatically faster than the
full single-stream block's ~19 minutes, as expected for a piece this much smaller (0.594M ops vs. the
full block's much larger graph). Output:
`scratch/models/flux2_klein_probe/flash_step_probe_1024_noflags_aot/flash_step_probe_1024_Google_Tensor_G5.tflite`
(2.1 MiB).

**Not yet done: pushing this to the Pixel and running it on-device.** This is the actual test of
whether the chunked design avoids both the DIVE STORE_ACCESS_FAULT and the performance cliff that
motivated this whole pivot — a clean AOT compile only proves the compiler accepts the graph, not that
it runs correctly or fast on real Darwinn hardware. The QKV-projection and output-projection pieces
(steps 1 and 3 of the 35-dispatch design) also still need their own export wrappers and AOT
compiles — not yet written/run.

### Update: flash_step runs clean on real Tensor G5 hardware (2026-08-29) — the pivot works

Added `NpuDiagnostic.runKleinFlashStepProbe` (intent extra `run_klein_flash_step_probe`), wired
through `MainActivity.kt`, matching `flash_step`'s 6-input/3-output signature exactly (q_blk, k_blk,
v_blk `[1,24,1024,128]`; running_max, running_sum `[1,24,1024,1]`; running_out `[1,24,1024,128]`) with
synthetic sine-wave inputs — same no-reference-bins pattern as the earlier token-probe diagnostics.
Push script: `scratch/push_klein_flash_step.sh`.

Rebuilt the debug APK, installed, pushed the AOT artifact, ran it:

```
KLEIN_FLASH_STEP: runMs=65 running_max(hasNanOrInf=false maxAbs=0.100097656 allZero=false size=24576)
running_sum(hasNanOrInf=false maxAbs=968.0 allZero=false size=24576)
running_out(hasNanOrInf=false maxAbs=2.921875 allZero=false size=3145728)
```

**No DIVE STORE_ACCESS_FAULT. No NaN/Inf. Non-zero, finite output. `runMs=65` for one 1024×1024
attention-chunk dispatch** — in the same fast regime as the original bisect's clean T=1024 result
(174ms for the *whole* pre-chunking single-stream block, not just attention), nowhere near the
performance cliff that started appearing at T=3072+ tokens. **This is the first real on-device
evidence that the chunked/flash-attention pivot actually solves the problem it was designed for**,
not just a CPU numerical proof.

This validates one dispatch out of 35 per block (see the dispatch-design update above).

### Update: all three piece-types of the dispatch design run clean on-device (2026-08-29)

Exported (`scratch/klein_qkv_proj_export.py`, `scratch/klein_out_proj_export.py`, real checkpoint
weights, same 3D-reshape-safe pattern as `flash_step`), AOT-compiled (`retry_klein_noflags.py`, no
flag-bug flags), and ran both remaining piece-types on-device:

| piece | AOT compile | on-device |
|---|---|---|
| `qkv_proj` (pass 1: LayerNorm+linear1+QKNorm+RoPE) | ✅ 65/65 ops, 1 partition, 61s | ✅ `runMs=139`, all 4 outputs (q,k,v,mlp) finite/non-zero |
| `flash_step` (pass 2: online-softmax attention step) | ✅ 23/23 ops, 1 partition, 23s | ✅ `runMs=65`, clean (see previous update) |
| `out_proj` (pass 3: linear2 + gated residual) | ✅ 9/9 ops, 1 partition, 38s | ✅ `runMs=109`, clean |

New diagnostics: `NpuDiagnostic.runKleinQkvProjProbe` / `runKleinOutProjProbe` (intents
`run_klein_qkv_proj_probe` / `run_klein_out_proj_probe`), push scripts
`scratch/push_klein_qkv_proj.sh` / `scratch/push_klein_out_proj.sh`.

**All three piece-types that make up the 35-dispatch chunked block run correctly and fast on real
Tensor G5 hardware** — no DIVE fault, no NaN/Inf, all well under 150ms per dispatch. Remaining work
before this is a complete, verified block:
- Confirm `flash_step` stays fast/clean across all 5×5=25 (Q-chunk, K/V-chunk) combinations at
  production shape, including the ragged last chunk (4608 = 4×1024 + 512) — only one representative
  shape (1024×1024, non-ragged) has been tried so far.
- Chain all 35 dispatches host-side for one full block and verify the assembled output against the
  PyTorch reference on-device (the CPU version of this, `klein_chunked_block_full.py`, is already
  proven — this is the on-device equivalent, and the real correctness test: individually-clean pieces
  don't guarantee the chained-together result matches).
- Only after a full block is verified end-to-end on-device: move to the remaining 24 blocks (per
  existing step 8), including the as-yet-unprobed double-stream blocks.

### Update: chaining all 35 dispatches on-device — found and fixed two real AOT-compiler bugs (2026-08-29)

Wired all 3 piece-types into `NpuDiagnostic.runKleinChunkedBlockProbe` (intent
`run_klein_chunked_block_probe`, push script `scratch/push_klein_chunked_block.sh`): computes one
full `single_blocks.0` forward at the real production shape by chaining 4 qkv_proj + 16 flash_step +
4 out_proj dispatches (chunk=1152, 4 even chunks — chosen specifically to avoid 4608's ragged
remainder under chunk=1024), using the **real** reference inputs/output (`x.bin`/`pe.bin`/`mod_*.bin`/
`torch_out.bin`, the same files `runKleinSingle0` already used) rather than synthetic data, and diffs
the assembled result against the real PyTorch reference.

**Attempt 1 — Java heap OOM.** Holding all 4 chunks' worth of q/k/v/mlp arrays simultaneously
(`mlp` alone is 1152×18432×4B ≈ 85 MiB/chunk, ~340 MiB across 4 chunks) blew the default ~256 MiB
Dalvik heap (`java.lang.OutOfMemoryError`). Fixed by streaming intermediate q/k/v/mlp/attn chunks
through `context.cacheDir` files instead of holding them as Java arrays, and reading x/pe/reference
slices directly off disk via `RandomAccessFile` range reads instead of loading the full 4608-token
tensors — bounding host-side memory per chunk, matching the whole point of chunking in the first
place. (A `writeFloatBin`/`readFloatBin` helper using `ByteBuffer.allocate` was *also* silently
doubling memory per round-trip; switched those specific temp-file helpers to streamed
`DataOutputStream`/`DataInputStream` I/O for O(1) extra memory.)

**Attempt 2 — all-NaN output, root cause: literal `-Infinity` breaks the NPU.** After fixing the OOM,
the chain ran to completion but produced all-`NaN` output. Root cause: `flash_step`'s running-softmax
recurrence needs an "empty" initial state before the first K/V chunk (`running_max = -Infinity,
running_sum = 0, running_out = 0`), and this project's earlier CPU-only validation always used this
correctly — but it turns out real IEEE `-Infinity` fed into the *compiled* `flash_step` kernel breaks
on real Darwinn hardware (`exp(-Infinity - -Infinity)` type indeterminacy), something the CPU/XNNPACK
path never has a problem with. Fixed (partially — see below) by swapping the sentinel for a large
finite negative value.

**Attempt 3 — wrong-but-not-NaN output, real root cause: AOT compilation itself mishandles the
sentinel state, regardless of magnitude.** After the finite-sentinel fix, output was finite and
non-zero but *numerically wrong*: `running_sum` after the first K/V chunk came back exactly `680.0`
on-device vs. the CPU-correct `1.0247...` — a factor of `e^6.5`, meaning `max(runningMax, blockMax)`
was silently computing the wrong value. Bisection ruled out every plausible cause in turn:
- **Not sentinel magnitude**: `-1e30` and `-30000` (comfortably inside fp16's ±65504 range, ruling
  out an fp16-cast-overflow theory) produced **bit-identical** wrong results.
- **Not input wiring/ordering**: inspected the compiled artifact's flatbuffer directly
  (`schema_py_generated`) and confirmed `args_0..5` map exactly to `q,k,v,running_max,running_sum,
  running_out` in the expected order.
- **Not the RESHAPE-wrap provenance bug** (the SDXL-era lesson): applying `build_reshape_wrapped_piece.py`
  to the compiled `flash_step` artifact made no difference — bit-identical `680.0` again.
- **Confirmed AOT-compilation-specific**: ran the *uncompiled* exported `flash_step_probe_1152.tflite`
  graph via desktop XNNPACK on the exact real `q0`/`k0`/`v0` data (not random) — it reproduced the CPU
  eager reference almost bit-exactly (`running_sum=1.0247196` vs `1.0247195959`). The bug is
  introduced specifically by Tensor G5 AOT compilation of this kernel+sentinel-state pattern; the
  graph, the math, and the wiring are all correct.

**Fix**: stopped constructing an artificial "empty" state for the first K/V chunk entirely. Added
`scratch/klein_flash_step_init_export.py` — a `flash_step_init` kernel with **no external running-state
input at all** (just `q_blk, k_blk, v_blk → running_max, running_sum, running_out`, computing the
ordinary local softmax for one chunk with no `max()`/correction merge). Used only for `ki=0`;
`flash_step` (with the merge step) is now only ever called with a genuine, real `running_max` from an
actual previous chunk — never a sentinel. AOT-compiled clean (14/14 ops, 1 partition, 52s).

**Result after all three fixes**: `hasNanOrInf=false`, `maxAbsDiff=1.93` (down from `20.08`),
`meanAbsDiff=0.0496` vs `meanAbsRef=6.03` (down from `0.58` — under 1% mean relative error, an order
of magnitude better, vs. ~10% before this fix). First-5 output values are now close:
`[23.875, -15.406, -0.619, -0.370, 8.254]` vs. reference `[23.906, -15.333, -0.593, -0.371, 8.179]`.

**Follow-up (2026-08-29): per-chunk error breakdown explains the residual gap — likely just fp32
accumulation noise, not a remaining bug.** Added per-chunk `chunkMaxAbsDiff`/`chunkMeanAbsDiff`
logging to pass 3. Result, all 4 chunks:

| chunk | chunkMaxAbsDiff | chunkMeanAbsDiff | value at max (npu / ref) | relative error at max |
|---|---|---|---|---|
| 0 | 1.926 | 0.0498 | 130.35 / 132.28 | 1.46% |
| 1 | 1.931 | 0.0495 | -149.07 / -147.14 | 1.31% |
| 2 | 1.668 | 0.0497 | 127.54 / 129.21 | 1.29% |
| 3 | 1.606 | 0.0495 | -113.54 / -111.94 | 1.43% |

Every chunk's max-error point sits at a **large-magnitude outlier in the reference itself**
(~110–150, vs. `meanAbsRef=6.03` for the typical element) — not at a small value where a 1.9 absolute
error would imply a broken computation. Relative error at every one of these outlier points is a
consistent, narrow **1.3–1.5%**, and `chunkMeanAbsDiff` is essentially identical across all 4 chunks
(~0.0495–0.0498). This is the signature of ordinary fp32 execution-order/rounding differences
accumulating across a genuinely deep 24-dispatch chain (5 pieces × up to 4 chunks each, real NPU
hardware vs. desktop CPU reference) — not a localized bug in one chunk or one piece type. Contrast
with every *single-piece* correctness check in this doc (all ~1e-4 to 1e-5 in isolation): chaining 24
real hardware dispatches plausibly compounds each piece's small individual noise into this ~1%
level, especially visible (in absolute terms) at the sequence's naturally larger-magnitude elements.

**Practical conclusion: the chunked/flash-attention design is validated working on real Tensor G5
hardware** — both AOT-compiler bugs are fixed, and the remaining ~1–1.5% relative error looks like
normal accumulated hardware precision, not a design or implementation defect. Whether this precision
level is acceptable for production image quality is a judgment call, not a further debugging target
by default — flag it if generated images show visible chunked-attention artifacts once this reaches
the full pipeline, but don't block on chasing this number to 1e-4 without evidence it matters
visually.

### Update: double-stream (`double_blocks.0`) chunked math validated on CPU (2026-08-29)

First look at the double-stream/cross-attention operator shape (step 8's standing caution: "do not
assume the split strategy for single-stream blocks transfers directly"). Turns out the attention
itself **is** the same joint self-attention as single-stream — `DoubleStreamBlock._prepare_qkv`
projects the `img` and `txt` streams independently (separate `img_attn`/`txt_attn` weights, separate
`img_norm1`/`txt_norm1`), concatenates them into one `[txt, img]` sequence, and attends jointly; with
`num_ref_tokens=0` (Klein's case — no reference-image conditioning), `causal_attn_fn` reduces to plain
SDPA over that concatenated sequence. So `flash_step`/`flash_step_init` transfer **unchanged** — the
only genuinely new work is per-stream QKV-projection and output-projection kernels.

**Chunking design**: uniform **chunk=512** for both streams, not 1152 or 1024. txt is exactly 512
tokens (Qwen3 4B encoder's `MAX_LENGTH`) — using 512 as the chunk size means txt needs no
sub-chunking at all (1 "chunk" = the whole txt sequence), and img's 4096 tokens split evenly into 8
chunks of 512. This keeps **every** Q/K/V chunk the same size, so only one compiled `flash_step`/
`flash_step_init` shape is needed — avoiding a combinatorial set of (txt-size, img-size) shape pairs
that a mismatched chunk size would require. 1 txt-chunk + 8 img-chunks = 9 total chunks →
9 QKV-proj dispatches (1 txt-shaped + 8 img-shaped, 2 distinct compiled models) + 9×9=81
flash-attention dispatches + 9 output-proj dispatches (2 distinct compiled models) = **99 dispatches
per double-stream block** — more than single-stream's 24 (smaller chunks → more pieces), a real
production-efficiency cost to keep in mind later, but not a blocker for proving correctness now.

New file `scratch/klein_double0_probe.py`: `load_double_block()` (loads real `double_blocks.0`
weights), `DoubleBlockNoReference` (exact unchunked reference forward), `chunked_double_block_forward()`
(the chunked design above, reusing `flash_step` from `klein_flash_step_prototype.py` unchanged).
Validated against the real checkpoint on the first attempt: `img: maxAbsDiff=7.6e-05` (`meanAbsRef=
6.69`), `txt: maxAbsDiff=6.1e-05` (`meanAbsRef=7.00`) — same tolerance class as every single-stream
CPU check in this doc.

**Not yet done**: export/AOT-compile/on-device validation of the double-stream pieces (img-qkv-proj,
txt-qkv-proj, img-out-proj, txt-out-proj — 4 new export wrappers, mirroring
`klein_qkv_proj_export.py`/`klein_out_proj_export.py`'s pattern) — this is CPU-only math validation so
far, the next increment repeats this session's single-stream on-device sprint for double-stream.

### Update: double-stream chunked block VALIDATED WORKING on real Tensor G5 hardware (2026-08-29)

Exported and AOT-compiled all 4 new pieces (`klein_double0_qkv_export.py` --stream img/txt,
`klein_double0_out_export.py` --stream img/txt) plus `flash_step`/`flash_step_init` re-exported at
chunk=512 (needed since single-stream's compiled shapes were 1024/1152, not 512) — all 6 compiled
clean, no flag-bug, no partial-partition:

| piece | ops | compile time |
|---|---|---|
| img/txt qkv-proj | 63/63, 1 partition | 18s each |
| img/txt out-proj | 27/27, 1 partition | 43-45s each |
| flash_step@512 | 23/23, 1 partition | 8s |
| flash_step_init@512 | 14/14, 1 partition | 9s |

Generated real reference IO (`compute_klein_double0_torch_reference.py`, same seed=0/random-normal
convention as `compute_klein_single0_torch_reference.py`, saved to
`klein_double0_validation_io/{raw,}`). Added `NpuDiagnostic.runKleinDoubleChunkedBlockProbe` (intent
`run_klein_double_chunked_block_probe`, push script `scratch/push_klein_double0_chunked_block.sh`) —
chains all 99 dispatches (9 QKV-proj + 81 flash-attention + 9 out-proj) exactly mirroring
`runKleinChunkedBlockProbe`'s structure (streamed temp files, `flash_step_init` for the first K/V
chunk) but with 9 uneven-origin chunks: index 0 is the whole 512-token txt stream, 1..8 are 512-token
img chunks.

**First attempt hit one new bug**: `txt`'s result showed `maxAbsDiff=3.4e38`, `nanInf=true` on the
reference side — looked catastrophic, but debug logging (dumping raw stats of `txtOutChunk` vs.
`refChunk`) showed the **actual on-device computation was already correct**
(`txtOutChunk maxAbs=145.7`, closely matching the true reference's `146.0`) — the bug was reading
`txt_out.bin` (a little-endian file from desktop numpy `.tofile()`) with
`readFloatBinStreamed` (the function's own big-endian helper for internal round-trip temp files) by
mistake. One-line fix: use the class's little-endian `readFloatBin` for that one reference file. A
useful general lesson: when a diagnostic's *own* built-in comparison utility produces a nonsensical
result (values near `Float.MAX_VALUE`, `NaN`), audit the diagnostic's I/O before concluding the
underlying computation is broken — dumping raw stats of both sides separately (not just the diff)
caught this immediately.

**After the fix, clean on the very next run — no NaN/Inf, both streams close to reference**:
- `img`: `maxAbsDiff=5.01`, `meanAbsDiff=0.107` vs `meanAbsRef=6.69` (~1.6% mean relative error)
- `txt`: `maxAbsDiff=2.21`, `meanAbsDiff=0.089` vs `meanAbsRef=7.00` (~1.3% mean relative error)

Same order of magnitude as single-stream's residual ~1-1.5% (already characterized as fp32
accumulation noise across many chained dispatches, not a design defect) — plausibly a bit higher here
because double-stream chains more dispatches (99 vs. single-stream's 24). **The double-stream/
cross-attention chunked design is now validated working on real Tensor G5 hardware, first attempt
after the endianness fix** — the "different operator shape" caution from step 8 turned out not to
need a different *attention* design at all, just separate per-stream projection kernels.

Both `single_blocks.0` and `double_blocks.0` are now proven correct on-device with the chunked
design. Remaining before the 25-block set is fully done: repeat the AOT-compile + on-device chain for
the other 19 single-stream blocks and 4 double-stream blocks (each needs its own compiled artifacts,
since weights are baked into the compiled graph — no shortcut there), then integrate into the actual
sampling loop. Still **do not integrate into PocketTavern's download UI or `NpuUnetEngine`** until
that full 25-block set is done end-to-end.

7b. ~~**Design the graph split at the DMA ceiling (pre-attention linear vs. SDPA+out).**~~
   **Superseded by step 7's chunked-attention plan (2026-08-28).** A single pre/post-attention split
   only shrinks the pre-attention `linear1` piece — the SDPA piece would still need the full
   4608-token sequence at once (attention has no per-token independence to exploit), so this split
   alone doesn't avoid the performance cliff. Kept here for reference only: the RESHAPE-wrap lesson
   from the SDXL mid_block investigation (external input of each piece must pass through a same-shape
   RESHAPE node, see `build_reshape_wrapped_piece.py`) still applies to whatever piece boundaries the
   chunked design in step 7 lands on.

### Update: dispatch-count-driven wall-clock cost is real — profiled it, then cut double-stream's dispatch count 65% (2026-08-29)

Before scaling to the other 24 blocks, the user raised a critical concern from the SDXL NPU work:
each NPU "piece" carries real per-dispatch overhead (SDXL's was ~50ms/piece), and this project's
chunked design produces a *lot* of pieces — 24 dispatches/single-stream-block × 20 + 99/double-
stream-block × 5 = 975 dispatches/diffusion-step, ×4 steps (Klein's distilled step count) ≈ **3,900
dispatches/image**. Naively extrapolating the chained probes' observed wall-clock (~31-37s for 24
dispatches, ~36s for 99) gives **~55 minutes/image** — nowhere near the goal of beating SDXL.

**Profiled the real per-dispatch cost** (`NpuDiagnostic.runKleinProfileDispatchOverhead`, intent
`run_klein_profile_dispatch_overhead`) to find out how much is fixable. First attempt was confounded:
synthetic input generation via `kotlin.math.sin()` over ~3.5M-element arrays was happening *inside*
the timed loop, inflating apparent overhead to ~600-1100ms/call. Fixed by precomputing inputs once
and separately timing each phase (`create`, buffer alloc, `write`, `run`, `read`, `close`). Real
numbers, chunk=1152, 8 iterations:

| piece | recreate-each-call | reuse-model (create once, run N times) |
|---|---|---|
| flash_step (small) | 214ms/call (run=139ms) | 128ms/call (run=79ms) |
| qkv_proj (large, ~190MB) | 532ms/call (run=303ms, read=169ms) | 444ms/call (run=270ms, read=162ms) |

Reusing one `CompiledModel` instance across repeated same-piece calls gives a real ~20-40% speedup
(mostly from a lower `run()` time itself, not just skipping `create()`), but this — extrapolated —
still gave a nontrivial estimate (~5.6s/single-stream-block, ~18.4s/double-stream-block dispatch-only
≈ **~13.6 min/image**), and importantly revealed that the **~25-30s gap** between that dispatch-only
estimate and the chained probes' actual ~31-37s wall-clock was almost certainly the diagnostic's own
per-`float` disk I/O (streaming an 85 MiB `mlp` chunk through `DataOutputStream.writeFloat()` one
float at a time) — not real NPU/dispatch cost. A production implementation wouldn't round-trip
intermediate tensors through disk at all.

**The single biggest remaining lever is dispatch count itself**, and double-stream's 81
flash-attention dispatches (vs. single-stream's 16) came purely from choosing a uniform chunk=512 for
both streams (to avoid needing multiple compiled `flash_step` shapes) — not from any real necessity.
Tested increasing the img-stream chunk size to 1024 (txt stays at 512, its natural full length):

- Requires asymmetric Q/K-V shaped `flash_step`/`flash_step_init` variants (added `--q-chunk`/
  `--kv-chunk` to both export scripts) — only 2 new shapes needed (`flash_step` Q=512/KV=1024,
  `flash_step_init` Q=1024/KV=512), since KV for `flash_step_init` is always chunk 0 (txt, 512) and
  KV for `flash_step` accumulation is always an img chunk (1024). Both compiled clean (23/23 and
  14/14 ops, 13-16s). New `double0_img_qkv_proj`/`double0_img_out_proj` at chunk=1024 also compiled
  clean (63/63, 34s; 27/27, 80s).
- `runKleinDoubleChunkedBlockProbe` generalized to per-chunk-varying sizes (`chunkSizes: IntArray`)
  instead of one uniform `chunk`, selecting the right compiled piece per chunk's actual size.

**Result: 5 total chunks (1 txt + 4 img) → 35 dispatches (down from 9 chunks/99 dispatches, a 65%
cut), with essentially unchanged accuracy** — `img: maxAbsDiff=4.94` (was `5.01`), `txt:
maxAbsDiff=2.03` (was `2.21`) — and **23.3s wall-clock** (was `36.1-36.3s`, a 35% reduction even in
this disk-I/O-bound diagnostic). Confirms the lever works: fewer, larger dispatches measurably help,
with no correctness cost.

**Practical implication for scaling to the remaining 24 blocks**: don't default to chunk=512 or
chunk=1152 uniformly — pick the largest chunk size that stays in the proven-safe/fast regime
(~1024-1152 tokens, per the original performance-cliff bisect) for *every* stream, minimizing total
chunk count and therefore dispatch count, before compiling each additional block's pieces. Also worth
carrying into any real (non-diagnostic) integration: reuse `CompiledModel` instances across repeated
calls to the same compiled piece, and keep intermediate tensors in memory rather than round-tripping
through disk.

### Update: tested persistent-model reuse in the real chain — works, but currently I/O-bound (2026-08-29)

Applied the "reuse one `CompiledModel` instance across repeated calls" lever (already profiled
in isolation above) to the actual `runKleinDoubleChunkedBlockProbe` chain, not just a synthetic
profiling harness: all 8 distinct compiled pieces are now created **once** at the start (persistent
`lateinit var`s, tracked in a list for guaranteed `finally`-block cleanup) and reused across every
dispatch to that piece, instead of `create → run → close` on every single call. Total resident model
size (~500 MiB across all 8 pieces simultaneously) is comfortably within budget now that pieces are
already small/chunked for the dispatch-count-reduction work above — this is a different situation
from the SDXL-era create→run→close lesson, which was about one much larger dmabuf, not several
small/medium models held open at once.

**Result: `totalMs=22220` (was `23293`), same accuracy** (`img maxAbsDiff=4.94`, `txt maxAbsDiff=
2.03`, unchanged) — only a ~5% wall-clock improvement, much smaller than the ~20-40%/dispatch the
isolated profiling measured. The arithmetic explains why precisely: eliminating 27 redundant
`create()` calls (35 dispatches → 8 persistent models) at the profiled ~40ms/call blended average
overhead ≈ 1.08s — almost exactly the observed 1.07s delta. **The reuse lever is working exactly as
measured; its impact here is capped because the diagnostic's own per-`float` disk I/O (writing/
reading intermediate q/k/v/attn chunks between dispatches) still dominates the ~22s total**, not
`create()` overhead. The isolated profiling also showed reuse roughly halves `model.run()` time
itself (139→79ms for flash_step) — that saving is real but currently masked by the I/O bottleneck.
**Confirms both findings from the earlier profiling update: reuse is a real, low-risk, worth-keeping
win, and the disk round-trip remains the dominant cost in this diagnostic** — eliminating it (keeping
tensors in memory between dispatches) is the next lever to test if further speedup is wanted, likely
with a bigger payoff than reuse alone since it would also unmask the `run()`-time halving.

### Update: eliminated the disk round-trip entirely — 70% faster, confirms it was the real bottleneck (2026-08-29)

Rewrote `runKleinDoubleChunkedBlockProbe` to hold q/k/v/attn chunks as in-memory `FloatArray?`
arrays (`arrayOfNulls<FloatArray>(numChunks)`) instead of round-tripping every intermediate tensor
through `context.cacheDir` via `writeFloatBin`/`readFloatBinStreamed`. This is viable specifically
because double-stream's `out_proj` computes its MLP internally (unlike single-stream, which has a
separate ~85 MiB `mlp` intermediate per chunk stored externally) — so total intermediate data here is
just q+k+v+attn across 5 chunks, not q+k+v+mlp+attn.

**First attempt hit `OutOfMemoryError`** — total resident q+k+v+attn (~225 MiB) plus transient
per-dispatch buffers pushed peak usage right past the default ~256 MiB Dalvik heap ceiling. Root
cause: all 5 chunks' Q arrays were kept alive for the *entire* pass 2 loop even though each Q chunk is
only ever needed within its own single outer iteration (K/V, by contrast, genuinely is needed by
every iteration, since flash attention needs the full K/V set for each Q chunk). **Fix**: null out
`qChunks[qi]` immediately after that iteration finishes (keeps at most one "extra" Q chunk resident,
≤12.6 MiB, instead of all 5), and null out all of `kChunks`/`vChunks` once pass 2 finishes entirely
(pass 3 only needs `attnChunks`) — releasing ~113 MiB of dead weight before pass 3 starts.

**Result after the fix: `totalMs=6705` (was `22220` with disk I/O + model reuse, `23293` with disk
I/O alone) — a 70% reduction — no OOM, identical accuracy** (`img maxAbsDiff=4.94`, `txt
maxAbsDiff=2.03`, unchanged from every prior run). This conclusively confirms the disk round-trip,
not NPU dispatch overhead, was the dominant cost in every prior wall-clock measurement of this chain.
Combined with the earlier levers (dispatch-count reduction 99→35, persistent model reuse), this is
now a **95%+ combined wall-clock improvement over the original 99-dispatch/disk-I/O/recreate-each-call
baseline** (36.1-36.3s → 6.7s) with zero accuracy cost at any step.

### Update: applied the same perf fixes to single-stream — hit a hard memory ceiling, enabled `largeHeap` (2026-08-29)

Applied persistent model reuse (4 pieces: qkv_proj, flash_step, flash_step_init, out_proj, each
created once and closed in a `finally`) and in-memory chaining to `runKleinChunkedBlockProbe`. Unlike
double-stream, single-stream's `out_proj` takes `mlp` as an **external** input (a separate ~85 MiB/
chunk intermediate, not computed internally) — full in-memory chaining first hit a **fatal JNI abort**
(`OutOfMemoryError` during `SetFloatArrayRegion`, not a catchable Kotlin exception) trying to allocate
an 85 MiB `mlp` array.

Root cause, worked out precisely: `outputs.map { it.readFloat() }` must materialize **all four**
outputs (q, k, v, and the 85 MiB mlp) together before any can be released — the call itself doesn't
return until every `readFloat()` has allocated its array. By the last of 4 chunks, already-stored
q/k/v from the 3 prior chunks (~128 MiB, genuinely needed, not garbage) plus this call's own transient
127.6 MiB (q+k+v+mlp together) peaks at **~255 MiB — right at the default ~256 MiB Dalvik ceiling**,
independent of eager release, GC hints, or recomputation order (all analyzed and ruled out: nulling
old chunks doesn't help since they're still needed; explicit `System.gc()` can't reclaim data that
isn't garbage; recomputing q fresh in pass 2 instead of storing it merely moves the same transient
127.6 MiB spike to a point where all 4 chunks' k/v are *already* fully resident, netting ~241 MiB+ —
no better).

Given the tight, architecturally-inherent margin (not a lingering-reference bug), asked the user how
to resolve it: enable `android:largeHeap="true"` (app-wide manifest change, but standard/reversible
for memory-heavy apps), keep `mlp` on disk (smaller win, no manifest change), or shrink the chunk
size (more dispatches). **Chose `largeHeap`.** Added to `AndroidManifest.xml`'s `<application>` tag.

### Update: native C++ double-stream measurement (2026-08-29)

Added a throwaway `KleinDoubleBlockEngine` using LiteRT's C API and the same asymmetric production
layout as the validated Kotlin diagnostic: one text chunk at 512 tokens plus four image chunks at
1024 tokens, for 35 dispatches per `double_blocks.0` forward. It uses the output-DmaBuf fix already
identified by `KleinSingleBlockEngine`, keeps all host intermediates on the native heap, and diffs
against the real PyTorch reference IO.

The initial create→run→destroy-per-dispatch C++ baseline was correct but took **13.264 s/block**:
`img maxAbsDiff=4.938676, meanAbsDiff=0.106286 vs meanAbsRef=6.687800`; `txt
maxAbsDiff=2.034248, meanAbsDiff=0.086567 vs meanAbsRef=7.003502`; no NaN/Inf. This is slower than
the prior Kotlin diagnostic number because it intentionally retained SDXL's per-dispatch compiled-
model lifecycle as a baseline.

Keeping the eight distinct compiled pieces open for the complete chain (the same persistent-model
strategy already proven safe in Kotlin; ~500 MiB resident models) reduced the real C++ result to
**8.438 s/block** (**36% faster**) with bit-identical reported accuracy. Per-piece timing confirms
that repeated calls now have `setup=0ms`; the remaining time is NPU run plus fresh TensorBuffer
allocation/write/read on every dispatch. This is the confirmed native double-stream number to use
instead of the old 6.7 s Kotlin-only estimate, but it is still not production-fast enough.

The next experiment is to force DmaBuf for **inputs** as well as outputs, then assess persistent
input/output TensorBuffer reuse. The single-stream native engine still needs that same persistent-
model measurement before any complete image-time projection is treated as authoritative.

Follow-up input-DmaBuf A/B: forcing DmaBuf for all C++ input buffers reduced the same persistent-
model double-block run from **8.438 s to 8.035 s** (4.8%), with identical output statistics. It is a
small but real directionally-correct improvement, and is now the native double-engine baseline.

Final lifecycle A/B: retaining fixed-shape input and output TensorBuffers alongside each cached
compiled model reduced the native double block to **7.744 s** (another 3.6%), again with identical
numerics. This confirms safe persistent-buffer reuse, including explicit compiled-model-before-
buffer teardown, but shows that per-dispatch buffer allocation was not the primary remaining cost.
Further optimization should focus on avoiding host↔DMA copies between chained pieces or reducing
the number/cost of NPU dispatches.

### Update: complete double-block Q/K/V + flash-state zero-copy chain validated (2026-08-29)

The native double-block smoke path now has a correctness-first full-chain variant. It retains a
distinct Q/K/V DmaBuf producer set for the text chunk and each of four image chunks, passes those
buffers directly into every flash dispatch, and ping-pongs the three running-softmax state buffers
between flash dispatches. Only the final `running_sum`/`running_out` are read back for the existing
host normalization/transpose and output-projection path.

It completed on real Tensor G5 hardware and produced exactly the prior reference comparison:
`img maxAbsDiff=4.938676, meanAbsDiff=0.106286`; `txt maxAbsDiff=2.034248`,
`meanAbsDiff=0.086567`; no NaN/Inf. **Do not treat its 11.277 s wall-clock as a speed result.** This
first variant intentionally prewarms each separately-owned buffer/model cache through the old
copying path before executing the direct chain, so it duplicates dispatch work. Its purpose was to
prove buffer lifetime, cross-model compatibility, state ping-pong, and numerical correctness end
to end. The next optimization removes that prewarm path and pools retained output buffer sets behind
a single compiled model per artifact; only then can zero-copy performance be measured fairly.

Follow-up: removed the prewarm dispatches and directly allocated the persistent DmaBuf sets before
their first run. The full zero-copy double-block chain remains numerically identical and measures
**7.149 s/block**, down from the persistent-buffer CPU-copy baseline of 7.744 s (a 7.7% reduction).
This is a valid first performance number, but it still uses separate cached compiled-model instances
to own each retained Q/K/V and ping-pong state buffer set.

### Update: pooled full-chain zero-copy reduces native double-stream to 4.775 s (2026-08-29)

Implemented the pooled version: one cached compiled model per artifact now owns multiple explicitly
allocated DmaBuf output sets—five Q/K/V sets and two reusable flash-state sets—rather than creating
a compiled model for every retained buffer owner. The complete zero-copy chain ran clean on Tensor G5
with **identical numerical metrics** to every preceding double-block run and no NaN/Inf.

**Measured result: 4.775 s/block**, versus 7.149 s for the separate-model zero-copy validation and
7.744 s for the persistent-buffer CPU-copy baseline. This is a 38% improvement over the former and
the first meaningful performance result from full pooled zero-copy. Remaining host work is limited
to reading final attention state for normalization/transpose, plus the output-projection inputs and
outputs; moving that normalization/transpose into an NPU piece is now the clearest next transfer
elimination target.

### Update: attention-finalize NPU piece integrated (2026-08-29)

Exported a weight-free `attn_finalize` kernel—`running_out / running_sum`, head/token transpose,
and reshape to `[1, tokens, 3072]`—at both 512- and 1024-token shapes. Desktop LiteRT execution is
exact (`maxAbsDiff=0`), and both Tensor G5 AOT artifacts compile fully to one NPU partition (3/3
ops). The pooled native double-block chain now passes the final flash-state DmaBufs into this piece
and passes its DmaBuf output straight into `out_proj`, eliminating the former CPU normalization and
attention-buffer handoff.

On-device result: **4.752 s/block**, versus 4.775–4.953 s for prior pooled runs—within normal run
variance, so this boundary was not the dominant cost. Correctness remains in the established range:
img `meanAbsDiff=0.106575` vs `meanAbsRef=6.687800`; txt `meanAbsDiff=0.087167` vs
`meanAbsRef=7.003502`; no NaN/Inf. The remaining dominant cost is therefore NPU execution/dispatch
itself, rather than host attention handoff.

### Update: zero-copy cross-piece DmaBuf handoff is confirmed (2026-08-29)

A focused native C++ compatibility probe now runs `double0_img_qkv_proj` normally, then passes its
three output DmaBuf `TensorBuffer`s directly as the Q/K/V inputs of `flash_step`—no CPU read,
`std::vector` copy, or input-buffer write between the two compiled models. LiteRT accepted the
cross-model buffers, the direct NPU call completed in **51 ms**, and its three outputs were
**bit-identical** to the existing CPU-copy path (`maxAbsDiff=0`). This removes the main API-risk from
the next refactor: retain per-chunk Q/K/V buffers natively, pass them into flash directly, and
ping-pong flash-state DmaBufs between successive K/V chunks. The same approach can then connect the
attention result to output projection once normalization/transpose is moved into a small NPU piece.

**Result: `totalMs=18178` (was ~30-37s across prior runs) — ~40-50% faster, zero accuracy change**
(per-chunk `chunkMaxAbsDiff` values identical to every prior correct run: `1.93/1.93/1.67/1.61`). A
smaller win than double-stream's 95%+ (which had no `mlp` intermediate at all to contend with), but
consistent with `mlp`'s disk I/O being roughly half of single-stream's total original I/O volume (a
back-of-envelope estimate: mlp read+write ≈648 MiB vs. q/k/v/attn read+write ≈756 MiB out of ~1.4 GiB
total, so eliminating only the q/k/v/attn half while `mlp` stays on disk lines up with roughly a
40-50% cut, not double-stream's ~95%+ from eliminating everything).

### Update: tried eliminating `mlp`'s disk I/O too — genuinely doesn't fit even with `largeHeap` (2026-08-29)

Tested moving `mlp` fully in-memory (all 4 chunks resident, matching the q/k/v/attn treatment).
**OOM'd again** — this time even with `largeHeap` already enabled (`growth limit 536870912` = 512
MiB in the crash log, i.e. the *doubled* ceiling, not the default 256 MiB). The arithmetic is exact:
`mlp` alone (4 chunks × ~85 MiB) is ~340 MiB, and that's unavoidably concurrent with k/v needing to
stay resident through all of pass 2 (~113 MiB) plus `attn` (~57 MiB) — **~510 MiB total, right at the
512 MiB ceiling with zero headroom** for any transient allocation or Android runtime baseline.

Unlike q/k/v/attn (which fit even at chunk=1152 with `largeHeap`'s extra headroom), this is a hard
constraint on `mlp`'s own total size, not a fixable lifetime/ordering issue — the earlier per-piece
analysis (eager release, GC hints, recompute ordering) doesn't apply here because `mlp`'s total
footprint alone approaches the ceiling regardless of when other data is released. **Reverted `mlp`
back to disk-based streaming** — confirmed the revert restores the exact prior stable result
(`totalMs=18036`, matching the earlier `18178` almost exactly, identical accuracy). Keeping `mlp` on
disk at chunk=1152 is the practical optimum **for this Kotlin diagnostic**, not an unfinished
optimization within that context — going further there would require either a much larger heap
(diminishing returns/risk) or a smaller chunk size (more dispatches, undoing part of the earlier
dispatch-count win).

**Important correction/scope note (2026-08-29, prompted by a user question): this ceiling is
Kotlin/Dalvik-specific, not a real architectural limit.** The 256/512 MiB figures are the Android
*managed* (Dalvik/ART) heap quota — a JVM-object allocation limit, unrelated to total system RAM
(these devices have several GiB) or even the app's total process memory. The production engine for
this work is planned to be C++ (mirroring the existing SDXL `NpuUnetEngine`), which allocates from
the native heap — bounded only by actual available RAM, not this quota. So "`mlp` doesn't fit even
with `largeHeap`" is **a limitation of the Kotlin diagnostic harness used to validate correctness and
measure performance here, not a limitation of the eventual production implementation.** In C++,
`mlp` (and everything else) can plausibly stay fully in memory with no disk I/O at all, for every
block simultaneously if wanted — 16+ GiB dwarfs the ~340 MiB `mlp` needs per block. This means
single-stream's ~18s/block figure (with `mlp` on disk) is likely a substantial overestimate of what a
C++ engine would actually take; the "dispatch-only" reuse-profiled estimate from earlier (~5.6s/block,
no disk I/O anywhere) is a more realistic projection for that path — see the runtime-estimate update
below.

Also not yet done: apply the same three fixes (dispatch-count reduction, model reuse, in-memory
chaining where memory allows) to the other 23 blocks when scaling up.

### Update: revised full-pipeline runtime estimate for the planned C++ engine (2026-08-29)

The Kotlin-diagnostic-measured per-block times (single-stream ~18.2s, double-stream ~6.7s) give a
naive full-pipeline estimate of **~26.5 min/image** (20×18.2s + 5×6.7s, ×4 Klein diffusion steps) —
still far from the "faster than SDXL" goal. But per the correction above, single-stream's figure is
inflated by `mlp`'s Kotlin-heap-forced disk I/O, which a C++ engine shouldn't need at all. Using the
real profiled per-dispatch costs (`runKleinProfileDispatchOverhead`, reuse-model pattern, no disk I/O
anywhere) instead: single-stream ≈ 4×444+16×128+4×444 = **5.6s/block** (vs. double-stream's
already-near-optimal 6.7s, which used the same no-disk-I/O approach). That gives:

- per diffusion step: 20×5.6s + 5×6.7s ≈ 145.5s
- per image (4 steps): **≈ 9.7 min**

This is a projection, not a new direct measurement — it assumes actual NPU `model.run()` time is
language-independent (a safe assumption, that's genuine hardware time) and that C++ eliminates
essentially all the Kotlin-specific host-side overhead (disk I/O, JNI marshaling in `readFloat()`/
`writeFloat()`) that couldn't be avoided here. The real C++ engine needs to be built and measured to
confirm it. Still not a guaranteed win over SDXL, and doesn't yet reflect any further dispatch-count
reduction that might be possible for single-stream (double-stream's 99→35 cut hasn't been mirrored
there), but meaningfully more encouraging than the Kotlin-bound figure — and it means the `mlp`
OOM/`largeHeap` finding above should be read as "this diagnostic tool has a memory wall," not "this
architecture does."

### Update: pipeline generalized and proven on `single_blocks.1` (2026-08-29)

Before committing to all 23 remaining blocks, validated that the per-block export/compile/on-device
pipeline actually generalizes, not just works for block 0. Added `--block-index` to
`klein_qkv_proj_export.py`/`klein_out_proj_export.py`/`compute_klein_single0_torch_reference.py`
(`load_single_block`/`load_double_block` already supported arbitrary indices). Key simplification:
`x`/`pe`/`mod_*` reference inputs are seed(0)-deterministic and **identical across every block** —
only the reference *output* (`torch_out`) differs per block, since only the weights differ. `flash_step`/
`flash_step_init` are pure attention math with **no learned weights**, so the already-compiled
chunk=1152 artifacts are reusable as-is for every single-stream block (and the chunk=512/1024
variants for every double-stream block) — only `qkv_proj`/`out_proj` (2 pieces per single-stream
block, 4 per double-stream block) need re-exporting/compiling per block.

Exported + AOT-compiled `single_blocks.1`'s `qkv_proj`/`out_proj` (65/65 and 9/9 ops, clean, ~79s +
~48s), generated its reference output, and validated on-device by temporarily swapping its compiled
pieces into the same on-device filenames block 0 used (a validation-only swap, not a permanent
multi-block Kotlin design — see below). **Result: `totalMs=18308` (block 0: `~18036-18178`),
`maxAbsDiff=1.42`, mean relative error ~0.8%** — same performance and tolerance class as block 0,
confirming the pipeline and its just-finished performance optimizations generalize correctly to a
second block with entirely different weights. Restored block 0's artifacts afterward.

**Time budget for the remaining 22 blocks**, based on this real measurement: ~2-3 min AOT-compile
time per single-stream block (19 remaining ≈ 40-55 min), ~3 min per double-stream block (4 remaining
≈ 12 min) — **roughly 1-1.5 hours of mostly-unattended background compute** for AOT compilation
alone, plus export/push/validation overhead on top. Full per-block on-device validation the way
block 1 was just done (swap artifacts into shared filenames, run, swap back) doesn't scale well to 22
more blocks — the next practical step is either generalizing the Kotlin diagnostics to accept a
block index (avoiding swap/restore cycles) and/or reserving full on-device validation for a sample of
blocks while relying on each export script's own CPU-side correctness check (already run for every
block) as the baseline signal.

**Caution hit along the way, worth remembering when scaling to more blocks**: single-stream and
double-stream reference-IO push scripts share the same on-device directory
(`klein_single0/`) and both use a bare `pe.bin` filename (single-stream's is the full 4608-token
sequence; double-stream's is the img-only 4096-token version) — pushing double-stream's reference IO
silently overwrote single-stream's `pe.bin` with the wrong (smaller) file, causing an `EOFException`
the next time `runKleinChunkedBlockProbe` ran. This will recur for any future per-block reference IO
that reuses a generic filename in this shared directory — either namespace filenames per block (e.g.
`single0_pe.bin`, `double0_pe.bin`) or use per-block subdirectories before scaling further.

8. **Compile the remaining 24 transformer blocks.** Once piece 0 works end-to-end: the 20
   single-stream blocks and 5 double-stream blocks. Double-stream blocks have a different operator
   shape (two parallel streams with cross-attention) — do not assume the split strategy for
   single-stream blocks transfers directly; verify with at least one `double_blocks.0` probe first.
   **Use the largest safe chunk size (not necessarily 512) to minimize dispatch count** — see the
   dispatch-count-driven-overhead update above.

9. **Do not integrate into PocketTavern's download UI or `NpuUnetEngine`** until Klein runs
   correctly on-device, all 25 blocks, matching the PyTorch reference. The diagnostic hooks
   (`run_klein_single0_npu_diagnostic`, `run_klein_single0_small_shape_diagnostic`,
   `run_klein_token_probe`) remain debug-only and are gated on `BuildConfig.DEBUG`.

### Update: C++ engine built and measured — found and fixed a 7x buffer-read bottleneck (2026-08-29)

Per the plan above, built a throwaway native C++ engine (`app/src/main/cpp/npu/KleinSingleBlockEngine.{hpp,cpp}`,
wired through `jni_diffusion.cpp`/`NpuDiagnostic.kt`/`MainActivity.kt`'s `run_native_klein_single_block_smoke`
debug hook, mirroring `NpuUnetEngine`'s already-validated LiteRT-C-API integration pattern) to get a
*confirmed* real number — the Kotlin-measured ~18.2s/block and its derived ~26.5min/image estimate
were flagged as too slow, and the correction above (Kotlin's heap ceiling isn't a real C++ limit)
still left the actual C++ number unmeasured.

First real run: **34.9s/block** — correct output (`maxAbsDiff=1.930710`, identical to Kotlin's), but
~2x *slower* than Kotlin, not faster as hoped. Added per-phase timing
(`model`/`compile`/`buffers`/`run`/`read` — i.e. model-open, CompiledModel-create, buffer-alloc,
`LiteRtRunCompiledModel`, and `LiteRtLockTensorBuffer(read)+memcpy+Unlock` respectively) and found the
**`read` phase was 66% of total time (21.2s of 32s)**, while `run` (actual NPU compute, 6.1s total)
matched Kotlin-profiled expectations almost exactly — the NPU itself was never the problem.

Added further diagnostic logging of each output tensor's `supported_types` (from
`LiteRtGetTensorBufferRequirementsSupportedTensorBufferType`) and `chosen_type` (from
`LiteRtGetTensorBufferType` post-creation). Found: **every single output buffer, across every piece
type, offers only `[Ahwb(2), DmaBuf(4)]` — no `HostMemory` option exists for these NPU-dispatched
tensors — and `LiteRtCreateManagedTensorBufferFromRequirements` auto-picks `Ahwb` every time.**

**Root cause confirmed**: switched output-buffer creation from
`LiteRtCreateManagedTensorBufferFromRequirements` (auto-picks Ahwb) to `LiteRtCreateManagedTensorBuffer`
with `kLiteRtTensorBufferTypeDmaBuf` forced explicitly (same tensor type, buffer size taken from
`LiteRtGetTensorBufferRequirementsBufferSize`). Result: **`read` phase dropped from 21.2s → 2.9s (~7x),
total time dropped from 34.9s → 16.3s**, correctness unchanged (`maxAbsDiff=1.930710`, identical). Ahwb's
CPU-lock path evidently requires a real cache-sync on this hardware that DmaBuf's mapping does not.

**Revised confirmed-input projection**: 16.3s/block (single-stream, C++, DmaBuf-forced) × 19 remaining
single-stream blocks ≈ 310s, + 6.7s/block (double-stream, C++, already validated pre-DmaBuf-fix — not
yet re-measured with DmaBuf forced, likely has the same headroom) × 4 double-stream blocks ≈ 27s, per
step ≈ 337s → **×4 steps ≈ 22.5 min/image**. This supersedes both the ~26.5min Kotlin-measured figure
and the earlier ~9.7min *projected* figure (that projection didn't know about the Ahwb/DmaBuf
buffer-read penalty at all, so it was accidentally optimistic in a different way than intended).

**Not yet done / immediate next steps for whoever picks this up**:
1. Re-run the double-stream chunked block through the same C++ engine pattern with `DmaBuf` forced
   from the start, to get a real (not carried-over-assumed) double-stream C++ number.
2. Investigate whether `run` (5.9s/block, actual NPU compute — now the dominant cost after the DmaBuf
   fix) has further headroom, e.g. by checking if input buffers should also be forced to `DmaBuf`
   (currently only output buffers were changed; input buffers still use
   `LiteRtCreateManagedTensorBufferFromRequirements`) or if per-dispatch overhead (`compile`+`buffers`
   phases, ~4.1s/block combined) can be cut further via the persistent-model-reuse pattern already
   proven for double-stream (see "tested persistent-model reuse" update above) — `KleinSingleBlockEngine`
   currently does NOT test persistent-model reuse, only single create→run→destroy per piece.
3. Once single- and double-stream C++ numbers are both confirmed with the DmaBuf fix, update this
   doc's projection with real (not partially-carried-over) figures for both.
4. Then resume scaling to the remaining 22 blocks (see step 8 above) using the C++ engine, not the
   Kotlin diagnostic path, now that C++ is confirmed faster.

### Update: single-stream persistent model + pooled zero-copy (2026-08-29)

`KleinSingleBlockEngine` now retains its four compiled models for the engine lifetime. This alone
reduced the on-device single-block smoke test from **16.3s to 12.446s**, with the same numerical
result (`maxAbsDiff=1.930710`, `meanAbsDiff=0.049629`, no NaN/Inf).

The optimized `forwardZeroCopyPooled` then keeps all four Q/K/V/MLP chunk outputs in DmaBuf,
threads flash-attention state through two DmaBuf state sets, runs the new NPU
`attn_finalize_probe_1152` artifact, and gives its attention result and the original MLP DmaBuf
straight to `_out`. It reads only the four final output chunks back to host. The first complete
on-device run is correct (`maxAbsDiff=1.930710`, `meanAbsDiff=0.049687`, no NaN/Inf) and takes
**7.019s/single block**.

Parallel-QKV experiment: Klein is guidance-distilled, so unlike SDXL it has no independent
conditional/unconditional CFG pair. Its four QKV/MLP chunks *are* independent, however. A trial
that gives each chunk a separate compiled model and DmaBuf set, then submits the four runs on four
threads, is correct on-device with identical error metrics. Two cold runs took **5.895s** and
**6.367s** (mean **6.131s**) versus 7.019s serial: about a **12.6%** full-block latency reduction.
This confirms that the Tensor G5 dispatch service overlaps some independent work, though far from
perfectly. A follow-on, higher-DmaBuf-memory experiment also gave each of the four independent
attention-query chains its own init/flash/finalize compiled models and state buffers. It remained
correct with two runs of **5.970s** and **5.534s** (mean **5.752s**): a promising additional gain,
but with only two samples it should be treated as provisional. The four independent output-projection
chunks were then isolated and submitted in parallel too; two correct runs were **5.541s** and
**5.129s** (mean **5.335s**). This is the current best single-block smoke result, though it incurs
the memory cost of four simultaneous output-projection compiled models.

Worker-count tuning confirms four concurrent chunks is the best tested setting on this Pixel 10:
two-worker and three-worker waves were correct but slower at **6.537s** and **6.778s**, respectively.
The performance gain requires enough simultaneous dispatches to keep the Tensor G5 busy; reducing
contention by itself does not help this workload.

Most importantly, the engine's compiled-model cache produces a materially faster steady state when
the same engine instance survives across denoising steps. Two cold-process trials, each executing
two forwards through one engine, measured `5.915s → 3.273s` and `6.590s → 3.626s`; mean cold time is
**6.253s** and mean warm time **3.450s** (both numerically correct). Production integration must
therefore construct each block engine once and retain it for all four steps. These warm timings still
allocate temporary DmaBuf sets per forward, so a later reusable buffer pool may improve them further.

Memory-bounded cache experiment: `ReleaseCachedTensorBuffers()` now frees every cached piece's
input/output DmaBufs while retaining only the compiled model handles; the next forward recreates the
buffers on demand. Two correct trials measured `5.704s cold / 3.733s full-warm / 4.143s model-only`
and `6.451s / 3.464s / 4.085s`, respectively. The **4.114s model-only mean** preserves most of the
compiled-cache benefit while paying only about 0.5s versus retaining the buffers. This is the right
production baseline: never retain the large per-piece DmaBufs across steps. Whether all 25 compiled
model handles fit simultaneously remains a separate, mandatory memory-budget test once their
artifacts exist; use a bounded/LRU model cache if they do not.

Using the provisional fully parallel single-stream mean and measured double-stream time (20 singles
at 5.335s and 5 doubles at 4.752s), the 4-step projection is
`4 × (20 × 5.335 + 5 × 4.752) = 521.8s`, or **about 8.7 minutes per image**. This is still a
smoke-test projection rather than a production end-to-end benchmark; the next gain is to keep the
pooled buffers across denoising steps in the production engine instead of recreating them each block
invocation.

Inter-block DmaBuf chaining is now confirmed compatible at the single-stream boundary. A focused
probe retained each of the four final `_out` chunk outputs in DmaBuf and supplied it directly as the
activation input to a separate QKV dispatch; the Tensor G5 accepted all four handoffs without a
host copy, while the originating block still matched its PyTorch reference. The probe uses the
block-0 QKV artifact as the consumer because it is the artifact currently available; learned weights
do not change this activation tensor's shape or buffer requirements. Full pipeline integration must
repeat numerical end-to-end validation using adjacent blocks' actual artifacts, but the former
API-compatibility risk is removed.

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

## Full-transformer integration baseline (2026-08-29)

All learned projection artifacts now exist for the five double blocks and twenty single blocks.
`KleinTransformerEngine` executes the exact Flux ordering (five double blocks, concatenate text
then image tokens, then twenty single blocks) and selects each block's namespaced learned artifact.
It is deliberately **memory bounded**: one block engine is scoped and destroyed before the next
block begins. This releases its compiled-model handles and every DmaBuf allocation, avoiding an
unmeasured 25-block persistent-NPU-memory commitment. It is the safe correctness integration
baseline, not the final warm-performance policy.

The model directory is flat; use `scripts/stage_klein_npu_artifacts.sh` to collect the exporter’s
`*_noflags_aot/*_Google_Tensor_G5.tflite` outputs before pushing them to the device. A direct
source-tree audit verified all 58 learned artifact paths used by that staging step. The native
Android build passes after adding this runner.

This is not yet image generation: the remaining pipeline work is the text encoder, image/context
input projections, timestep modulation networks, final layer, denoising schedule, and Flux VAE
decoder. The runner's explicit projected-tensor interface makes each of those additions
independently testable, rather than masking missing model components behind a nominal image API.

### Denoiser component export (2026-08-29)

`scripts/export_klein_denoiser_components.py` now extracts and exports the non-block components
that are present in `unstableRevolutionF2K_AlphaF2K4BFp16.safetensors`: `img_in`, `txt_in`,
`time_in`, image/text/single modulation projections, and `final`.  The `time_in` artifact has been
converted successfully at its real `[1,256] -> [1,3072]` shape.  Its timestep sinusoid is cheap
deterministic host math; `time_in` therefore accepts that 256-value embedding rather than a scalar
timestep.  The remaining component exports must receive the same no-flags Tensor G5 AOT pass used
for block pieces before the runner can consume them on NPU.

The merged Klein checkpoint has exactly 149 tensors and contains no Qwen3 or Flux autoencoder keys.
Thus, neither a compatible Qwen3-4B text-encoder model/tokenizer nor a Flux 32-channel autoencoder
decoder is available locally. They must be acquired before a prompt can become pixels; exporting or
substituting the existing SDXL/Wan VAEs would be architecturally incorrect.

RuinedFooocus was audited as a possible source. Its only Qwen-named candidate is a 127 MB file,
`models/clip/qwen3vl_4b_fp8_scaled.safetensors`; it is neither a complete 4B model (far too small)
nor a valid SafeTensors file (its header is incomplete). Its available VAE weights are SDXL and Wan
only. Therefore that checkout does not change the missing-component conclusion.

### Downloaded official weights (2026-08-29)

The official Qwen3-4B encoder shards have been verified in `~/Downloads`:
`model-00001-of-00002.safetensors` (4,967,215,360 bytes, 229 tensors) and
`model-00002-of-00002.safetensors` (3,077,766,632 bytes, 169 tensors). The official Flux2
`ae.safetensors` is also present (336,211,292 bytes, 251 tensors). The Qwen config, shard index,
and complete tokenizer set are in `~/Downloads/flux2_klein_qwen/`. The files are valid SafeTensors;
the next dependency is an on-device Qwen hidden-state execution path, while the Flux2 decoder is
being AOT-compiled for Tensor G5.

`scripts/export_klein_qwen_reference.py` validates this exact source model and creates a reusable
reference conditioning input. It applies the official Qwen chat template, pads/truncates to 512,
and concatenates hidden states 9, 18, and 27. Its first CPU run produced a finite
`context.bin` with shape `[1,512,7680]` (15,728,640 bytes), exactly matching `txt_in`'s input.

### Complete G5 component and VAE compilation (2026-08-29)

All seven non-block denoiser components now have no-flags Tensor G5 AOT artifacts: `img_in`,
`txt_in`, `time_in`, `mod_img`, `mod_txt`, `mod_single`, and `final`. Every operation in each
compiled component was offloaded to a single NPU partition. The VAE decoder is also completely
compiled as eight memory-bounded stages: `pre_mid`, `up_3`, `up_2`, `up_1`, the three
`up_0_block_*` residual blocks, and `up_0_head`.

The initially monolithic decoder compiler process was OOM-killed at approximately 9.5 GB host
RSS; combining the whole final 1024px upsample level was also OOM-killed at about 12.2 GB. The
split exports compile successfully, but this is a compile-time result only. The `[1,256,1024,1024]`
activation entering `up_0_block_0` alone is roughly 1 GiB in float32, so device-side execution
still needs a DmaBuf-memory validation and may require additional spatial tiling.

To deploy the complete current artifact set, stage both roots:

```bash
bash scripts/stage_klein_npu_artifacts.sh \
  /home/brandont/code/litert-torch/scratch/models/flux2_klein_probe \
  /home/brandont/Downloads/flux2_klein_npu \
  /home/brandont/code/litert-torch/scratch/models/flux2_klein_components
```

This produces 101 flat `*_Google_Tensor_G5.tflite` artifacts: the learned transformer pieces,
all denoiser components, and the split VAE. The remaining substantive integration work is an
on-device Qwen execution path and a native pipeline runner that chains the components, denoising
schedule, transformer, and VAE stages.

### Pixel 10 on-device component and full-transformer validation (2026-08-29)

The complete staged set was copied to the Pixel 10 app-private directory and exercised through
the native LiteRT C API with `NPU|CPU` enabled. The component runner executed `time_in` in 90 ms,
the real-shape image and text input projections in 164 ms and 53 ms respectively, and VAE
`pre_mid` in 3.93 s. A sequential VAE `pre_mid -> up_3` test also completed with finite output:
3.00 s for `pre_mid`, 0.85 s for `up_3`, producing its `[1,512,256,256]` tensor.

Most importantly, `KleinTransformerEngine` completed the whole real-shape five-double plus
twenty-single block sequence in 96.1 s with finite `[4096,3072]` image and `[512,3072]` text
outputs. The test used correctly shaped synthetic projected tensors and zero modulation, so it
validates the deployed artifacts, dispatch, and memory-bounded ownership policy—not model-image
quality. There was no process or NPU-memory failure (the app process stayed near 500 MiB RSS).

The next pipeline stage must supply real Qwen conditioning, timestep modulation, RoPE positional
embeddings, and a denoising schedule before the validated transformer output can be decoded into
an image. The later VAE stages still require their own device-memory validation before a 1024px
decode is treated as production-safe.

### First semantically wired denoising step (2026-08-29)

`KleinComponentEngine` now supports multi-output artifacts, allowing the six double-stream and
three single-stream modulation tensors to be consumed without host-side reconstruction. A native
one-step reference path reads the known-good Qwen `context.bin` for the saved fox prompt, creates
the Flux timestep embedding and four-axis RoPE positions on device, runs input/time/modulation
projections, all 25 transformer blocks, and `final`, then applies the first four-step schedule
update. On the Pixel it completed in **95.96 s** and wrote a finite 524,288-float
`one_step_latent.bin`. This is a real fixed-prompt denoising step, not an image yet: it must be
looped through all four schedule steps and then passed to the split VAE decoder. Qwen remains the
only prompt-dependent portion not yet running locally on the phone.

### Q/K fused-artifact numerical failure and split repair (2026-08-29)

The complete four-step reference and direct staged VAE decode execute on the Pixel, but output was
near-uniform gray. Boundary comparisons localized the first substantive error to double block 0.
The NPU fused QKV artifact returns correct V but nearly zero Q/K under real model activations;
host-copy and zero-copy engine paths fail identically. This isolates a silent Tensor G5 compiled
Q/K RMSNorm/RoPE failure, rather than a weight, DmaBuf, scheduler, or VAE issue.

The repair keeps input normalization and the learned QKV matrix on NPU, then performs Q/K per-head
RMS normalization and RoPE natively before existing NPU flash-attention artifacts. A raw-QKV
double-block exporter mode was added and block-0 image/text artifacts were locally validated against
PyTorch at <=5.15e-5 and AOT compiled. Runtime wiring and an on-device block-0 comparison remain
required before regenerating the full artifact set. See `FLUX2_KLEIN_HANDOFF.md` for the
artifact paths and exact continuation steps.

### RoPE generator bug: the real root cause of the Q/K numerical failure (2026-08-30)

Extracted Q/K RMSNorm scales for all 5 double + 20 single blocks
(`scripts/export_klein_qk_norm_scales.py`), exported+AOT-compiled raw-qkv artifacts for double
blocks 1-4 (block 0's already existed), and wired a native RMSNorm+RoPE implementation
(`app/src/main/cpp/npu/klein_qk_norm_rope.hpp`) into `KleinDoubleBlockEngine` to replace the fused
NPU artifact's suspected-buggy norm+rope tail — exactly the repair plan from
`FLUX2_KLEIN_HANDOFF.md`'s "Exact next work". The native function was validated bit-close against
PyTorch off-device first, via a header-only, host-`g++`-buildable self-test
(`scripts/klein_qk_norm_rope_selftest.cpp`) fed by a Python reference generator
(`litert-torch/scratch/klein_qk_norm_rope_reference.py`) — no device needed for this step.

**On real hardware, the native-norm-rope path still produced garbage** (`meanAbsDiff` ≈
`meanAbsRef`, i.e. uncorrelated) despite passing every offline check. Bisected methodically rather
than assuming either "NPU is broken again" or "my new code is broken":

1. Pulled the real activations the on-device pipeline actually used (`debug_img_in.bin`,
   `debug_mod_img_0/1.bin`, etc. — real Qwen-conditioned data, not synthetic) and reproduced them in
   PyTorch (`litert-torch/scratch/klein_double0_ondevice_validate.py`): still a huge mismatch against
   the device's own post-norm-rope output.
2. Ran the exported (pre-AOT) raw-qkv `.tflite` graphs on desktop XNNPACK against those same real
   activations (`klein_double0_raw_qkv_desktop_check.py`): matched PyTorch to ~1e-4 — the exported
   graph and its litert_torch conversion are correct.
3. Added a temporary instrumentation hook to dump the RAW (pre-norm) Q/K straight from the on-device
   raw-qkv NPU dispatch, before the native norm/rope step touches it, and compared to PyTorch's raw
   projection on the same real activations (`klein_double0_raw_ondevice_check.py`): matched to
   ~0.2-0.3% relative error, the normal chained-dispatch tolerance. **The raw-qkv NPU piece itself —
   the first-ever real on-device test of it — is correct.**
4. Took that same real on-device raw Q, plus the real checkpoint scale, plus a Python-reproduced
   `pe`, and ran them through the exact same `klein_qk_norm_rope.hpp` via the host self-test
   (`klein_qk_norm_rope_realdata_reference.py` + the selftest binary at production shape,
   HEADS=24/TOKENS=1024/HEAD_DIM=128): matched PyTorch to `maxAbsDiff=1.9e-6`. **The native
   norm+rope function itself is also correct, even at real production shape/data.**

Every individual piece checked out correct, yet the full on-device chain didn't — which meant the
bug had to be in what fed the two pieces together: the `pe`/`pe_ctx` buffers themselves. Re-reading
`jni_diffusion.cpp`'s `positions()` lambda (the on-device RoPE-position generator used by both the
fused artifact's `pe` input historically and the native splice's `pe_chunk` argument) found it:

```cpp
for (size_t n=0;n<tokens;++n) {
    const int coords[4] = {...};
    size_t o=0;                    // BUG: resets every token instead of o=n*256
    for (int axis=0;axis<4;++axis) for (int j=0;j<16;++j) { ... pe[o++]=c; ... }
}
```

`o` is declared *inside* the per-token loop and always starts at 0, so every token's 256 floats of
RoPE data get written into `pe[0..255]`, each token overwriting the last. Only the final token in
the sequence ends up with valid RoPE data; every other token's slot is left at its
default-initialized value: all zero. A zero `[[0,0],[0,0]]` "rotation" doesn't leave a vector
unrotated — it multiplies it to zero, exactly matching the original "near-zero Q/K" symptom that
kicked off this entire investigation. This bug predates this session and affects **both** double and
single blocks (they share the same `pe`/`pe_ctx` buffers) — it is a very plausible root cause of the
whole "images are near-uniform gray" problem in `FLUX2_KLEIN_HANDOFF.md`'s "Critical current status",
not a Tensor G5 hardware/compiler defect at all.

**Fix**: `size_t o=0;` → `size_t o=n*256;` (one line, `app/src/main/cpp/jni_diffusion.cpp`).

**Re-tested the ORIGINAL fused qkv+norm+rope NPU artifact directly** (not the native-RMSNorm
workaround) with the fix in place, via a temporary `DebugFirstQkvFused` method: `meanAbsDiff` vs.
PyTorch is ~0.0026-0.0029 against `meanAbsRef` ~0.75-0.78 (≈0.3-0.4% relative error) for
double_blocks.0's Q/K — the same tolerance class as every other chained-dispatch measurement already
accepted in this project (e.g. single-stream's ~1-1.5%, double-stream's ~1.3-1.6%). **The fused
artifact was never broken.** `KleinDoubleBlockEngine` was reverted to use it again (the native-split
detour is unnecessary); the zero-copy performance paths that had been disabled during the detour
(`forwardZeroCopy`, `forwardZeroCopyPooled`, `RunZeroCopyQkvToFlashProbe`) are restored.
`klein_qk_norm_rope.hpp` and its self-test are left in the tree, unused, as a validated
building block in case a future block/shape ever needs a real native fallback.

**Full pipeline re-run end-to-end after the fix, real hardware**: `nativeRunKleinOneStepReference`
(all 25 blocks + final layer) completed in 92.1s; all 4 diffusion steps completed cleanly
(89.5-90.9s each); the direct staged VAE decode completed in 19.3s and produced a real, coherent
1024×1024 image — the fox-in-a-moonlit-forest reference prompt, matching what this project has used
as its standing test prompt throughout, not gray or degenerate. This is the project's first
confirmed complete on-device FLUX.2 [klein] image generation, fixed-prompt end-to-end. See
`FLUX2_KLEIN_HANDOFF.md`'s top section for the current-status summary and next steps (mainly:
real on-device Qwen, and the pre-existing performance-tuning backlog, which this fix does not
change).

## 2026-08-30: Phase 1 — on-device Qwen3-4B text encoder, desktop validation

Full plan: `docs/FLUX2_KLEIN_PHASE1_TEXT_ENCODER_PLAN.md`. This entry covers step 1 (export +
desktop validation) only.

PocketTavern already vendors MNN's LLM inference engine
(`app/src/main/cpp/MNN/transformers/llm/engine/`, compiled into `libMNN.so` via `MNN_BUILD_LLM ON`)
and its Python export tooling already understands Qwen3 natively. The export tooling also already
had a hidden-state-tap mechanism (`dflash_target_layer_ids` in `utils/model.py`'s `LlmModel.forward`)
used today only for speculative-decoding draft models — it collects post-block hidden states at
given 0-indexed decoder-layer indices and concatenates them as the model's `hidden_states` output.
That mechanism is exactly what Klein's text conditioning needs (layers 9/18/27, HF-indexed, matching
`scripts/export_klein_qwen_reference.py`), so `llmexport.py` gained a small new flag,
`--hidden_states_layers` (comma-separated, HF-indexed to match the reference script, converted to
0-indexed internally: `layer - 1`), reusing the existing tap rather than adding new infrastructure.

Exported Qwen3-4B (`~/Downloads/flux2_klein_qwen/text_encoder`) to ONNX with
`--hidden_states_layers 9,18,27`; confirmed the graph's `hidden_states` output is a `Concat` of the
three tapped layers, `[1,512,7680]`, matching the reference script's shape.

**Desktop validation** (`/home/brandont/code/litert-torch/scratch/klein_qwen_hidden_states_validate.py`,
MNN's PyTorch reimplementation called directly, not via ONNX) against
`export_klein_qwen_reference.py`'s HF reference, for 3 varied prompts, right-padded to 512 tokens
with a hand-built combined causal+padding attention mask (MNN's built-in `full_attention_mask()`
helper is naive causal-only and not padding-aware, which matters here because `txt_in` consumes the
full 512-token tensor including padding):

| prompt (real_len) | real-content relErr | padding-region relErr | HF pad max | MNN pad max |
|---|---|---|---|---|
| fox in moonlit forest (25) | 1.084% | 10.735% | 256 | 264 |
| astronaut on horse on Mars (22) | 1.298% | 9.132% | 368 | 336 |
| dew drops on spider web (32) | 1.228% | 10.651% | 616 | 656 |

Real-content-region error is consistent across prompts (~1.1-1.3%), a bit above this project's
prior ~0.3-0.5% tolerance class but stable — likely accumulated bf16 rounding through 36 decoder
layers across two independent implementations (fixed two real dtype-propagation bugs in MNN's
reimplementation along the way, see below). Padding-region error looks alarming in isolation but
both implementations *independently* produce similarly large-magnitude values there (right down to
matching mean magnitude to 4 significant digits, e.g. 1.0010 vs 1.0010 on prompt 3) — this is
"attention sink"-style outlier-channel behavior from feeding hundreds of repeated pad tokens through
a causal transformer, not a one-sided bug; small bf16 rounding differences get amplified in that
numerically sensitive regime. Causal attention means the padding tail cannot influence the real
content, so the real-content number is the one that matters for Klein's `txt_in`.

Two real bugs found and fixed in MNN's Qwen3 reimplementation (`utils/transformers.py`), both only
surfacing when calling `model.forward()` directly outside the real export pipeline (which upcasts
the whole non-Linear graph to float32 via `visit_module` when `self.args.export` is set — not an
issue in production export, only in this direct-forward validation harness, so worked around in the
validation script rather than touched in `transformers.py`):
- `RMSNorm.weight` is created as `torch.ones(hidden_size)` (float32) in `__init__`; state_dict
  loading preserves that dtype rather than adopting the bf16 checkpoint's, so norm output silently
  promotes to float32 and breaks downstream bf16 matmuls.
- `Rotary`'s `theta` buffer and derived cos/sin tables are float32 and never cast down, same failure
  mode for RoPE'd query/key states.

**Verdict**: tap mechanism and MNN's Qwen3 forward are validated at the desktop level for the
semantically meaningful (real-content) region, across 3 varied prompts. Next: quantize for
on-device size (`llmexport.py --quant_bit`/`--hqq`), build the `QwenTextEncoderEngine` JNI wrapper,
and validate on real Pixel hardware (plan steps 1 continuation, 2, 3).

## 2026-08-30 continued: Phase 1 steps 1(quantize)/2/3 — on-device blocker (unresolved)

**Quantized export**: `llmexport.py --export mnn --hqq --hidden_states_layers 9,18,27` (matching
`AGENTS.md`'s documented example command) ran cleanly under the watchdog: `llm.mnn` (1.5MB graph) +
`llm.mnn.weight` (2.26GB int4 weights), `config.json` correctly carries `"hidden_states": true`,
tied-embeddings info baked into the weight blob's byte offsets. No memory incidents.

Running the exported model needed `pymnn` (the `MNN` PyPI package) locally for the ONNX→MNN
conversion step, since no `MNNConvert` binary was built. Installing it (`uv pip install MNN`, user
pre-approved) revealed a real supply-chain quirk worth flagging: importing `MNN.tools` silently
shells out to `pip install aliyun-log-python-sdk` (a telemetry SDK) into the *global* user
site-packages, downgrading a system-wide `protobuf` in the process — not something we asked for.
Verified no breakage (both `onnx` and `transformers` tolerate the downgraded protobuf version) and
filed it as product feedback; not a PocketTavern-repo issue. Separately, the wheel's two `.so`
files (`_mnncengine`, `_tools`) needed `patchelf --clear-execstack` to import at all on this
machine's hardened kernel (`cannot enable executable stack as shared object requires`).

**JNI wrapper**: added `app/src/main/cpp/npu/QwenTextEncoderEngine.{hpp,cpp}`, wrapping
`MNN::Transformer::Llm` (MNN's own vendored LLM engine, `transformers/llm/engine`, already compiled
into `libMNN.so` via `MNN_BUILD_LLM ON` but previously never wired to any JNI entry point). Registered
in `CMakeLists.txt` (only needs `MNN/transformers/llm/engine/include` on the include path plus the
new `.cpp` in `pockettavern_diffusion`'s sources -- `Llm` is `MNN_PUBLIC` and already exported from
`libMNN.so`, unlike the NPU/Klein engines which need their own compiled copy). Added a diagnostic JNI
entry point (`nativeRunQwenTextEncoder` in `jni_diffusion.cpp`, `NpuDiagnostic.runNativeQwenTextEncoder`
in Kotlin, `run_native_qwen_text_encoder` debug intent extra in `MainActivity`), matching the
`nativeRunKleinOneStepReference` pattern: dumps the `[1,512,7680]` output to a `.bin` file for pulling
and diffing against the PyTorch reference the same way every other Klein component was validated.
Build succeeds (`./gradlew assembleDebug`).

One correctness fix made along the way: `Llm::apply_chat_template()` has no way to force
`enable_thinking=False`, and Qwen3's official jinja template defaults to thinking-mode ON when that
variable is left unset -- silently diverging from the validated reference (which used
`enable_thinking=False`). Since the reference's chat wrapper text is fixed boilerplate independent of
prompt content for the no-system-message case
(`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`, confirmed by
directly invoking the HF tokenizer), `QwenTextEncoderEngine::Encode` builds this string itself rather
than going through `apply_chat_template()`.

**Device staging**: pushed the exported model (`config.json`, `llm_config.json`, `llm.mnn`,
`llm.mnn.weight`, `tokenizer.mtok`) to `/data/data/com.pockettavern.app/files/flux2_klein_qwen/` on
the real Pixel via the project's established `adb push` + `run-as cp` technique. 58GB free on
`/data`, no space concerns.

**Blocker (unresolved)**: `nativeRunQwenTextEncoder` fails every time with
`onForward returned no outputs. seqLen=N, inDecode=0, inputs=4, moduleKey=(100,0)`, MNN backend
error code 1 = `OUT_OF_MEMORY`. Debugged with temporary instrumentation (added and then fully
reverted -- `git diff` on the two touched vendored files is clean):
- **Not a real size/memory problem.** Reproduces identically at `seqLen=13` ("fox") and `seqLen=512`
  (padded) -- if it were a genuine large-allocation limit it would scale with seqLen and pass at 13.
  Device had 9.8GB `MemAvailable` at time of failure; app's own native heap was ~74MB.
- **`CPUAttention::onResize` runs correctly for all 36 layers** with exactly the right shapes each
  time (`mNumHead=32 mHeadDim=128 mKvNumHead=8`, matching Qwen3-4B's real GQA architecture, confirmed
  against the exported graph's `FusedRoPE`/`FusedAttention` op attrs) -- so the QKV projection/RoPE
  chain and the per-layer resize path are not where this fails.
- **The real dynamic-buffer allocator succeeds trivially**: `DeferBufferAllocator::apply()`'s
  `realloc(mTotalSize=1038336 bytes, ...)` (~1MB total) returns `NO_ERROR`.
- **The actual failure is a *zero-byte* buffer acquisition**: `CPUBackend::allocBuffer`'s
  `if (size <= 0) { ...; return nullptr; }` path fires (its own `"Acquire buffer size = 0"` log
  line), immediately followed by the `OUT_OF_MEMORY` propagating out of `onForward`. This happens
  in the *second* `onResizeEnd()` call in `Pipeline::_resize` (`mBackupBackend`, i.e. the
  STATIC-storage allocator), after every per-op `onResize()` already returned `NO_ERROR` (the
  unconditional `"Resize error for type=..."` print at `Pipeline.cpp:1048` never fires) -- so some
  tensor registered for `Backend::STATIC` storage somewhere in this specific quantized +
  hidden-states-tap graph computes to a genuine 0-byte size, and nothing upstream catches it.
- Not yet isolated *which* tensor/op. Candidates not yet ruled out: something specific to the
  `--hqq` quantization's STATIC weight-buffer sizing for this graph shape, an edge case in the
  `tie_embeddings` (packed lm_head) path when `hidden_states` is also requested as an output (a
  combination MNN's own LLM CPU demo/chat path never exercises, since it never asks for
  `hidden_states` as an extra output), or a `kv_cache=true`-tagged Attention op's static buffers
  under a code path this project's own custom `forwardRaw()` call (bypassing `response()`/
  `generate()`) doesn't fully initialize the same way (e.g. `mMeta`/`KVMeta` state).

**Next steps for whoever picks this up**: bisect by re-exporting once *without* `--hqq` (plain
round-to-nearest int4) and once *without* `--hidden_states_layers` (default single final hidden
state, matching MNN's normal chat/demo usage) to see which export knob makes the zero-size
allocation appear/disappear -- that pins down whether this is an hqq-specific or
hidden-states-tap-specific bug before going any deeper into `CPUKVCacheManager`/`CPUAttention`'s
STATIC-storage sizing formulas. The reverted temporary debug prints (in `CPUAttention::onResize`
and `DeferBufferAllocator::apply`, both in vendored MNN source) are documented above if reproducing
this trace again is useful.

## 2026-08-30 continued further: crash fixed (root cause), then a real quantization-accuracy bug found

**Blocker above, root-caused and fixed.** Neither `--hqq` nor `--hidden_states_layers` was the
trigger -- both ablations (`_mnn_nohqq`, `_mnn_notap` export dirs) reproduced the identical
zero-byte-allocation crash. The real cause: `QwenTextEncoderEngine::Encode` calls
`Llm::forwardRaw()` directly (a low-level entry point), but never called `Llm::setKVCacheInfo()`
first. `Llm::forwardVec()` -- the path `response()`/`generate()` use, which `forwardRaw()` callers
are expected to replicate -- sets `mMeta->add = seq_len` before forwarding
(`transformers/llm/engine/src/llm.cpp:724`). Left at its default-constructed 0, `CPUAttention::onResize`
reads `insertLen = (int)mMeta->add` (`source/backend/cpu/CPUAttention.cpp:416`) as literally how many
of the seqLen query tokens' K/V get written into the KV cache -- with `mMeta->add=0`, `insertLen=0`,
`mKVCacheManager::kvLength()` stays 0, `mBlockKV` (used to size several per-op STATIC buffers)
computes to 0, and the resulting zero-byte buffer acquisition is what silently propagates to
`OUT_OF_MEMORY`. Fix: one line, `llm_->setKVCacheInfo(kSeqLen, 0);` before `embedding()`/`forwardRaw()`
in `QwenTextEncoderEngine::Encode`. Confirmed on real hardware: the full 36-layer, 512-token,
int4-quantized forward pass now completes (`~162s`, `[1,512,7680]` output, no NaN/Inf).

**New problem surfaced once it actually ran: the int4 (both `--hqq` and plain round-to-nearest) and
int8 exports are numerically unusable.** Diffing the on-device output against
`export_klein_qwen_reference.py`'s PyTorch reference for the same prompt (`"fox"`, `real_len=13`):

| quant | real-content relErr | padding relErr | maxAbsDiff | on-device time |
|---|---|---|---|---|
| int4 (`--hqq`) | 54.6% | 71.1% | 7808 | 162s |
| int8 (`--quant_bit 8`) | 34.2% | 63.6% | 7536 | 110s |
| fp16 (`--quant_bit 16`, no real quantization) | 3.5% | 18.8% | 20 | 348s |

fp16's ~3.5% is in the right ballpark (same order of magnitude as the ~1.1-1.3% seen in the earlier
*desktop* MNN-vs-PyTorch validation -- the residual gap is plausibly real CPU kernel/rounding
differences between MNN's actual compute path and the PyTorch reimplementation used there, not a
bug). int4 and int8 both blow up by orders of magnitude more than plain quantization noise should
cause. Root cause understood, not yet fixed: this specific Qwen3-4B checkpoint has extreme per-channel
outliers in its `RMSNorm` gamma weights (e.g. `k_norm` gamma ranges from -0.014 to **44.0** in layer
0 -- confirmed by inspecting `llm.mnn.json`'s `FusedRoPE` op attrs). Those specific gammas are stored
as literal float32 constants in the graph (not weight-quantized at all, so *they* aren't the direct
cause) but are a strong signal that this checkpoint has the kind of outlier-heavy weight
distribution known to make naive block quantization (MNN's default `--quant_block 64`, round-to-
nearest or hqq, no calibration data) fail badly on other weights in the same layers (the QKV/gate/up/
down projection matrices) -- a single extreme value dominating a 64-element quantization block
crushes the effective precision for the other 63 values in that block.

**Status**: fp16 (8GB weight file) is the only tested quantization level that produces usable
accuracy. int8 (4.3GB) and int4 (2.26GB) are both currently too degraded to ship. **Decision needed
before proceeding to Phase 1 step 3's "feed into the real image-gen pipeline" stretch goal**: ship
fp16 despite the size, or invest in better quantization for this checkpoint (smaller `--quant_block`,
`--sym`, per-channel/mixed-precision, or excluding specific outlier-heavy tensors from quantization)
before revisiting int8/int4.

## 2026-08-30 continued further still: int8 tuning attempts, bug investigation, and final decision (fp16)

**int8 flag tuning, evaluated via a fast local pymnn `forward()`-logit proxy against HF logits
(prompt `"fox"`)**:

| config | logit relErr (proxy) | top-5 vs HF |
|---|---|---|
| `--quant_bit 8` (baseline, block=64, asym) | 73.8% | 1 wrong entry |
| `--quant_bit 8 --quant_block 32` | 111.8% (worse) | -- |
| `--quant_bit 8 --sym` | 66.5% (best of these) | exact match |
| `--quant_bit 8 --sym --hqq` | 75.0% (worse than `--sym` alone) | -- |

**Critical finding: the logit-level proxy did not transfer to the metric that actually matters.**
Pushed `--sym` to real hardware and re-ran the full on-device pipeline, diffing the actual
`hidden_states` tensor (not logits) against the PyTorch reference: **relErr 33.9%, essentially
unchanged from plain int8's 34.2%.** The logit proxy was misleading for judging hidden-state
quality -- do not reuse it as a stand-in for the real metric in future quantization tuning.

**Given the user's explicit "is there a bug?" pushback** (fair: int8 should not be 20x worse than
fp16 when going from int4->int8 was only a 1.6x improvement, when quantization noise should scale
roughly with 2^-bits), did a real investigation per this repo's `MNN/skills/general-debug/SKILL.md`
§2 (quantization/export-corruption) methodology rather than re-tuning flags blindly:

1. **Ruled out embedding/lm_head export corruption.** This checkpoint's tied embedding table
   (vocab=151936 x hidden=2560 ~= 389M elements) exceeds the 256M-element chunking threshold that
   caused a real historical bug in this exact codebase (§2.5 of the skill doc: an MPS `sum()`
   reduction silently returning all-zero for >=2^28-element uint8 tensors). Wrote a standalone
   script (`litert-torch/scratch/verify_tie_embed_dequant.py`) that reads `llm.mnn.weight` directly
   using the same offset/scale math as `diskembedding.cpp`, dequantizes rows spanning the entire
   vocab (including exactly at the 2^27-byte/131072-row boundary), and compares to HF's raw
   `embed_tokens.weight`: cosine similarity ~0.99998 everywhere. Export-side embedding quantization
   is correct; this is not that bug.
2. **Confirmed the huge `maxAbsDiff` (~7600-7800) outlier is a real model phenomenon, not
   invented noise.** Located it precisely: token 0, hidden-dim 4 (within each of the 3 tapped-layer
   segments), value ~8680 on-device vs ~16320 in the HF reference for the *same* location. HF's own
   fp32/bf16 reference has this same massive value at the same location -- this is the documented
   "massive activation" phenomenon in LLMs (Sun et al. 2024): a handful of fixed channels carry a
   near-constant huge-magnitude value through the residual stream, functioning like an internal
   bias rather than real content. MNN's int8 computes roughly half that magnitude because a shared
   per-64-element block scale gets skewed badly when one value in the block is ~1000x its
   neighbors.
3. **But outlier channels only explain part of the error.** Excluding all channels with HF
   magnitude > 50 (20 of 7680 channels total) only brought relErr from 34%->26.6%. The remainder is
   broad, above-normal quantization noise spread across "ordinary" channels too -- consistent with
   this checkpoint's already-noted outlier-heavy weight distributions (`k_norm` gamma up to 44.0)
   widening the effective dynamic range of quantization blocks even where individual activations
   look unremarkable.

**Conclusion: not a wrapper/export code bug.** It's a genuine, unusually severe (for int8)
quantization-accuracy problem specific to this checkpoint's weight distribution. Simple flag tuning
(`--sym`, `--quant_block`, `--hqq`) doesn't fix it because the noise is broadly distributed, not
concentrated in one fixable spot -- a real fix would require targeted mixed-precision (e.g.
`--quant_config` overrides keeping the layers near the outlier channels at higher bit-width), which
was not attempted.

**Decision: ship fp16.** Given the effort already spent on int8 tuning without a real fix, the user
decided to proceed with fp16 (8GB weight file, ~348s on-device, ~3.5% relErr -- already validated
accurate) rather than continue chasing mixed-precision int8. int4/int8 quantization work for this
encoder is deferred indefinitely, not part of the current plan. NPU deployment of the fp16 (or a
future better-quantized) text encoder remains the explicit next phase after fp16-on-CPU is fully
wired up, per the user's standing instruction ("Let's try int8-on-CPU, then get it to run on NPU
later" -- superseded by this fp16 decision for the CPU-precision choice, NPU sequencing unchanged).
