# SDXL UNet on Tensor G5 NPU — handoff notes

**Status as of 2026-08-27 (updated same day, latest pass)**: The core architecture is proven
correct and the memory problem that motivated this whole effort is solved. The apparent native
C++ performance regression was Doze throttling, not a C++ regression. The 36-piece NPU UNet is
wired into PocketTavern's real SDXL path and confirmed working end-to-end on-device, including a
real persona-avatar generation triggered through the actual app UI (not a diagnostic harness). NPU
support is no longer hardcoded to one model: `SdxlModelManager` now has a generic
`npu-unet/<modelId>/` bundle convention (parallel to `sd-models/<modelId>/`), and there's a real
per-model "Run on" AUTO/CPU/NPU control in Settings -> Image Generation, gating
`MnnDiffusionEngine`'s NPU path via `ImageGenConfig.sdxlRunModeByModel`. Confirmed live on-device
(NPU badge shows correctly, mode selection persists per model, generation succeeds with NPU
selected).

**Real CFG (batch=2) is reliable but not performance-viable.** The epoll crash was caused by
our C++ destruction order, not an unfixable Google Tensor runtime issue: `TensorBuffer`s were
being destroyed before their `CompiledModel`, allowing a stale dispatcher epoll event to collide
with a reused fd in the next piece. `NpuUnetEngine::RunPiece()` now destroys the `CompiledModel`
before its buffers, with the same order guaranteed on error returns. With no inter-piece sleep,
two complete 36-piece batch-2 forwards in one `LiteRtEnvironment` completed correctly on the
Pixel 10 Pro XL (114.1s total; duplicated batch rows were exactly equal). However, one batch-2
forward takes about 61–64s, making a 20-step image roughly 20 minutes. Do not wire it into the app.
The app now uses the materially faster strategy: two independent **batch-1** engines, one for
the unconditional CFG row and one for the conditional row, run concurrently for each step. The
native bridge combines their outputs with normal CFG before the scheduler update. A real
desktop-driven 20-step image using this exact pair-of-engines design produced substantially
better output than the former conditional-only path; one on-device forward completed in 32.610s.

This doc is the handoff summary for continuing this work in a fresh agent/session (originally
built with Claude Code; being handed to Codex). It supersedes any older content that was here
before — this file was stale/pre-dated this investigation.

## The goal

Replace `MnnDiffusionEngine`'s ~30s/step CPU UNet forward pass (via MNN, vendored at
`app/src/main/cpp/MNN`) with a version that runs on the Pixel 10 Pro XL's Tensor G5 NPU, using
Google's gated Tensor AOT compiler SDK + the public LiteRT runtime.

## What's proven (do not re-derive these — they cost real effort to establish)

1. **The UNet converts and AOT-compiles as 36 separate pieces**, each a single `DISPATCH_OP`
   custom-op `.tflite` file (opaque pre-compiled NPU bytecode blob). Plain fp32 throughout — int8
   quantization was explored and abandoned (see "Dead ends" below); it's not needed once the
   architecture below is used, since memory no longer scales with precision.
2. **Running the 36 pieces as ONE merged multi-op `CompiledModel` blows up memory**: ~12-13GB
   `dmabuf_rss` real cost (confirmed via a real `lowmemorykiller` kill,
   `dmabuf_rss=12702716kB`), because the dmabuf cost scales with dispatch/op COUNT in one
   session, not weight size or precision. This made the merged-file approach unusable.
3. **Fix: run each piece as its own separate `CompiledModel` instance** (create → run → close),
   closing between pieces so peak memory is bounded by the largest single piece, not the sum of
   all 36. This requires every piece's real inputs to be wrapped in a same-shape `RESHAPE` node
   (a cheap, real op) within that piece's own session — **without this, standalone pieces produce
   silently WRONG output** (deterministic per-context, but different from the correct merged-graph
   result — a genuine LiteRT/NPU dispatch quirk about tensor provenance: internally-produced
   tensors behave differently from externally-supplied ones for at least some tensor shapes).
   RESHAPE-wrap fix: `~/code/litert-torch/scratch/build_reshape_wrapped_piece.py`.
4. **Confirmed via a real 20-step diffusion loop** (desktop Python driving scheduler/text-encode/
   VAE-decode, each step's UNet forward done on-device): completed cleanly, all 20 steps, valid
   converging output, real image decoded — where the old merged-file approach died at step 6
   every time. **This is the proof the memory problem is actually solved**, not just plausible.
5. **Ported to real C++ (`app/src/main/cpp/npu/NpuUnetEngine.{hpp,cpp}`)**, linking directly
   against LiteRT's C API (`libLiteRt.so`, NOT the Kotlin/JNI wrapper). Compiles, links, and — as
   of the last on-device smoke test — **produces correct output matching the Kotlin reference to
   displayed float precision**.
6. **Wired into the real SDXL path**: `MnnDiffusionBridge.nativeCreate()` optionally configures
   `StableDiffusionXL`, whose `load()` uses LiteRT for the UNet and keeps MNN for text encoders,
   scheduler and VAE. `MnnDiffusionEngine` keys an NPU bundle to the matching MNN model ID,
   preventing a silent mix of incompatible checkpoint weights.

## Performance re-check (2026-08-27): Android Doze caused the apparent C++ regression

The accelerator-options hypothesis was tested on the same Pixel after the buffer-requirements
fix:

- Explicit `kLiteRtHwAcceleratorNpu` fails on the first (`embed`) wrapped piece with
  `kLiteRtStatusErrorCompilation` (504).
- Leaving a non-null `LiteRtOptions` object's accelerators unset fails with
  `kLiteRtStatusErrorInvalidArgument` (1).
- `kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu` remains the only working C API
  configuration. This is expected: every standalone piece contains a CPU `RESHAPE` wrapper in
  addition to its AOT NPU dispatch op.

The historical 20-step desktop run was found at
`~/code/litert-torch/scratch/npu_diffusion_run_separate.log`: its real Kotlin steps took
19.0–21.6s and its final output starts `[1.265625, 0.265625, 0.17675781]`. The same model files
and final real inputs were rerun successfully, but initially took **39.69s**. At that time,
`dumpsys power` reported `mWakefulness=Dozing` and `mHalInteractiveModeEnabled=false`.

Waking the phone (`adb shell input keyevent KEYCODE_WAKEUP`) before the exact Kotlin real-input
step restored it to **19.15s**, with the exact same final output. The native C++ 36-piece smoke
chain, run while the phone remained interactive, completed in **18.97s** with its known-correct
output. Therefore the previous 33–34s C++ observation was a Doze-throttled measurement, not a
C++/LiteRT-C performance regression.

`run_npu_diffusion.py` now wakes and verifies the device before every step so its 20-step test is
reproducible. The production NPU generation flow must similarly keep the device interactive (for
example, with a scoped keep-screen-on policy) while a generation is active; a CPU partial wakelock
alone may not prevent this NPU throttling. Keep `NPU | CPU`; CPU is required for the RESHAPE
wrappers and is not a removable delegate candidate.

## Two non-obvious LiteRT C API facts, hard-won on-device (would burn real time to rediscover)

1. **Manually-sized tensor buffers can be too small even when the byte math is "correct."**
   `num_elements * sizeof(float)` is NOT always enough — the NPU dispatch layer can require
   padding/alignment beyond the raw tensor size. Confirmed on-device: a `[1,6]` float32 tensor
   (24 bytes of real data) needed a 64-byte buffer; a naively-sized 24-byte buffer produced a
   dispatch-layer warning ("MediaTek dispatch API returned contradictory buffer requirements..."
   — this is generic delegate-kernel log text, NOT an indication of actually using MediaTek
   hardware) and, in at least one configuration, an outright failure. **Fix**: always query
   `LiteRtGetCompiledModelInput/OutputBufferRequirements` and create buffers via
   `LiteRtCreateManagedTensorBufferFromRequirements`, never compute buffer size by hand.
2. **`LiteRtCreateCompiledModel`'s `compilation_options` argument is fussy** despite
   `litert_compiled_model.h` documenting it as "optional and can be null": passing `nullptr`
   produced `kLiteRtStatusErrorInvalidArgument`. Passing `kLiteRtHwAcceleratorNpu` alone produced
   `kLiteRtStatusErrorCompilation` (likely actually caused by bug #1 above, not the accelerator
   flags — untested independently, see "Open issue"). What currently works: a real
   (non-null) `LiteRtOptions` object with
   `LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorNpu | kLiteRtHwAcceleratorCpu)`.

## Dead ends (don't retry without new information)

- **Int8 quantization of attention blocks**: AOT-compile fails unconditionally on this SDK build
  for any int8-quantized attention graph, regardless of segment size or op scoping. Confirmed via
  3 separate experiments, all failing identically (`Compilation has failed with error type:
  INTERNAL`). Not worth revisiting without an SDK update from Google.
- **A cheap "dummy predecessor op" trick** to make a standalone piece behave like it's mid-session
  (to avoid the RESHAPE-wrap fix's cost) — directly disproven. What matters is that a tensor was
  produced BY a real preceding op in the SAME session, not merely that "some op" ran first.
  RESHAPE is the actual (cheap) fix; there's no shortcut past it.
- **MNN `Backend` extension** (adding a formal NPU backend to the vendored MNN fork, like the
  existing HiAI/QNN/NNAPI/CoreML backends) — investigated and rejected: MNN's `Backend` interface
  assumes per-op dispatch (`Backend::onCreate` is called once per graph `Op`), which doesn't fit
  36 opaque whole-subgraph pre-compiled blobs with no MNN op IR inside them. The `Module` layer
  (used by `IfModule`/`WhileModule`/`MoEModule` for "one graph node → arbitrary C++ logic,
  bypassing Backend/Execution entirely") would be the right MNN extension point if a formal
  MNN-native integration is ever wanted, but the current approach (a standalone `NpuUnetEngine`
  class called directly from `unet()`, not going through MNN's Module/graph system at all) is
  simpler and was chosen instead — no FlatBuffer schema changes, no `PipelineModule.cpp` edits.

## Real CFG (batch=2): fd lifetime ordering fix

Everything on the conversion/compile side of batch=2 CFG works: all 36 pieces convert and
AOT-compile at batch=2 with 100% NPU offload, zero failures (see
`~/code/litert-torch/scratch/build_batch2_all.sh`, `.summary`). `NpuUnetEngine::Load()` and
`StableDiffusionXL::configureNpuUnet()` both take a `batch` parameter (default 1, fully
backward-compatible with the shipped batch=1 path); `unet()`'s NPU branch builds a real stacked
[uncond, cond] input and applies the CFG formula on a real batch=2 output when `batch()==2`.

The initial native batch-2 run aborted inside the dispatcher:

```
F0000 ... :833] Check failed: pending_sink_fds.erase(fd) > 0 (0 vs. 0) Unexpected fd from epoll: 0
Fatal signal 6 (SIGABRT) ... Thread-5
```

The apparently timing-sensitive workaround was a 50ms sleep after each piece teardown. It let the
chain complete, but added 1.8 seconds per forward and was not a correctness guarantee. The actual
problem was visible in `RunPiece()`'s local declaration order: `cm_holder` was declared before
the input/output `TensorBufferHolder` vectors, so ordinary reverse-order C++ destruction closed
the buffers' fds before `LiteRtDestroyCompiledModel()` could unregister its epoll state.

The fix declares the buffer owners before `cm_holder` (making this order safe for every early
return too) and explicitly destroys the successful piece's compiled model while all buffers are
still alive. No sleep is used. This preserves the separate-piece memory bound; it changes only
teardown ordering.

On 2026-08-27, `nativeRunUnetEngineBatch2Smoke` ran two consecutive full 36-piece forwards in a
single engine/environment with no delay and completed in 114,079ms. The result had 131,072 floats,
no NaN/Inf, and `maxRowDiff=0` for identical inputs in the two CFG rows. A prior one-forward run
also completed in 59,732ms. This validates the dispatcher lifetime pattern required by successive
denoising steps, but it is not yet a real prompt/image CFG validation.

### Performance result: do not use batch 2 for CFG

The fixed batch-2 execution is still about 61–64s per UNet forward while the device is awake. The
dominant cost is NPU execution, not model loading, buffer copies, or fd teardown: each of the five
large spatial-attention pieces takes about 6.7–7.3s on the batch-2 AOT binaries. This makes the
batch-2 design slower than the CPU path for a 20-step image.

Two independent batch-1 engines can be launched concurrently, one for the unconditional row and
one for the conditional row. Four same-environment passes completed in 31,241ms, 29,562ms,
28,914ms, and 27,879ms. A real one-step desktop-driven CFG pass took 32.610s and the full
20-step output was visually confirmed to be substantially better than conditional-only output.
This is now the production NPU CFG strategy. It is slower than the old ~19s single-row NPU
benchmark, but still materially faster than the ~50-52s CPU UNet step while retaining quality.

Batch 2 does **not** receive a comparable later-step speedup: three consecutive same-environment
passes took 63,583ms, 61,074ms, and 62,153ms. The device was awake and Android reported
`Thermal Status: 0` throughout. Vendor skin sensors reached a mild warning level during the long
run, but there was no Android thermal throttle and the timing stayed flat rather than improving.
Thus neither AOT compilation nor normal warm-up explains the batch-2 slowdown.

## What's done (this pass, 2026-08-27)

- Multi-step real PocketTavern NPU generation, triggered through the actual app UI (persona
  avatar flow), confirmed working with NPU mode selected.
- The scoped interactive-device policy (`OnDeviceImageGenerationScreenState` /
  `FLAG_KEEP_SCREEN_ON`) is live for both CPU and NPU generation.
- **NPU support generalized beyond the one hardcoded `pureTukanoNSFW-xl` check** (former item #5
  below) and a real run-mode toggle now exists (former item #1 not yet done, but this
  supersedes it): see "Generic NPU bundle support + run-mode toggle" below.
- **Batch-2 fd lifecycle fixed**: `CompiledModel` is now destroyed before every one of its tensor
  buffers, replacing the 50ms inter-piece sleep. The two-forward on-device batch-2 smoke test
  completed without an epoll error.
- **Production NPU CFG**: `StableDiffusionXL` now loads two independent batch-1
  `NpuUnetEngine`s, sends the unconditional and conditional rows to them concurrently at every
  denoising step, and applies `uncond + cfgScale * (cond - uncond)` before scheduling. The UI no
  longer claims that NPU ignores the negative prompt or CFG scale.

## What's NOT done yet (in priority order)

1. **All pipeline stages on NPU, not just UNet.** Text encoder(s) and VAE decode still run on
   MNN/CPU always, regardless of `sdxlRunMode`. Converting them to LiteRT/NPU is a materially
   bigger effort than UNet was — different op mixes, likely new AOT dead-ends to rediscover from
   scratch (int8-attention-style blockers are not guaranteed to be the only ones). Not started;
   no design work done yet on what these pieces' op graphs look like or whether the same
   RESHAPE-wrap/separate-instances architecture will even be needed (may depend on op count).
2. **Further CFG performance work.** The production pair-of-batch-1 strategy is about 28-33s per
   step and is a quality-preserving win over CPU, but not as fast as the old 19s single-row
   benchmark. A future attempt needs a graph/compiler strategy that improves the five large
   spatial-attention kernels. Do not reintroduce a timing sleep: the lifetime fix above is the
   required teardown behavior.
3. **Model file distribution**: the 36 wrapped `.tflite` pieces (~5GB total) are currently only
   on the test device via manual `adb push` (see below) — there's no `SdxlModelManager`-style
   download flow for them (the *catalog/selection* side now exists generically, see below — only
   the download mechanism is still missing).
4. **Numeric correctness vs. a real PyTorch fp32 reference** (not just "converges, doesn't
   crash") is still open — always been secondary priority behind the memory problem, still true.
5. **On-device cleanup**: several GB of throwaway `NpuDiagnostic.kt`-era files sit loose in
   `filesDir` root on the test device (`unet_merged_full.tflite` 5GB, `up_merged.tflite` 2.7GB,
   `down_merged.tflite` 1.7GB, several `clip*.tflite`/`mid_*.tflite` files) — leftover from the
   investigation, not referenced by production code. `NpuDiagnostic.kt` itself (~1700 lines,
   explicitly marked throwaway in its own comments) is also still present but no longer wired
   into `MainActivity` (that hook was removed since it's superseded by the real integration).
   Safe to delete once confirmed nothing else references it — not done yet, don't delete without
   checking first.

## Generic NPU bundle support + run-mode toggle (done, 2026-08-27)

NPU is no longer gated by a hardcoded model-id string match. `SdxlModelManager`
(`data/local/inference/SdxlModelManager.kt`) now has a parallel bundle-directory convention:
`filesDir/npu-unet/<modelId>/`, keyed by the same `modelId` as its MNN model set in
`filesDir/sd-models/<modelId>/` — an NPU bundle's baked-in weights must never be paired with a
different checkpoint's MNN weights, so the join key is deliberate. `hasNpuBundle(modelId)` /
`npuBundlePathFor(modelId)` are the new accessors (best-effort non-empty-directory check; real
per-file validation still happens in `NpuUnetEngine::Load()` on the native side).

`MnnDiffusionEngine.ensureLoaded()` resolves the bundle generically instead of hardcoding
`pureTukanoNSFW-xl`, gated by a new `ImageGenConfig.sdxlRunMode` field (`SdxlRunMode` enum in
`domain/model/ImageGenModels.kt`: `AUTO` — use NPU if a bundle exists for the selected model, else
CPU; `CPU` — always MNN, even if a bundle exists; `NPU` — require a bundle, throw a clear
`IllegalStateException` instead of silently falling back to CPU if none exists). Threaded through
`MnnDiffusionBackend` (reads `settingsDataStore.getImageGenConfig().sdxlRunModeType`) into
`MnnDiffusionEngine.generate(..., runMode: SdxlRunMode)`. A run-mode change alone (same model
path) now correctly triggers a native handle reload — `ensureLoaded` tracks `loadedRunMode`
alongside `loadedModelPath`.

UI: `ImageGenSettingsScreen.kt`'s `SdxlModelSection` shows an "NPU" badge next to any downloaded
model set that has a bundle, plus a `SingleChoiceSegmentedButtonRow` "Run on" control
(AUTO/CPU/NPU) with a live description of what the current mode will actually do for the selected
model. `NPU` is disabled in the control when the selected model has no bundle.

**Important: the C++ `NpuUnetEngine`/`PieceSpec` table did NOT need to change for this.** It
already takes `model_dir` as a `Load()` parameter, and the 36-piece topology/role manifest is a
property of the SDXL UNet *architecture* (any SDXL-1.0-shaped checkpoint shares it), not of one
checkpoint's specific weights — a new NPU-compiled model just needs its own 36 wrapped `.tflite`
files in a same-shaped `npu-unet/<newModelId>/` directory. This only holds for another SDXL 1.0
UNet-shaped checkpoint; a genuinely different architecture (e.g. SD1.5) would need a new manifest.

On-device migration performed on the one test device (via adb, not through the app):
`files/unet_wrapped/` → `files/npu-unet/pureTukanoNSFW-xl/` (36 files, verified intact after
move). The old `unet_wrapped` path is gone — don't assume it still exists if picking this up
fresh. The throwaway `nativeRunUnetEngineSmoke` JNI entry point in `jni_diffusion.cpp` and its
`BuildConfig.DEBUG`-only trigger in `MainActivity.onCreate` were removed in the same pass (dead
code after the rename, and superseded by the real integration anyway).

## Key files

**PocketTavern app** (`~/code/PocketTavern`):
- `app/src/main/cpp/npu/NpuUnetEngine.{hpp,cpp}` — the real C++ NPU UNet engine. Takes `model_dir`
  as a `Load()` parameter; the 36-piece `PieceSpec` table is a property of the SDXL UNet
  architecture, not one checkpoint, so this file does not need per-model changes.
- `app/src/main/cpp/jni_diffusion.cpp` — production JNI (`nativeCreate`/`nativeLoad`/
  `nativeGenerateXL`) plus the real `configureNpuUnet` wiring. The old throwaway
  `nativeRunUnetEngineSmoke` entry point has been removed (superseded).
- `app/src/main/cpp/CMakeLists.txt` — links `libLiteRt.so` via an IMPORTED target resolved from
  `LITERT_JNI_DIR`, which `app/build.gradle.kts` supplies (extracted from the `litert:2.2.0` AAR
  via a `Sync` task, not a hardcoded Gradle-cache path — that was fixed).
- `app/src/main/cpp/litert_headers/` — vendored public LiteRT C API headers (from
  `google-ai-edge/LiteRT` GitHub, not the gated SDK — small text files).
- `app/src/main/kotlin/com/pockettavern/app/util/NpuDiagnostic.kt` — ~1700 lines of throwaway
  Kotlin diagnostics built up over the investigation (the `UNET_PIECES` table, all the
  `compareXxx`/`runXxx` functions used to establish the facts above). Not production code, not
  wired into the real app anywhere (the `MainActivity` debug hook that called it was removed).
  Safe to delete once confirmed nothing references it — not done, see "On-device cleanup" above.
- `app/src/main/kotlin/com/pockettavern/app/data/local/inference/MnnDiffusionEngine.kt` — the
  real, shipped SDXL pipeline. `ensureLoaded()` resolves an NPU bundle generically via
  `SdxlModelManager` and gates it on `ImageGenConfig.sdxlRunModeType`/`SdxlRunMode`
  (AUTO/CPU/NPU); `generate(..., runMode)` and `loadedRunMode` track this alongside
  `loadedModelPath` so a mode change alone triggers a reload.
- `app/src/main/kotlin/com/pockettavern/app/data/local/inference/SdxlModelManager.kt` — owns both
  `sd-models/<modelId>/` (MNN weights) and the new `npu-unet/<modelId>/` bundle convention
  (`hasNpuBundle`/`npuBundlePathFor`), same `modelId` join key on purpose (an NPU bundle's weights
  must never be paired with a different checkpoint's MNN weights).
- `app/src/main/kotlin/com/pockettavern/app/domain/model/ImageGenModels.kt` — `SdxlRunMode` enum
  and `ImageGenConfig.sdxlRunModeByModel` (per-model, NOT a single global setting — switching
  models must not carry e.g. "NPU" onto a model with no bundle).
- `app/src/main/kotlin/com/pockettavern/app/ui/screens/settings/ImageGenSettingsScreen.kt` /
  `ImageGenSettingsViewModel.kt` — `SdxlModelSection`'s "NPU" badge + "Run on" AUTO/CPU/NPU
  segmented control, confirmed working live on-device.
- `app/src/main/cpp/MNN/transformers/diffusion/engine/src/stable_diffusion_xl.cpp` — the vendored
  MNN pipeline (submodule, the user's own fork). `configureNpuUnet()` + the `mNpuUnet` branch in
  `unet()` is the real integration, already landed — not a TODO anymore.

**Conversion pipeline** (`~/code/litert-torch/scratch/`, desktop-only, not part of the app):
- `build_full_unet_wrapped.py` — batch-builds all 36 RESHAPE-wrapped pieces + emits
  `full_unet_pieces.json` (the manifest both `NpuDiagnostic.kt`'s `UNET_PIECES` and
  `NpuUnetEngine.cpp`'s `PieceSpec` table were transcribed from — regenerate from here, don't
  hand-edit either transcription independently if the piece set ever changes).
- `build_reshape_wrapped_piece.py` — the generic per-piece RESHAPE-wrap builder.
- `run_npu_diffusion.py` — the real 20-step diffusion loop validation script (desktop
  scheduler/text-encode/VAE + real per-step on-device NPU UNet calls via adb). Confirmed working
  end-to-end. Uses `watchdog_run.sh` for memory safety (mandatory on this dev machine for any
  multi-GB model load — see that script's own comments).
- `merge_compiled_pieces.py` / `merge_full_unet*.py` — the OLDER single-merged-file approach
  (superseded by the separate-instances architecture, kept for reference/history, not the current
  design).

## On-device testing mechanics

- Pixel 10 Pro XL, non-rooted debug build. Push files into app-private storage via:
  `adb push foo /sdcard/foo && adb shell "cat /sdcard/foo | run-as com.pockettavern.app sh -c 'cat > /data/data/com.pockettavern.app/files/foo'"`
  (plain `run-as cp` from `/sdcard` fails — SELinux/scoped-storage restriction).
- The 36 wrapped piece files are on the current test device at
  `/data/data/com.pockettavern.app/files/npu-unet/pureTukanoNSFW-xl/` (36 files, ~5GB) — moved
  there from the old `unet_wrapped/` path on 2026-08-27 to match `SdxlModelManager`'s generic
  `npu-unet/<modelId>/` convention; the old path no longer exists. If a fresh device/reinstall
  wipes this, rebuild+repush from `~/code/litert-torch/scratch/models/full_unet_wrapped/` into
  `npu-unet/<modelId>/` (`<modelId>` must match the corresponding `sd-models/<modelId>/` dir).
- `app/src/main/jniLibs/arm64-v8a/libLiteRtDispatch_GoogleTensor.so` (the gated Tensor NPU SDK's
  dispatch library) is committed to the repo — the user confirmed they never agreed to any
  redistribution restriction when granted closed-beta access, so it ships like any other
  prebuilt native lib in this project.
- Build: `./gradlew assembleDebug` from `~/code/PocketTavern`. Install:
  `adb install -r app/build/outputs/apk/debug/app-debug.apk`. Force a fresh `onCreate` (not just
  re-showing the activity) with `adb shell am force-stop com.pockettavern.app` before
  `adb shell am start -n com.pockettavern.app/.MainActivity`.
- Logs: `NpuDiagnostic` tag (Kotlin diagnostics, no longer wired to anything real), `MnnDiffusionEngine`/
  `DebugLogger` tag (Kotlin production path, e.g. "Configured LiteRT NPU UNet for SDXL"),
  `PocketTavernDiffusion` tag (native — both `jni_diffusion.cpp`'s own `LOGE` and
  `NpuUnetEngine.cpp`'s `NPU_LOGE`, which was deliberately changed from MNN's own `MNN_ERROR`
  macro because `MNN_ERROR` produced no visible logcat output at all when called from this
  target — root cause not fully understood, see the comment above `NPU_LOGE`'s definition in
  `NpuUnetEngine.cpp` if revisiting this).
- **Do not drive this device's UI via `adb shell input tap`/`swipe` to test changes** — this is
  the user's real personal device with real, irreplaceable data (chats, personas, characters).
  Build+install and check logcat, then ask the user to exercise the flow themselves.
