package com.pockettavern.app.data.local.inference

/**
 * Raw JNI surface for the production FLUX.2 [klein] pipeline (`app/src/main/cpp/npu/
 * KleinDiffusionEngine.{hpp,cpp}`, called from `jni_diffusion.cpp`) — no business logic here,
 * matches the thin style of [MnnDiffusionBridge]. See [KleinDiffusionEngine] for the actual
 * generation flow.
 *
 * Unlike [MnnDiffusionBridge], there is no create/load/destroy handle lifecycle: Klein's own
 * diagnostic architecture already reloads every component (including the ~8GB fp16 text encoder)
 * per invocation, so a single blocking call is the natural shape here — no persistent native
 * state to manage or leak.
 */
object KleinDiffusionBridge {
    init {
        System.loadLibrary("pockettavern_diffusion")
    }

    /**
     * Blocking — runs the full prompt-encode / 4-step NPU denoise / NPU VAE-decode pipeline on
     * the calling thread. Must be called from a background dispatcher, never the main thread.
     * Confirmed on real hardware this is a multi-minute call (encoder alone is ~120-350s,
     * see docs/flux2-klein-conversion.md). [progressCallback] is invoked synchronously, on that
     * same calling thread, once per pipeline phase (13 total: 1 text encode + 4 denoise steps +
     * 8 VAE decode stages).
     *
     * @param npuModelDir flat directory of `*_Google_Tensor_G5.tflite` files plus
     *   `qk_norm_scales/` (see `scripts/stage_klein_npu_artifacts.sh`) — manually staged, not
     *   downloaded (see [KleinModelManager]).
     * @param dispatchLibDir the app's own native library directory
     *   (`context.applicationInfo.nativeLibraryDir`) — bundles the gated Tensor NPU dispatch
     *   `.so`, already part of the APK.
     * @param qwenConfigPath the fp16 Qwen3-4B encoder's exported `config.json`.
     * @param mmapCacheDir a writable, app-private directory (e.g. `context.cacheDir`) MNN
     *   materializes the ~8GB fp16 weight file into as a file-backed mmap allocation instead of
     *   anonymous heap memory -- what actually keeps Android 17's Memory Limiter from OOM-killing
     *   the process (its `memory.high` cgroup ceiling is enforced on anon+swap; file-backed pages
     *   are reclaimable instead of needing swap). See `QwenTextEncoderEngine::Load()`'s doc
     *   comment for the full explanation. Persists across app launches (`use_cached_mmap`), so
     *   pass the same directory every call.
     * @param seed must already be resolved to a non-negative value (a negative "random seed"
     *   request is resolved Kotlin-side, matching [MnnDiffusionEngine]'s convention).
     */
    external fun nativeGenerate(
        npuModelDir: String,
        dispatchLibDir: String,
        qwenConfigPath: String,
        mmapCacheDir: String,
        prompt: String,
        outputPngPath: String,
        seed: Int,
        progressCallback: (Int) -> Unit,
    ): Boolean
}
