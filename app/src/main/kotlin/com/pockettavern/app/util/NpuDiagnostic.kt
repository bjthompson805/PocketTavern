package com.pockettavern.app.util

import android.content.Context
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import java.io.File
import kotlinx.coroutines.runBlocking

/**
 * TEMPORARY one-off diagnostic: does a real Google Tensor NPU AOT-compiled model (produced via
 * the gated Google Tensor SDK's `aot_compile`, see ~/code/litert-torch/scratch/run_aot.py --
 * 627/627 CLIP ops fully offloaded to the Tensor_G5 target at compile time) actually load and run
 * on-device via the official `CompiledModel` API, and is it faster than the CPU baseline?
 *
 * This supersedes the earlier raw-JNI classic-NNAPI diagnostic (deleted jni_nnapi_diag.cpp): that
 * approach hit a hard wall (Darwinn rejected float32 operands, "Ops supported = 0"). The real,
 * supported path turned out to be `com.google.ai.edge.litert:litert-api`'s `CompiledModel` +
 * `Accelerator.NPU`, which consumes a model pre-compiled for the NPU offline (not float32 ops
 * dispatched live through Android's generic NNAPI HAL).
 *
 * Delete this file's diagnostic-specific logging/timing once the question is answered, or fold it
 * into a real ImageGenBackendType if NPU compilation proves viable for the full SDXL pipeline.
 *
 * Correctness check: runs the same real (non-zero) token sequence through both the AOT
 * NPU-compiled model (clip_npu.tflite, Accelerator.NPU) and the plain fp32 model (clip_cpu.tflite,
 * Accelerator.CPU -- LiteRT's own XNNPACK CPU backend), and diffs the two outputs. The token
 * sequence is "a photograph of an astronaut riding a horse" (CLIP-ViT-L/14 BPE, sd15 tokenizer);
 * the expected values below are from onnxruntime running the original fp32 ONNX export on the
 * same input (~/code/litert-torch/scratch/full_pipeline.py's export step), i.e. a third,
 * independent reference: output[0,0,:5] = [-0.3758, 0.0174, -0.0612, -0.1855, -0.0404].
 */
object NpuDiagnostic {
    private const val TAG = "NpuDiagnostic"

    // THROWAWAY smoke test for the native (C++/LiteRT-C-API) NpuUnetEngine -- see
    // jni_diffusion.cpp's Java_..._nativeRunUnetEngineSmoke doc comment. Loads
    // pockettavern_diffusion (the MNN diffusion JNI lib) explicitly since nothing else on this
    // path guarantees it's already loaded.
    private external fun nativeRunUnetEngineSmoke(modelDir: String, dispatchLibDir: String): String

    fun runNativeUnetEngineSmoke(context: Context) {
        try {
            System.loadLibrary("pockettavern_diffusion")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== native UnetEngine smoke test: FAILED to load pockettavern_diffusion ===", e)
            return
        }
        val modelDir = File(context.filesDir, "npu-unet/pureTukanoNSFW-xl")
        if (!modelDir.exists()) {
            Log.w(TAG, "$modelDir not found, skipping native UnetEngine smoke test")
            return
        }
        val dispatchLibDir = context.applicationInfo.nativeLibraryDir
        Log.i(TAG, "=== native UnetEngine smoke test starting (modelDir=$modelDir dispatchLibDir=$dispatchLibDir) ===")
        val result = nativeRunUnetEngineSmoke(modelDir.absolutePath, dispatchLibDir)
        Log.i(TAG, "NATIVE_UNET_ENGINE_SMOKE: $result")
    }

    private external fun nativeRunTextEncoderEngineSmoke(modelDir: String, dispatchLibDir: String): String

    fun runNativeTextEncoderEngineSmoke(context: Context) {
        try {
            System.loadLibrary("pockettavern_diffusion")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== native TextEncoderEngine smoke test: FAILED to load pockettavern_diffusion ===", e)
            return
        }
        val modelDir = File(context.filesDir, "npu-unet/pureTukanoNSFW-xl")
        if (!modelDir.exists()) {
            Log.w(TAG, "$modelDir not found, skipping native TextEncoderEngine smoke test")
            return
        }
        val dispatchLibDir = context.applicationInfo.nativeLibraryDir
        Log.i(TAG, "=== native TextEncoderEngine smoke test starting (modelDir=$modelDir dispatchLibDir=$dispatchLibDir) ===")
        val result = nativeRunTextEncoderEngineSmoke(modelDir.absolutePath, dispatchLibDir)
        Log.i(TAG, "NATIVE_TEXT_ENCODER_ENGINE_SMOKE: $result")
    }

    private external fun nativeRunKleinSingleBlockSmoke(modelDir: String, dispatchLibDir: String, refDir: String): String
    private external fun nativeRunKleinDoubleBlockSmoke(modelDir: String, dispatchLibDir: String, refDir: String): String
    private external fun nativeRunKleinZeroCopyQkvFlashProbe(modelDir: String, dispatchLibDir: String, refDir: String): String
    private external fun nativeRunKleinTimeInSmoke(modelDir: String, dispatchLibDir: String): String
    private external fun nativeRunKleinComponentBoundarySmoke(modelDir: String, dispatchLibDir: String): String
    private external fun nativeRunKleinVaeUp3Smoke(modelDir: String, dispatchLibDir: String): String
    private external fun nativeRunKleinTransformerSmoke(modelDir: String, dispatchLibDir: String): String
    private external fun nativeRunKleinOneStepReference(modelDir: String, dispatchLibDir: String, outputFile: String, stepIndex: Int): String
    private external fun nativeDecodeKleinReferenceLatent(modelDir: String, dispatchLibDir: String, outputFile: String): String
    private external fun nativeRunQwenTextEncoder(configPath: String, prompt: String, outputFile: String): String

    // THROWAWAY perf/correctness confirmation for the native (C++/LiteRT-C-API) KleinSingleBlockEngine
    // -- see jni_diffusion.cpp's Java_..._nativeRunKleinSingleBlockSmoke doc comment. Reuses the
    // klein_single0/ directory: it already has both the 4 compiled pieces (from
    // runKleinChunkedBlockProbe's own setup) and the real reference IO (x/pe/mod_*/torch_out.bin).
    fun runNativeKleinSingleBlockSmoke(context: Context) {
        try {
            System.loadLibrary("pockettavern_diffusion")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== native KleinSingleBlockEngine smoke test: FAILED to load pockettavern_diffusion ===", e)
            return
        }
        val dir = File(context.filesDir, "klein_single0")
        if (!dir.exists()) {
            Log.w(TAG, "$dir not found, skipping native KleinSingleBlockEngine smoke test")
            return
        }
        val dispatchLibDir = context.applicationInfo.nativeLibraryDir
        Log.i(TAG, "=== native KleinSingleBlockEngine smoke test starting (dir=$dir dispatchLibDir=$dispatchLibDir) ===")
        val result = nativeRunKleinSingleBlockSmoke(dir.absolutePath, dispatchLibDir, dir.absolutePath)
        Log.i(TAG, "NATIVE_KLEIN_SINGLE_BLOCK_SMOKE: $result")
    }

    fun runNativeKleinDoubleBlockSmoke(context: Context) {
        try {
            System.loadLibrary("pockettavern_diffusion")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== native KleinDoubleBlockEngine smoke test: FAILED to load pockettavern_diffusion ===", e)
            return
        }
        val dir = File(context.filesDir, "klein_single0")
        if (!dir.exists()) {
            Log.w(TAG, "$dir not found, skipping native KleinDoubleBlockEngine smoke test")
            return
        }
        val result = nativeRunKleinDoubleBlockSmoke(dir.absolutePath, context.applicationInfo.nativeLibraryDir, dir.absolutePath)
        Log.i(TAG, "NATIVE_KLEIN_DOUBLE_BLOCK_SMOKE: $result")
    }

    fun runNativeKleinZeroCopyQkvFlashProbe(context: Context) {
        try {
            System.loadLibrary("pockettavern_diffusion")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== native Klein zero-copy probe: FAILED to load pockettavern_diffusion ===", e)
            return
        }
        val dir = File(context.filesDir, "klein_single0")
        val result = nativeRunKleinZeroCopyQkvFlashProbe(dir.absolutePath, context.applicationInfo.nativeLibraryDir, dir.absolutePath)
        Log.i(TAG, "NATIVE_KLEIN_ZERO_COPY_QKV_FLASH: $result")
    }

    fun runNativeKleinTimeInSmoke(context: Context) {
        try {
            System.loadLibrary("pockettavern_diffusion")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== native Klein time_in smoke test: failed to load library ===", e)
            return
        }
        val dir = File(context.filesDir, "flux2_klein_npu")
        if (!File(dir, "time_in_Google_Tensor_G5.tflite").isFile) {
            Log.w(TAG, "time_in artifact missing from $dir, skipping native component smoke test")
            return
        }
        val result = nativeRunKleinTimeInSmoke(dir.absolutePath, context.applicationInfo.nativeLibraryDir)
        Log.i(TAG, "NATIVE_KLEIN_TIME_IN_SMOKE: $result")
    }

    fun runNativeKleinComponentBoundarySmoke(context: Context) {
        try {
            System.loadLibrary("pockettavern_diffusion")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== native Klein component-boundary smoke: failed to load library ===", e)
            return
        }
        val dir = File(context.filesDir, "flux2_klein_npu")
        val required = listOf("img_in_Google_Tensor_G5.tflite", "txt_in_Google_Tensor_G5.tflite", "vae_decoder_pre_mid_Google_Tensor_G5.tflite")
        if (required.any { !File(dir, it).isFile }) {
            Log.w(TAG, "Klein component artifacts missing from $dir, skipping boundary smoke test")
            return
        }
        val result = nativeRunKleinComponentBoundarySmoke(dir.absolutePath, context.applicationInfo.nativeLibraryDir)
        Log.i(TAG, "NATIVE_KLEIN_COMPONENT_BOUNDARY_SMOKE: $result")
    }

    fun runNativeKleinVaeUp3Smoke(context: Context) {
        try {
            System.loadLibrary("pockettavern_diffusion")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== native Klein VAE up_3 smoke: failed to load library ===", e)
            return
        }
        val dir = File(context.filesDir, "flux2_klein_npu")
        val required = listOf("vae_decoder_pre_mid_Google_Tensor_G5.tflite", "vae_decoder_up_3_Google_Tensor_G5.tflite")
        if (required.any { !File(dir, it).isFile }) {
            Log.w(TAG, "Klein VAE artifacts missing from $dir, skipping up_3 smoke test")
            return
        }
        val result = nativeRunKleinVaeUp3Smoke(dir.absolutePath, context.applicationInfo.nativeLibraryDir)
        Log.i(TAG, "NATIVE_KLEIN_VAE_UP3_SMOKE: $result")
    }

    fun runNativeKleinTransformerSmoke(context: Context) {
        try {
            System.loadLibrary("pockettavern_diffusion")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== native Klein full-transformer smoke: failed to load library ===", e)
            return
        }
        val dir = File(context.filesDir, "flux2_klein_npu")
        if (File(dir, "qkv_proj_probe_1152_Google_Tensor_G5.tflite").isFile.not()) {
            Log.w(TAG, "Klein transformer artifacts missing from $dir, skipping full-transformer smoke")
            return
        }
        val result = nativeRunKleinTransformerSmoke(dir.absolutePath, context.applicationInfo.nativeLibraryDir)
        Log.i(TAG, "NATIVE_KLEIN_TRANSFORMER_SMOKE: $result")
    }

    fun runNativeKleinOneStepReference(context: Context) {
        try { System.loadLibrary("pockettavern_diffusion") } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== Klein one-step reference: failed to load library ===", e); return
        }
        val dir = File(context.filesDir, "flux2_klein_npu")
        if (!File(dir, "context.bin").isFile) { Log.w(TAG, "context.bin missing from $dir, skipping one-step reference"); return }
        val result = nativeRunKleinOneStepReference(dir.absolutePath, context.applicationInfo.nativeLibraryDir,
            File(dir, "one_step_latent.bin").absolutePath, 0)
        Log.i(TAG, "NATIVE_KLEIN_ONE_STEP_REFERENCE: $result")
    }

    fun runNativeKleinRemainingReferenceSteps(context: Context) {
        try { System.loadLibrary("pockettavern_diffusion") } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== Klein remaining reference steps: failed to load library ===", e); return
        }
        val dir = File(context.filesDir, "flux2_klein_npu")
        val latent = File(dir, "one_step_latent.bin")
        if (!File(dir, "context.bin").isFile || !latent.isFile) { Log.w(TAG, "reference context or step-0 latent missing, skipping remaining steps"); return }
        for (step in 1..3) {
            val result = nativeRunKleinOneStepReference(dir.absolutePath, context.applicationInfo.nativeLibraryDir, latent.absolutePath, step)
            Log.i(TAG, "NATIVE_KLEIN_REFERENCE_STEP_$step: $result")
            if (!result.startsWith("OK")) break
        }
    }

    fun decodeKleinReferenceLatent(context: Context) {
        try { System.loadLibrary("pockettavern_diffusion") } catch (e: UnsatisfiedLinkError) { Log.e(TAG, "Klein VAE decode library load failed", e); return }
        val dir = File(context.filesDir, "flux2_klein_npu")
        val result = nativeDecodeKleinReferenceLatent(dir.absolutePath, context.applicationInfo.nativeLibraryDir, File(dir, "reference.ppm").absolutePath)
        Log.i(TAG, "NATIVE_KLEIN_VAE_DECODE: $result")
    }

    // FLUX.2 [klein] Phase 1 (docs/FLUX2_KLEIN_PHASE1_TEXT_ENCODER_PLAN.md): runs the on-device
    // Qwen3-4B text encoder for an arbitrary prompt and dumps the [1,512,7680] hidden-state tensor
    // to context_qwen.bin, to be pulled and diffed against the PyTorch reference for that prompt.
    // encoderDir must contain the MNN-exported encoder (config.json, llm.mnn, llm.mnn.weight,
    // tokenizer.mtok -- staged the same way the Klein NPU artifacts are staged).
    fun runNativeQwenTextEncoder(context: Context, prompt: String) {
        try { System.loadLibrary("pockettavern_diffusion") } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== Qwen text encoder: failed to load library ===", e); return
        }
        val dir = File(context.filesDir, "flux2_klein_qwen")
        val configPath = File(dir, "config.json")
        if (!configPath.isFile) { Log.w(TAG, "config.json missing from $dir, skipping Qwen text encoder"); return }
        val result = nativeRunQwenTextEncoder(configPath.absolutePath, prompt, File(dir, "context_qwen.bin").absolutePath)
        Log.i(TAG, "NATIVE_QWEN_TEXT_ENCODER: $result")
    }

    fun runEndToEndSdxlNpuGenerationTest(context: Context) {
        try {
            System.loadLibrary("pockettavern_diffusion")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== E2E SDXL NPU test: FAILED to load pockettavern_diffusion ===", e)
            return
        }
        val modelPath = File(context.filesDir, "sd-models/pureTukanoNSFW-xl")
        val npuPath = File(context.filesDir, "npu-unet/pureTukanoNSFW-xl")
        val dispatchLibDir = context.applicationInfo.nativeLibraryDir
        Log.i(TAG, "=== E2E SDXL NPU test starting (modelPath=$modelPath npuPath=$npuPath) ===")

        val handle = com.pockettavern.app.data.local.inference.MnnDiffusionBridge.nativeCreate(
            modelPath.absolutePath,
            4, // STABLE_DIFFUSION_XL
            2, // MEMORY_MODE_FAST
            npuPath.absolutePath,
            dispatchLibDir
        )
        if (handle == 0L) {
            Log.e(TAG, "E2E SDXL NPU test: nativeCreate returned 0")
            return
        }
        if (!com.pockettavern.app.data.local.inference.MnnDiffusionBridge.nativeLoad(handle)) {
            Log.e(TAG, "E2E SDXL NPU test: nativeLoad failed")
            com.pockettavern.app.data.local.inference.MnnDiffusionBridge.nativeDestroy(handle)
            return
        }
        Log.i(TAG, "E2E SDXL NPU test: nativeLoad succeeded!")

        val outFile = File(context.filesDir, "test_npu_sdxl_out.png")
        if (outFile.exists()) outFile.delete()

        val ok = com.pockettavern.app.data.local.inference.MnnDiffusionBridge.nativeGenerateXL(
            handle,
            prompt = "a serene zen garden with blooming cherry blossom trees, photorealistic",
            negativePrompt = "blurry, low quality",
            outputPath = outFile.absolutePath,
            width = 1024,
            height = 1024,
            steps = 1,
            seed = 42,
            cfgScale = 5.0f,
            progressCallback = { pct -> Log.i(TAG, "E2E SDXL NPU Progress: $pct%") }
        )
        com.pockettavern.app.data.local.inference.MnnDiffusionBridge.nativeDestroy(handle)
        Log.i(TAG, "E2E_SDXL_NPU_RESULT: ok=$ok outFileExists=${outFile.exists()} size=${outFile.length()}")
    }

    // THROWAWAY: validates the batch=2 (real CFG) piece set -- see jni_diffusion.cpp's
    // Java_..._nativeRunUnetEngineBatch2Smoke doc comment for what this actually checks.
    private external fun nativeRunUnetEngineBatch2Smoke(modelDir: String, dispatchLibDir: String): String

    fun runNativeUnetEngineBatch2Smoke(context: Context) {
        try {
            System.loadLibrary("pockettavern_diffusion")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== native UnetEngine batch=2 smoke test: FAILED to load pockettavern_diffusion ===", e)
            return
        }
        val modelDir = File(context.filesDir, "npu-unet-b2-test")
        if (!modelDir.exists()) {
            Log.w(TAG, "$modelDir not found, skipping native UnetEngine batch=2 smoke test")
            return
        }
        val dispatchLibDir = context.applicationInfo.nativeLibraryDir
        Log.i(TAG, "=== native UnetEngine batch=2 smoke test starting (modelDir=$modelDir dispatchLibDir=$dispatchLibDir) ===")
        val result = nativeRunUnetEngineBatch2Smoke(modelDir.absolutePath, dispatchLibDir)
        Log.i(TAG, "NATIVE_UNET_ENGINE_BATCH2_SMOKE: $result")
    }

    // THROWAWAY: measures whether two independent batch-1 NPU forwards overlap when started
    // concurrently. This is the only potentially practical CFG alternative to the much slower
    // batch-2 Tensor G5 kernels.
    private external fun nativeRunTwoBatch1EnginesInParallel(modelDir: String, dispatchLibDir: String): String

    fun runTwoBatch1EnginesInParallel(context: Context) {
        try {
            System.loadLibrary("pockettavern_diffusion")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== parallel batch-1 smoke test: FAILED to load pockettavern_diffusion ===", e)
            return
        }
        val modelDir = File(context.filesDir, "npu-unet/pureTukanoNSFW-xl")
        if (!modelDir.exists()) {
            Log.w(TAG, "$modelDir not found, skipping parallel batch-1 smoke test")
            return
        }
        val result = nativeRunTwoBatch1EnginesInParallel(
            modelDir.absolutePath,
            context.applicationInfo.nativeLibraryDir,
        )
        Log.i(TAG, "NATIVE_UNET_ENGINE_PARALLEL_BATCH1_SMOKE: $result")
    }

    private external fun nativeRunParallelBatch1FileStep(
        modelDir: String,
        dispatchLibDir: String,
        inputDir: String,
        outputFile: String,
    ): String

    /**
     * Debug-only one-step entry point for the desktop parallel-CFG driver. Its input directory
     * contains `uncond_*.bin` and `cond_*.bin` tensors; native C++ runs the two batch-1 forwards
     * concurrently and writes their concatenated outputs. The desktop caller applies CFG and the
     * scheduler, so this is a real image-generation path rather than synthetic test data.
     */
    fun runParallelBatch1FileStep(context: Context) {
        try {
            System.loadLibrary("pockettavern_diffusion")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "=== parallel batch-1 file step: FAILED to load pockettavern_diffusion ===", e)
            return
        }
        val inputDir = File(context.filesDir, "unet_step_in")
        val outputFile = File(context.filesDir, "unet_step_out.bin")
        val modelDir = File(context.filesDir, "npu-unet/pureTukanoNSFW-xl")
        if (!inputDir.isDirectory || !modelDir.isDirectory) {
            Log.w(TAG, "parallel batch-1 file step inputs or model bundle missing, skipping")
            return
        }
        val result = nativeRunParallelBatch1FileStep(
            modelDir.absolutePath,
            context.applicationInfo.nativeLibraryDir,
            inputDir.absolutePath,
            outputFile.absolutePath,
        )
        Log.i(TAG, "PARALLEL_BATCH1_FILE_STEP: $result")
    }

    private fun currentRssKb(): Long {
        return try {
            File("/proc/self/status").readLines()
                .firstOrNull { it.startsWith("VmRSS:") }
                ?.let { line -> line.substringAfter("VmRSS:").trim().split(Regex("\\s+"))[0].toLong() }
                ?: -1L
        } catch (e: Throwable) {
            -1L
        }
    }

    /** Runs [block] while polling this process's VmRSS every 15ms; returns (result, peakRssKb). */
    private fun <T> withPeakRss(block: () -> T): Pair<T, Long> {
        val peakKb = java.util.concurrent.atomic.AtomicLong(currentRssKb())
        val running = java.util.concurrent.atomic.AtomicBoolean(true)
        val sampler = Thread {
            while (running.get()) {
                val kb = currentRssKb()
                peakKb.updateAndGet { prev -> if (kb > prev) kb else prev }
                Thread.sleep(15)
            }
        }
        sampler.start()
        try {
            val result = block()
            return result to peakKb.get()
        } finally {
            running.set(false)
            sampler.join()
        }
    }

    private val PROMPT_TOKENS = intArrayOf(
        49406, 320, 8853, 539, 550, 18376, 6765, 320, 4558, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
    )

    private fun runOnce(
        modelPath: String,
        accelerator: Accelerator,
        env: Environment?,
    ): Pair<FloatArray, Long> {
        val options = if (env != null) CompiledModel.Options(accelerator) else CompiledModel.Options.CPU
        val model = CompiledModel.create(modelPath, options, env)
        val inputs = model.createInputBuffers()
        val outputs = model.createOutputBuffers()
        inputs[0].writeInt(PROMPT_TOKENS)

        val start = System.currentTimeMillis()
        model.run(inputs, outputs)
        val elapsedMs = System.currentTimeMillis() - start

        val result = outputs[0].readFloat()
        model.close()
        return result to elapsedMs
    }

    fun run(context: Context) {
        val npuModelFile = File(context.filesDir, "clip_npu.tflite")
        val cpuModelFile = File(context.filesDir, "clip_cpu.tflite")
        if (!npuModelFile.exists() || !cpuModelFile.exists()) {
            Log.w(TAG, "clip_npu.tflite / clip_cpu.tflite not found in ${context.filesDir}, skipping")
            return
        }

        Log.i(TAG, "=== NPU vs CPU correctness diagnostic starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)

            val (npuOut, npuMs) = runOnce(npuModelFile.absolutePath, Accelerator.NPU, env)
            Log.i(TAG, "NPU: ${npuMs}ms, output[0..4]=${npuOut.take(5)}")

            val (cpuOut, cpuMs) = runOnce(cpuModelFile.absolutePath, Accelerator.CPU, null)
            Log.i(TAG, "CPU: ${cpuMs}ms, output[0..4]=${cpuOut.take(5)}")

            var maxAbsDiff = 0f
            var sumAbsDiff = 0.0
            for (i in npuOut.indices) {
                val diff = kotlin.math.abs(npuOut[i] - cpuOut[i])
                if (diff > maxAbsDiff) maxAbsDiff = diff
                sumAbsDiff += diff
            }
            val meanAbsDiff = sumAbsDiff / npuOut.size
            Log.i(TAG, "=== NPU vs CPU diff: maxAbsDiff=$maxAbsDiff meanAbsDiff=$meanAbsDiff (expected ref output[0,0,:5]=[-0.3758, 0.0174, -0.0612, -0.1855, -0.0404]) ===")
        } catch (e: Throwable) {
            Log.e(TAG, "=== NPU vs CPU correctness diagnostic: FAILED ===", e)
        }
    }

    /**
     * Does the merged mid_block model (4 chained DISPATCH_OP pieces -- resnet0, attn[0:5],
     * attn[5:10], resnet1 -- spliced into one FlatBuffer, see
     * ~/code/litert-torch/scratch/merge_compiled_pieces.py) actually resolve and run its NPU
     * dispatch ops on real Tensor G5 hardware, and how fast? Inputs are arbitrary non-zero
     * values -- this checks that dispatch resolves and produces finite output, not numeric
     * correctness against a PyTorch reference (that needs a separate validation pass).
     */
    private fun runMidOnce(
        modelPath: String,
        env: Environment,
        sample: FloatArray = FloatArray(1 * 1280 * 32 * 32) { i -> kotlin.math.sin(i.toFloat()) * 0.1f },
        emb: FloatArray = FloatArray(1 * 1280) { i -> kotlin.math.sin(i.toFloat()) * 0.1f },
        encoderHiddenStates: FloatArray = FloatArray(1 * 77 * 2048) { i -> kotlin.math.sin(i.toFloat()) * 0.1f },
    ): Pair<FloatArray, Long> {
        val options = CompiledModel.Options(Accelerator.NPU)
        val model = CompiledModel.create(modelPath, options, env)
        val inputs = model.createInputBuffers()
        val outputs = model.createOutputBuffers()

        inputs[0].writeFloat(sample)
        inputs[1].writeFloat(emb)
        inputs[2].writeFloat(encoderHiddenStates)

        val start = System.currentTimeMillis()
        model.run(inputs, outputs)
        val elapsedMs = System.currentTimeMillis() - start

        val result = outputs[0].readFloat()
        model.close()
        return result to elapsedMs
    }

    /**
     * Like [runMid], but holds the CompiledModel open (created + run, not yet closed) for a long
     * pause so an external `adb shell dumpsys meminfo` snapshot can be taken of both this
     * process and the EdgeTPU HAL service (vendor.google.edgetpu_app_service) while the model is
     * actually resident/dispatched -- to check whether the compiled model's dma-buf is SHARED
     * (same buffer mapped into both processes, expected Graphics/dmabuf RSS appearing in BOTH
     * dumpsys outputs at matching sizes) or independently duplicated (would show up as ordinary
     * unrelated memory growth with no matching cross-process correlation).
     */
    fun runMidWithPause(context: Context, pauseMs: Long = 25000) {
        val modelFile = File(context.filesDir, "mid_merged.tflite")
        if (!modelFile.exists()) {
            Log.w(TAG, "mid_merged.tflite not found in ${context.filesDir}, skipping")
            return
        }
        Log.i(TAG, "=== mid_block with pause diagnostic starting (for external memory inspection) ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)
            val model = CompiledModel.create(modelFile.absolutePath, options, env)
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()
            val sample = FloatArray(1 * 1280 * 32 * 32) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val emb = FloatArray(1 * 1280) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val ehs = FloatArray(1 * 77 * 2048) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            inputs[0].writeFloat(sample)
            inputs[1].writeFloat(emb)
            inputs[2].writeFloat(ehs)
            model.run(inputs, outputs)
            val result = outputs[0].readFloat()
            Log.i(TAG, "runMidWithPause: run complete, output[0..4]=${result.take(5)}, PAUSING ${pauseMs}ms for external inspection NOW")
            Thread.sleep(pauseMs)
            Log.i(TAG, "runMidWithPause: pause complete, closing")
            model.close()
        } catch (e: Throwable) {
            Log.e(TAG, "=== mid_block with pause diagnostic: FAILED ===", e)
        }
    }

    fun runMid(context: Context) {
        val modelFile = File(context.filesDir, "mid_merged.tflite")
        if (!modelFile.exists()) {
            Log.w(TAG, "mid_merged.tflite not found in ${context.filesDir}, skipping")
            return
        }

        Log.i(TAG, "=== mid_block NPU dispatch diagnostic (merged single file) starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)

            val (runResult, peakRssKb) = withPeakRss { runMidOnce(modelFile.absolutePath, env) }
            val (out, ms) = runResult
            var hasNanOrInf = false
            var maxAbs = 0f
            for (v in out) {
                if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                val a = kotlin.math.abs(v)
                if (a > maxAbs) maxAbs = a
            }
            Log.i(
                TAG,
                "mid_block NPU (merged): ${ms}ms, peakRSS=${peakRssKb / 1024}MB, " +
                    "outputSize=${out.size}, hasNanOrInf=$hasNanOrInf, maxAbs=$maxAbs, " +
                    "output[0..4]=${out.take(5)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== mid_block NPU dispatch diagnostic (merged): FAILED ===", e)
        }
    }

    /**
     * Runs mid_block both as one merged CompiledModel (runMid's path) and as 4 separate
     * CompiledModel instances chained via host-side readFloat()/writeFloat() (runMidSeparate's
     * path, same default sin() inputs, real resnet0->attn0->attn1->resnet1 topology), then diffs
     * the two final output arrays element-by-element. Tests whether forcing a host round-trip
     * between each piece's run() and the next piece's input (rather than any kind of pipelined
     * dispatch) produces output numerically consistent with the known-correct single-graph
     * result, or whether a same async/ordering issue shows up even with that round-trip in place.
     */
    fun compareMidMergedVsSeparate(context: Context) {
        val dir = context.filesDir
        val mergedFile = File(dir, "mid_merged.tflite")
        val pieceFiles = listOf("resnet0", "attn0", "attn1", "resnet1").map { File(dir, "mid_$it.tflite") }
        if (!mergedFile.exists() || pieceFiles.any { !it.exists() }) {
            Log.w(TAG, "mid_merged.tflite / mid_{resnet0,attn0,attn1,resnet1}.tflite not all found in $dir, skipping")
            return
        }
        Log.i(TAG, "=== mid_block merged vs separate-chained comparison starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val (mergedOut, mergedMs) = runMidOnce(mergedFile.absolutePath, env)

            fun runPiece(path: String, vararg inputArrays: FloatArray): List<FloatArray> {
                val model = CompiledModel.create(path, options, env)
                val inputs = model.createInputBuffers()
                val outputs = model.createOutputBuffers()
                inputArrays.forEachIndexed { i, arr -> inputs[i].writeFloat(arr) }
                model.run(inputs, outputs)
                val results = outputs.map { it.readFloat() }
                model.close()
                return results
            }

            val sepStart = System.currentTimeMillis()
            val sample = FloatArray(1 * 1280 * 32 * 32) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val emb = FloatArray(1 * 1280) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val encoderHiddenStates = FloatArray(1 * 77 * 2048) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val resnet0Out = runPiece(pieceFiles[0].absolutePath, sample, emb)[0]
            val attn0Out = runPiece(pieceFiles[1].absolutePath, resnet0Out, encoderHiddenStates)
            val hiddenStatesSeq = attn0Out[0]
            val residual = resnet0Out
            val attn1Out = runPiece(pieceFiles[2].absolutePath, hiddenStatesSeq, residual, encoderHiddenStates)[0]
            val separateOut = runPiece(pieceFiles[3].absolutePath, attn1Out, emb)[0]
            val separateMs = System.currentTimeMillis() - sepStart

            var maxAbsDiff = 0.0
            var sumAbsDiff = 0.0
            var mismatchIdx = -1
            for (j in mergedOut.indices) {
                val diff = kotlin.math.abs((mergedOut[j] - separateOut[j]).toDouble())
                if (diff > maxAbsDiff) {
                    maxAbsDiff = diff
                    mismatchIdx = j
                }
                sumAbsDiff += diff
            }
            val meanAbsDiff = sumAbsDiff / mergedOut.size

            Log.i(
                TAG,
                "COMPARE: mergedMs=$mergedMs separateMs=$separateMs mergedSize=${mergedOut.size} " +
                    "separateSize=${separateOut.size} maxAbsDiff=$maxAbsDiff meanAbsDiff=$meanAbsDiff " +
                    "mismatchIdx=$mismatchIdx merged[mismatchIdx]=${if (mismatchIdx >= 0) mergedOut[mismatchIdx] else 0f} " +
                    "separate[mismatchIdx]=${if (mismatchIdx >= 0) separateOut[mismatchIdx] else 0f} " +
                    "merged[0..2]=${mergedOut.take(3)} separate[0..2]=${separateOut.take(3)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== mid_block merged vs separate-chained comparison: FAILED ===", e)
        }
    }

    /**
     * Isolates the resnet0->attn0 boundary: runs mid_r0_a0_merged.tflite (a 2-op merge of just
     * those pieces, built by merge_mid_r0_a0.py) vs the same 2 pieces as separate CompiledModel
     * instances chained via host-side readFloat()/writeFloat(), and diffs attn0's real computed
     * output (hidden_states_seq). Bisects whether the large (~24 maxAbsDiff) divergence seen in
     * [compareMidMergedVsSeparate]'s full 4-piece chain is already present after just 2 pieces,
     * or only appears with attn1/resnet1 added.
     */
    fun compareR0Attn0(context: Context) {
        val dir = context.filesDir
        val mergedFile = File(dir, "mid_r0_a0_merged.tflite")
        val resnetFile = File(dir, "mid_resnet0.tflite")
        val attnFile = File(dir, "mid_attn0.tflite")
        if (!mergedFile.exists() || !resnetFile.exists() || !attnFile.exists()) {
            Log.w(TAG, "mid_r0_a0_merged.tflite / mid_resnet0.tflite / mid_attn0.tflite not all found in $dir, skipping")
            return
        }
        Log.i(TAG, "=== resnet0->attn0 boundary isolation comparison starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val sample = FloatArray(1 * 1280 * 32 * 32) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val emb = FloatArray(1 * 1280) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val encoderHiddenStates = FloatArray(1 * 77 * 2048) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }

            fun runPiece(path: String, vararg inputArrays: FloatArray): List<FloatArray> {
                val model = CompiledModel.create(path, options, env)
                val inputs = model.createInputBuffers()
                val outputs = model.createOutputBuffers()
                inputArrays.forEachIndexed { i, arr -> inputs[i].writeFloat(arr) }
                model.run(inputs, outputs)
                val results = outputs.map { it.readFloat() }
                model.close()
                return results
            }

            // merged: single 2-op DISPATCH_OP chain, external inputs sample/emb/ehs ->
            // [resnet0_out, hidden_states_seq] (extra_outputs exposes resnet0's own output too,
            // so we can check whether resnet0 itself already diverges before blaming attn0)
            val mergedResults = runPiece(mergedFile.absolutePath, sample, emb, encoderHiddenStates)
            val mergedR0Out = mergedResults[0]
            val mergedHidden = mergedResults[1]

            // Repeat-consistency check: is attn0 itself deterministic (same context, same
            // input, same output every time), or does it vary even run-to-run within the SAME
            // execution context? Distinguishes "context-dependent but repeatable" from "just
            // flaky/racy internally" before concluding anything about WHY merged != standalone.
            val mergedResults2 = runPiece(mergedFile.absolutePath, sample, emb, encoderHiddenStates)
            val mergedHidden2 = mergedResults2[1]

            // separate: 2 standalone instances, resnet0's real output fed into attn0's input
            val resnet0Out = runPiece(resnetFile.absolutePath, sample, emb)[0]
            val separateHidden = runPiece(attnFile.absolutePath, resnet0Out, encoderHiddenStates)[0]
            val separateHidden2 = runPiece(attnFile.absolutePath, resnet0Out, encoderHiddenStates)[0]

            fun diff(a: FloatArray, b: FloatArray): Triple<Double, Double, Int> {
                var maxAbsDiff = 0.0
                var sumAbsDiff = 0.0
                var mismatchIdx = -1
                for (j in a.indices) {
                    val d = kotlin.math.abs((a[j] - b[j]).toDouble())
                    if (d > maxAbsDiff) {
                        maxAbsDiff = d
                        mismatchIdx = j
                    }
                    sumAbsDiff += d
                }
                return Triple(maxAbsDiff, sumAbsDiff / a.size, mismatchIdx)
            }

            val (r0MaxDiff, r0MeanDiff, r0MismatchIdx) = diff(mergedR0Out, resnet0Out)
            val (hMaxDiff, hMeanDiff, hMismatchIdx) = diff(mergedHidden, separateHidden)
            val (mergedSelfMaxDiff, _, _) = diff(mergedHidden, mergedHidden2)
            val (separateSelfMaxDiff, _, _) = diff(separateHidden, separateHidden2)

            Log.i(
                TAG,
                "COMPARE_R0A0_RESNET0: maxAbsDiff=$r0MaxDiff meanAbsDiff=$r0MeanDiff mismatchIdx=$r0MismatchIdx " +
                    "merged[0..2]=${mergedR0Out.take(3)} standalone[0..2]=${resnet0Out.take(3)}",
            )
            Log.i(
                TAG,
                "COMPARE_R0A0_HIDDEN: maxAbsDiff=$hMaxDiff meanAbsDiff=$hMeanDiff mismatchIdx=$hMismatchIdx " +
                    "merged[0..2]=${mergedHidden.take(3)} separate[0..2]=${separateHidden.take(3)}",
            )
            Log.i(
                TAG,
                "COMPARE_R0A0_REPEAT: mergedSelfMaxDiff=$mergedSelfMaxDiff (merged run1 vs run2) " +
                    "separateSelfMaxDiff=$separateSelfMaxDiff (standalone run1 vs run2)",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== resnet0->attn0 boundary isolation comparison: FAILED ===", e)
        }
    }

    /**
     * Tests whether attn0's merged-vs-standalone divergence (see [compareR0Attn0]) is a pure
     * SESSION-POSITION effect (attn0 dispatched as op #2 of any CompiledModel session) rather
     * than something depending on resnet0's actual computed values. Runs
     * mid_dummy_then_attn0_merged.tflite -- a 2-op merge where op #1 is resnet0 with its output
     * wired to NOTHING (a functionally disconnected dummy predecessor) and op #2 is attn0 wired
     * directly to fresh external inputs (real resnet0 output supplied externally, not chained
     * from op #1). If attn0's output here matches the TRUE merged (real resnet0->attn0 chain)
     * result, session position alone is sufficient. If it matches the standalone-alone result
     * instead, position isn't enough -- something about op #1 actually computing on real/
     * matching data matters.
     */
    fun compareDummyThenAttn0(context: Context) {
        val dir = context.filesDir
        val dummyFile = File(dir, "mid_dummy_then_attn0_merged.tflite")
        val trueMergedFile = File(dir, "mid_r0_a0_merged.tflite")
        val resnetFile = File(dir, "mid_resnet0.tflite")
        val attnFile = File(dir, "mid_attn0.tflite")
        if (!dummyFile.exists() || !trueMergedFile.exists() || !resnetFile.exists() || !attnFile.exists()) {
            Log.w(TAG, "required mid_*.tflite files not all found in $dir, skipping")
            return
        }
        Log.i(TAG, "=== dummy-op1-then-attn0 session-position isolation test starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val sample = FloatArray(1 * 1280 * 32 * 32) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val emb = FloatArray(1 * 1280) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val encoderHiddenStates = FloatArray(1 * 77 * 2048) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }

            fun runPiece(path: String, vararg inputArrays: FloatArray): List<FloatArray> {
                val model = CompiledModel.create(path, options, env)
                val inputs = model.createInputBuffers()
                val outputs = model.createOutputBuffers()
                inputArrays.forEachIndexed { i, arr -> inputs[i].writeFloat(arr) }
                model.run(inputs, outputs)
                val results = outputs.map { it.readFloat() }
                model.close()
                return results
            }

            // ground truth: real chained resnet0->attn0 merged result
            val trueHidden = runPiece(trueMergedFile.absolutePath, sample, emb, encoderHiddenStates)[1]

            // standalone-alone: attn0 as the sole op in its own session (already known-wrong)
            val resnet0Out = runPiece(resnetFile.absolutePath, sample, emb)[0]
            val standaloneHidden = runPiece(attnFile.absolutePath, resnet0Out, encoderHiddenStates)[0]

            // dummy-op1-then-attn0: op #1 (resnet0) disconnected, attn0's real inputs external
            // signature order: dummy_sample, dummy_emb, real_r0_out, ehs -> [resnet0_out_unused, hidden_states_seq, real_r0_out_pass]
            val dummyResults = runPiece(dummyFile.absolutePath, sample, emb, resnet0Out, encoderHiddenStates)
            val dummyHidden = dummyResults[1]

            fun diff(a: FloatArray, b: FloatArray): Double {
                var maxAbsDiff = 0.0
                for (j in a.indices) {
                    val d = kotlin.math.abs((a[j] - b[j]).toDouble())
                    if (d > maxAbsDiff) maxAbsDiff = d
                }
                return maxAbsDiff
            }

            val vsTrue = diff(dummyHidden, trueHidden)
            val vsStandalone = diff(dummyHidden, standaloneHidden)

            Log.i(
                TAG,
                "DUMMY_THEN_ATTN0: maxAbsDiffVsTrueMerged=$vsTrue maxAbsDiffVsStandaloneAlone=$vsStandalone " +
                    "true[0..2]=${trueHidden.take(3)} standalone[0..2]=${standaloneHidden.take(3)} dummy[0..2]=${dummyHidden.take(3)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== dummy-op1-then-attn0 session-position isolation test: FAILED ===", e)
        }
    }

    /**
     * Tests the cheap fix implied by [compareDummyThenAttn0]'s finding (provenance, not session
     * position, is what matters): does a real but trivial passthrough op #1 -- RESHAPE(real_r0_out)
     * to the SAME shape, making the tensor graph-internal via a genuine op node instead of an
     * app-written external buffer -- reproduce the TRUE merged result, without needing a whole
     * extra real NPU dispatch as a dummy predecessor? Runs mid_reshape_then_attn0.tflite (built by
     * build_reshape_then_attn0.py: RESHAPE -> attn0's real DISPATCH_OP, single CompiledModel).
     */
    fun compareReshapeThenAttn0(context: Context) {
        val dir = context.filesDir
        val reshapeFile = File(dir, "mid_reshape_then_attn0.tflite")
        val trueMergedFile = File(dir, "mid_r0_a0_merged.tflite")
        val resnetFile = File(dir, "mid_resnet0.tflite")
        val attnFile = File(dir, "mid_attn0.tflite")
        if (!reshapeFile.exists() || !trueMergedFile.exists() || !resnetFile.exists() || !attnFile.exists()) {
            Log.w(TAG, "required mid_*.tflite files not all found in $dir, skipping")
            return
        }
        Log.i(TAG, "=== reshape-then-attn0 provenance-fix test starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val sample = FloatArray(1 * 1280 * 32 * 32) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val emb = FloatArray(1 * 1280) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val encoderHiddenStates = FloatArray(1 * 77 * 2048) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }

            fun runPiece(path: String, vararg inputArrays: FloatArray): List<FloatArray> {
                val model = CompiledModel.create(path, options, env)
                val inputs = model.createInputBuffers()
                val outputs = model.createOutputBuffers()
                inputArrays.forEachIndexed { i, arr -> inputs[i].writeFloat(arr) }
                model.run(inputs, outputs)
                val results = outputs.map { it.readFloat() }
                model.close()
                return results
            }

            val trueHidden = runPiece(trueMergedFile.absolutePath, sample, emb, encoderHiddenStates)[1]

            val resnet0Out = runPiece(resnetFile.absolutePath, sample, emb)[0]
            val standaloneHidden = runPiece(attnFile.absolutePath, resnet0Out, encoderHiddenStates)[0]

            // reshape-then-attn0 signature: args_0=real_r0_out_ext, args_1=ehs_ext -> output_0=hidden_states_seq
            val reshapeHidden = runPiece(reshapeFile.absolutePath, resnet0Out, encoderHiddenStates)[0]

            fun diff(a: FloatArray, b: FloatArray): Double {
                var maxAbsDiff = 0.0
                for (j in a.indices) {
                    val d = kotlin.math.abs((a[j] - b[j]).toDouble())
                    if (d > maxAbsDiff) maxAbsDiff = d
                }
                return maxAbsDiff
            }

            val vsTrue = diff(reshapeHidden, trueHidden)
            val vsStandalone = diff(reshapeHidden, standaloneHidden)

            Log.i(
                TAG,
                "RESHAPE_THEN_ATTN0: maxAbsDiffVsTrueMerged=$vsTrue maxAbsDiffVsStandaloneAlone=$vsStandalone " +
                    "true[0..2]=${trueHidden.take(3)} standalone[0..2]=${standaloneHidden.take(3)} reshape[0..2]=${reshapeHidden.take(3)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== reshape-then-attn0 provenance-fix test: FAILED ===", e)
        }
    }

    /**
     * Generalizes the RESHAPE-passthrough fix ([compareReshapeThenAttn0]) to the FULL mid_block
     * chain: resnet0 (unwrapped, already proven bit-exact standalone) -> attn0_wrapped ->
     * attn1_wrapped -> resnet1_wrapped (each *_wrapped piece has every one of its own inputs
     * routed through a same-shape RESHAPE first, built by build_reshape_wrapped_piece.py), each a
     * SEPARATE CompiledModel instance. Diffs the final output against the TRUE merged
     * (mid_merged.tflite, single 4-op session) result -- the same comparison
     * [compareMidMergedVsSeparate] did unwrapped (which found maxAbsDiff=23.9). If this drops to
     * ~0, the RESHAPE-wrap fix generalizes across the whole chain, not just the one boundary.
     */
    fun compareMidMergedVsReshapeWrapped(context: Context) {
        val dir = context.filesDir
        val mergedFile = File(dir, "mid_merged.tflite")
        val resnetFile = File(dir, "mid_resnet0.tflite")
        val attn0File = File(dir, "mid_attn0_wrapped.tflite")
        val attn1File = File(dir, "mid_attn1_wrapped.tflite")
        val resnet1File = File(dir, "mid_resnet1_wrapped.tflite")
        if (!mergedFile.exists() || !resnetFile.exists() || !attn0File.exists() ||
            !attn1File.exists() || !resnet1File.exists()
        ) {
            Log.w(TAG, "required mid_*.tflite files not all found in $dir, skipping")
            return
        }
        Log.i(TAG, "=== mid_block merged vs RESHAPE-wrapped-separate comparison starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val (mergedOut, mergedMs) = runMidOnce(mergedFile.absolutePath, env)

            fun runPiece(path: String, vararg inputArrays: FloatArray): List<FloatArray> {
                val model = CompiledModel.create(path, options, env)
                val inputs = model.createInputBuffers()
                val outputs = model.createOutputBuffers()
                inputArrays.forEachIndexed { i, arr -> inputs[i].writeFloat(arr) }
                model.run(inputs, outputs)
                val results = outputs.map { it.readFloat() }
                model.close()
                return results
            }

            val sepStart = System.currentTimeMillis()
            val sample = FloatArray(1 * 1280 * 32 * 32) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val emb = FloatArray(1 * 1280) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val encoderHiddenStates = FloatArray(1 * 77 * 2048) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val resnet0Out = runPiece(resnetFile.absolutePath, sample, emb)[0]
            val hiddenStatesSeq = runPiece(attn0File.absolutePath, resnet0Out, encoderHiddenStates)[0]
            val residual = resnet0Out
            val attn1Out = runPiece(attn1File.absolutePath, hiddenStatesSeq, residual, encoderHiddenStates)[0]
            val separateOut = runPiece(resnet1File.absolutePath, attn1Out, emb)[0]
            val separateMs = System.currentTimeMillis() - sepStart

            var maxAbsDiff = 0.0
            var sumAbsDiff = 0.0
            var mismatchIdx = -1
            for (j in mergedOut.indices) {
                val diff = kotlin.math.abs((mergedOut[j] - separateOut[j]).toDouble())
                if (diff > maxAbsDiff) {
                    maxAbsDiff = diff
                    mismatchIdx = j
                }
                sumAbsDiff += diff
            }
            val meanAbsDiff = sumAbsDiff / mergedOut.size

            Log.i(
                TAG,
                "COMPARE_WRAPPED: mergedMs=$mergedMs separateMs=$separateMs maxAbsDiff=$maxAbsDiff " +
                    "meanAbsDiff=$meanAbsDiff mismatchIdx=$mismatchIdx " +
                    "merged[0..2]=${mergedOut.take(3)} separate[0..2]=${separateOut.take(3)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== mid_block merged vs RESHAPE-wrapped-separate comparison: FAILED ===", e)
        }
    }

    /**
     * Same computation as [runMid], but as 4 SEPARATE CompiledModel instances -- create, run,
     * close -- one at a time, manually threading each piece's output into the next piece's
     * input. Compares peak RSS against the merged single-file approach: does loading one NPU
     * binary at a time (instead of all 4 resident via one CompiledModel) actually reduce peak
     * memory?
     */
    /**
     * Runs mid_block's 4 pieces STANDALONE, one at a time (create -> run -> pause -> close),
     * with a gap between each -- to measure each piece's own unmapped dma-buf delta via external
     * `adb shell dumpsys meminfo` snapshots taken during each piece's pause window. Compares the
     * SUM of the 4 individual deltas against the merged 4-op file's measured total (348MB
     * unmapped, from runMidWithPause) to test whether merging independently-compiled pieces adds
     * a measurable penalty beyond the sum of each piece's own intrinsic per-op staging cost.
     */
    fun runMidPiecesStandalone(context: Context, pauseMs: Long = 15000, gapMs: Long = 4000) {
        val dir = context.filesDir
        val pieceFiles = listOf("resnet0", "attn0", "attn1", "resnet1").map { File(dir, "mid_$it.tflite") }
        if (pieceFiles.any { !it.exists() }) {
            Log.w(TAG, "mid_{resnet0,attn0,attn1,resnet1}.tflite not all found in $dir, skipping")
            return
        }
        Log.i(TAG, "=== mid_block pieces standalone (sequential, external memory inspection) starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val sample = FloatArray(1 * 1280 * 32 * 32) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val emb = FloatArray(1 * 1280) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val ehs = FloatArray(1 * 77 * 2048) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }

            // Arbitrary consistent-shape inputs per piece -- this test only measures memory, not
            // correctness, so real chained data flow between pieces isn't needed.
            val perPieceInputs = listOf(
                listOf(sample, emb), // resnet0: [sample, emb]
                listOf(sample, ehs), // attn0: [resnet0_out, ehs] -- shape-compatible stand-in
                listOf(sample, sample, ehs), // attn1: [hidden_states_seq, residual, ehs]
                listOf(sample, emb), // resnet1: [attn_out, emb]
            )

            pieceFiles.forEachIndexed { idx, file ->
                val name = file.nameWithoutExtension
                val model = CompiledModel.create(file.absolutePath, options, env)
                val inputs = model.createInputBuffers()
                val outputs = model.createOutputBuffers()
                perPieceInputs[idx].forEachIndexed { i, arr -> inputs[i].writeFloat(arr) }
                model.run(inputs, outputs)
                val out = outputs[0].readFloat()
                Log.i(TAG, "runMidPiecesStandalone: $name run complete, output[0..2]=${out.take(3)}, PAUSING ${pauseMs}ms for external inspection NOW")
                Thread.sleep(pauseMs)
                model.close()
                Log.i(TAG, "runMidPiecesStandalone: $name closed")
                if (idx != pieceFiles.lastIndex) Thread.sleep(gapMs)
            }
            Log.i(TAG, "=== mid_block pieces standalone: ALL DONE ===")
        } catch (e: Throwable) {
            Log.e(TAG, "=== mid_block pieces standalone diagnostic: FAILED ===", e)
        }
    }

    fun runMidSeparate(context: Context) {
        val dir = context.filesDir
        val pieceFiles = listOf("resnet0", "attn0", "attn1", "resnet1").map { File(dir, "mid_$it.tflite") }
        if (pieceFiles.any { !it.exists() }) {
            Log.w(TAG, "mid_{resnet0,attn0,attn1,resnet1}.tflite not all found in $dir, skipping")
            return
        }

        Log.i(TAG, "=== mid_block NPU dispatch diagnostic (4 separate CompiledModels) starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            fun runPiece(path: String, vararg inputArrays: FloatArray): List<FloatArray> {
                val model = CompiledModel.create(path, options, env)
                val inputs = model.createInputBuffers()
                val outputs = model.createOutputBuffers()
                inputArrays.forEachIndexed { i, arr -> inputs[i].writeFloat(arr) }
                model.run(inputs, outputs)
                val results = outputs.map { it.readFloat() }
                model.close()
                return results
            }

            val (finalOut, timingAndRss) = withPeakRss {
                val start = System.currentTimeMillis()

                val sample = FloatArray(1 * 1280 * 32 * 32) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
                val emb = FloatArray(1 * 1280) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
                val encoderHiddenStates = FloatArray(1 * 77 * 2048) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }

                val resnet0Out = runPiece(pieceFiles[0].absolutePath, sample, emb)[0]
                val attn0Out = runPiece(pieceFiles[1].absolutePath, resnet0Out, encoderHiddenStates)
                val hiddenStatesSeq = attn0Out[0]
                // attn0's 2nd output is a pure pass-through alias of its own 1st input (same
                // tensor index in the flatbuffer) -- that resolves correctly inside the merged
                // graph (same underlying tensor), but reading it back from a STANDALONE run gives
                // whatever garbage is in that separately-allocated output buffer, since the
                // compiled op has nothing to actually compute for an aliased passthrough. Reuse
                // the already-known value directly instead of trusting attn0Out[1].
                val residual = resnet0Out
                val attn1Out = runPiece(pieceFiles[2].absolutePath, hiddenStatesSeq, residual, encoderHiddenStates)[0]
                val resnet1Out = runPiece(pieceFiles[3].absolutePath, attn1Out, emb)[0]

                resnet1Out to (System.currentTimeMillis() - start)
            }
            val (out, ms) = finalOut
            val peakRssKb = timingAndRss

            var hasNanOrInf = false
            var maxAbs = 0f
            for (v in out) {
                if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                val a = kotlin.math.abs(v)
                if (a > maxAbs) maxAbs = a
            }
            Log.i(
                TAG,
                "mid_block NPU (separate): ${ms}ms, peakRSS=${peakRssKb / 1024}MB, " +
                    "outputSize=${out.size}, hasNanOrInf=$hasNanOrInf, maxAbs=$maxAbs, " +
                    "output[0..4]=${out.take(5)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== mid_block NPU dispatch diagnostic (separate): FAILED ===", e)
        }
    }

    /**
     * The merged down_blocks[0,1,2] stage alone: 12 chained DISPATCH_OP pieces, ~1.66GB of NPU
     * blobs -- roughly a third of the full 36-op/5.39GB UNet. mid_block (4 ops/850MB) ran fine
     * (~1.1GB peak RSS); the full 36-op/5.39GB merge triggered a kernel lowmemorykiller
     * (~13GB dmabuf_rss). This is the middle data point: does the DMA-buf blowup scale with op
     * count, total blob size, or something else? Returns the 8 skip-connection outputs so a
     * later sequential test can feed them onward, but this diagnostic only checks that it loads
     * and runs without dying.
     */
    private fun runDownOnce(modelPath: String, env: Environment): Pair<List<FloatArray>, Long> {
        val options = CompiledModel.Options(Accelerator.NPU)
        val model = CompiledModel.create(modelPath, options, env)
        val inputs = model.createInputBuffers()
        val outputs = model.createOutputBuffers()

        // Signature order: skip0, emb, ehs (see merge_down.py)
        val skip0 = FloatArray(1 * 320 * 128 * 128) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
        val emb = FloatArray(1 * 1280) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
        val ehs = FloatArray(1 * 77 * 2048) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
        inputs[0].writeFloat(skip0)
        inputs[1].writeFloat(emb)
        inputs[2].writeFloat(ehs)

        val start = System.currentTimeMillis()
        model.run(inputs, outputs)
        val elapsedMs = System.currentTimeMillis() - start

        val results = outputs.map { it.readFloat() }
        model.close()
        return results to elapsedMs
    }

    fun runDownMerged(context: Context) {
        val modelFile = File(context.filesDir, "down_merged.tflite")
        if (!modelFile.exists()) {
            Log.w(TAG, "down_merged.tflite not found in ${context.filesDir}, skipping")
            return
        }

        Log.i(TAG, "=== down_blocks NPU dispatch diagnostic (merged, 12 ops/1.66GB) starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)

            val (runResult, peakRssKb) = withPeakRss { runDownOnce(modelFile.absolutePath, env) }
            val (outs, ms) = runResult
            var hasNanOrInf = false
            var maxAbs = 0f
            for (out in outs) {
                for (v in out) {
                    if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                    val a = kotlin.math.abs(v)
                    if (a > maxAbs) maxAbs = a
                }
            }
            Log.i(
                TAG,
                "down_blocks NPU (merged): ${ms}ms, peakRSS=${peakRssKb / 1024}MB, " +
                    "numOutputs=${outs.size}, hasNanOrInf=$hasNanOrInf, maxAbs=$maxAbs, " +
                    "skip8[0..4]=${outs.last().take(5)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== down_blocks NPU dispatch diagnostic (merged): FAILED ===", e)
        }
    }

    /**
     * Loads+runs down_merged.tflite, closes it, THEN loads+runs mid_merged.tflite -- both fully
     * sequential, nothing overlapping. Tests whether CompiledModel.close() actually releases its
     * DMA-buf allocations back to the system: if peak RSS/dmabuf here stays close to
     * max(down alone, mid alone) rather than growing to their SUM, stage-chunked sequential
     * loading is a viable architecture for the full UNet even though the full 36-op single merge
     * is not.
     */
    fun runDownThenMid(context: Context) {
        val downFile = File(context.filesDir, "down_merged.tflite")
        val midFile = File(context.filesDir, "mid_merged.tflite")
        if (!downFile.exists() || !midFile.exists()) {
            Log.w(TAG, "down_merged.tflite / mid_merged.tflite not both found in ${context.filesDir}, skipping")
            return
        }

        Log.i(TAG, "=== down_blocks -> close -> mid_block sequential diagnostic starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)

            val (result, peakRssKb) = withPeakRss {
                val start = System.currentTimeMillis()
                val (downOuts, downMs) = runDownOnce(downFile.absolutePath, env)
                val afterDownRssKb = currentRssKb()
                val (midOut, midMs) = runMidOnce(midFile.absolutePath, env)
                val afterMidRssKb = currentRssKb()
                Quad(downOuts, downMs, midOut, midMs) to Triple(
                    System.currentTimeMillis() - start,
                    afterDownRssKb,
                    afterMidRssKb,
                )
            }
            val (quad, timing) = result
            val (downOuts, downMs, midOut, midMs) = quad
            val (totalMs, afterDownRssKb, afterMidRssKb) = timing

            var hasNanOrInf = false
            for (v in midOut) if (v.isNaN() || v.isInfinite()) hasNanOrInf = true

            Log.i(
                TAG,
                "down->mid sequential: downMs=$downMs midMs=$midMs totalMs=$totalMs, " +
                    "afterDownRSS=${afterDownRssKb / 1024}MB, afterMidRSS=${afterMidRssKb / 1024}MB, " +
                    "peakRSS=${peakRssKb / 1024}MB, midHasNanOrInf=$hasNanOrInf, " +
                    "downSkip8[0..4]=${downOuts.last().take(5)}, midOut[0..4]=${midOut.take(5)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== down_blocks -> mid_block sequential diagnostic: FAILED ===", e)
        }
    }

    private data class Quad<A, B, C, D>(val first: A, val second: B, val third: C, val fourth: D)

    /**
     * The merged up_blocks[0,1,2] stage alone: 18 chained DISPATCH_OP pieces, ~2.6GB of NPU
     * blobs -- the third and largest stage chunk. Signature order (see merge_up.py):
     * mid_final, skip8, emb, ehs, skip7, skip6, skip5, skip4, skip3, skip0, skip1, skip2 ->
     * final_output. Shapes read directly out of up_merged.tflite's own SignatureDef.
     */
    private fun sinArray(size: Int): FloatArray = FloatArray(size) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }

    private fun runUpOnce(modelPath: String, env: Environment, inputArrays: List<FloatArray>): Pair<FloatArray, Long> {
        val options = CompiledModel.Options(Accelerator.NPU)
        val model = CompiledModel.create(modelPath, options, env)
        val inputs = model.createInputBuffers()
        val outputs = model.createOutputBuffers()
        inputArrays.forEachIndexed { i, arr -> inputs[i].writeFloat(arr) }

        val start = System.currentTimeMillis()
        model.run(inputs, outputs)
        val elapsedMs = System.currentTimeMillis() - start

        val result = outputs[0].readFloat()
        model.close()
        return result to elapsedMs
    }

    /**
     * Like [runMidWithPause], but for up_merged.tflite (18 ops, ~2.6GB blob) -- the largest
     * standalone stage we've confirmed safe (peak RSS ~3.0GB). Checks whether the mapped/unmapped
     * dma-buf split measured for mid_block (~1x mapped in-process + ~0.4x unmapped, ~1.33x total)
     * holds at 3x the blob size, or shifts as we approach whatever's actually driving the fatal
     * cliff seen at 5.2-5.4GB of concurrent residency.
     */
    fun runUpWithPause(context: Context, pauseMs: Long = 25000) {
        val modelFile = File(context.filesDir, "up_merged.tflite")
        if (!modelFile.exists()) {
            Log.w(TAG, "up_merged.tflite not found in ${context.filesDir}, skipping")
            return
        }
        Log.i(TAG, "=== up_blocks with pause diagnostic starting (for external memory inspection) ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val inputArrays = listOf(
                sinArray(1 * 1280 * 32 * 32), sinArray(1 * 1280 * 32 * 32), sinArray(1 * 1280),
                sinArray(1 * 77 * 2048), sinArray(1 * 1280 * 32 * 32), sinArray(1 * 640 * 32 * 32),
                sinArray(1 * 640 * 64 * 64), sinArray(1 * 640 * 64 * 64), sinArray(1 * 320 * 64 * 64),
                sinArray(1 * 320 * 128 * 128), sinArray(1 * 320 * 128 * 128), sinArray(1 * 320 * 128 * 128),
            )
            val options = CompiledModel.Options(Accelerator.NPU)
            val model = CompiledModel.create(modelFile.absolutePath, options, env)
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()
            inputArrays.forEachIndexed { i, arr -> inputs[i].writeFloat(arr) }
            model.run(inputs, outputs)
            val result = outputs[0].readFloat()
            Log.i(TAG, "runUpWithPause: run complete, output[0..4]=${result.take(5)}, PAUSING ${pauseMs}ms for external inspection NOW")
            Thread.sleep(pauseMs)
            Log.i(TAG, "runUpWithPause: pause complete, closing")
            model.close()
        } catch (e: Throwable) {
            Log.e(TAG, "=== up_blocks with pause diagnostic: FAILED ===", e)
        }
    }

    /**
     * Standalone int8-quantized resnet0 (32.3MB compiled blob, vs 64.2MB fp32 -- ~2x smaller
     * compiled, weights alone were 4x smaller before AOT compilation packs in the NPU program).
     * Same pause-and-measure pattern as [runMidWithPause]/[runUpWithPause]: checks whether the
     * unmapped dma-buf component actually shrinks the way the activation-driven-cost hypothesis
     * predicts, now that both weights AND activations are int8 (1 byte vs 4 bytes).
     */
    fun runResnet0Int8WithPause(context: Context, pauseMs: Long = 15000) {
        val modelFile = File(context.filesDir, "resnet0_int8.tflite")
        if (!modelFile.exists()) {
            Log.w(TAG, "resnet0_int8.tflite not found in ${context.filesDir}, skipping")
            return
        }
        Log.i(TAG, "=== resnet0 int8 with pause diagnostic starting (for external memory inspection) ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)
            val model = CompiledModel.create(modelFile.absolutePath, options, env)
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()
            val sample = ByteArray(1 * 1280 * 32 * 32) { i -> (i % 127).toByte() }
            val emb = ByteArray(1 * 1280) { i -> (i % 127).toByte() }
            inputs[0].writeInt8(sample)
            inputs[1].writeInt8(emb)
            model.run(inputs, outputs)
            val result = outputs[0].readInt8()
            Log.i(TAG, "runResnet0Int8WithPause: run complete, output[0..4]=${result.take(5)}, PAUSING ${pauseMs}ms for external inspection NOW")
            Thread.sleep(pauseMs)
            Log.i(TAG, "runResnet0Int8WithPause: pause complete, closing")
            model.close()
        } catch (e: Throwable) {
            Log.e(TAG, "=== resnet0 int8 with pause diagnostic: FAILED ===", e)
        }
    }

    fun runUpMerged(context: Context) {
        val modelFile = File(context.filesDir, "up_merged.tflite")
        if (!modelFile.exists()) {
            Log.w(TAG, "up_merged.tflite not found in ${context.filesDir}, skipping")
            return
        }

        Log.i(TAG, "=== up_blocks NPU dispatch diagnostic (merged, 18 ops/2.6GB) starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)

            // args_0..args_11: mid_final, skip8, emb, ehs, skip7, skip6, skip5, skip4, skip3,
            // skip0, skip1, skip2 (shapes from inspect_up_shapes.py)
            val inputArrays = listOf(
                sinArray(1 * 1280 * 32 * 32), // mid_final
                sinArray(1 * 1280 * 32 * 32), // skip8
                sinArray(1 * 1280), // emb
                sinArray(1 * 77 * 2048), // ehs
                sinArray(1 * 1280 * 32 * 32), // skip7
                sinArray(1 * 640 * 32 * 32), // skip6
                sinArray(1 * 640 * 64 * 64), // skip5
                sinArray(1 * 640 * 64 * 64), // skip4
                sinArray(1 * 320 * 64 * 64), // skip3
                sinArray(1 * 320 * 128 * 128), // skip0
                sinArray(1 * 320 * 128 * 128), // skip1
                sinArray(1 * 320 * 128 * 128), // skip2
            )

            val (runResult, peakRssKb) = withPeakRss { runUpOnce(modelFile.absolutePath, env, inputArrays) }
            val (out, ms) = runResult
            var hasNanOrInf = false
            var maxAbs = 0f
            for (v in out) {
                if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                val a = kotlin.math.abs(v)
                if (a > maxAbs) maxAbs = a
            }
            Log.i(
                TAG,
                "up_blocks NPU (merged): ${ms}ms, peakRSS=${peakRssKb / 1024}MB, " +
                    "outputSize=${out.size}, hasNanOrInf=$hasNanOrInf, maxAbs=$maxAbs, " +
                    "output[0..4]=${out.take(5)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== up_blocks NPU dispatch diagnostic (merged): FAILED ===", e)
        }
    }

    /**
     * Full end-to-end test: down_merged -> close -> mid_merged -> close -> up_merged -> close,
     * with REAL data flow between stages (down's skip outputs feed mid and up; mid's output
     * feeds up as mid_final) rather than each stage getting independent arbitrary inputs. This
     * is the complete SDXL UNet computation, broken into 3 sequentially-loaded NPU dispatch
     * files instead of 1 giant 36-op/5.39GB file (which OOMs) or 36 fully-independent pieces
     * (which has an unresolved async dispatch race). Checks whether peak RSS across the whole
     * chain stays bounded to the largest single stage (up_merged, ~2.6GB blobs) rather than
     * growing to the sum of all three, and whether the final output is finite.
     */
    fun runFullSequential(context: Context) {
        val downFile = File(context.filesDir, "down_merged.tflite")
        val midFile = File(context.filesDir, "mid_merged.tflite")
        val upFile = File(context.filesDir, "up_merged.tflite")
        if (!downFile.exists() || !midFile.exists() || !upFile.exists()) {
            Log.w(TAG, "down/mid/up_merged.tflite not all found in ${context.filesDir}, skipping")
            return
        }

        Log.i(TAG, "=== full sequential down->mid->up diagnostic starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)

            val skip0 = sinArray(1 * 320 * 128 * 128)
            val emb = sinArray(1 * 1280)
            val ehs = sinArray(1 * 77 * 2048)

            val (chainResult, peakRssKb) = withPeakRss {
                val t0 = System.currentTimeMillis()

                // down_merged: in=[skip0,emb,ehs] -> out=[skip3,skip1,skip2,skip4,skip5,skip6,skip7,skip8]
                val downOptions = CompiledModel.Options(Accelerator.NPU)
                val downModel = CompiledModel.create(downFile.absolutePath, downOptions, env)
                val downInputs = downModel.createInputBuffers()
                val downOutputs = downModel.createOutputBuffers()
                downInputs[0].writeFloat(skip0)
                downInputs[1].writeFloat(emb)
                downInputs[2].writeFloat(ehs)
                downModel.run(downInputs, downOutputs)
                val downOuts = downOutputs.map { it.readFloat() }
                downModel.close()
                val tDown = System.currentTimeMillis()
                val rssAfterDown = currentRssKb()

                val skip3 = downOuts[0]
                val skip1 = downOuts[1]
                val skip2 = downOuts[2]
                val skip4 = downOuts[3]
                val skip5 = downOuts[4]
                val skip6 = downOuts[5]
                val skip7 = downOuts[6]
                val skip8 = downOuts[7]

                // mid_merged: in=[sample,emb,ehs] -> out=[final_output]; sample := skip8
                val (midFinal, _) = runMidOnce(midFile.absolutePath, env, sample = skip8, emb = emb, encoderHiddenStates = ehs)
                val tMid = System.currentTimeMillis()
                val rssAfterMid = currentRssKb()

                // up_merged: args_0..11 = mid_final, skip8, emb, ehs, skip7, skip6, skip5, skip4,
                // skip3, skip0, skip1, skip2 -> final_output
                val upInputs = listOf(midFinal, skip8, emb, ehs, skip7, skip6, skip5, skip4, skip3, skip0, skip1, skip2)
                val (finalOutput, _) = runUpOnce(upFile.absolutePath, env, upInputs)
                val tUp = System.currentTimeMillis()
                val rssAfterUp = currentRssKb()

                SequentialResult(finalOutput, tDown - t0, tMid - tDown, tUp - tMid, rssAfterDown, rssAfterMid, rssAfterUp)
            }
            val r = chainResult

            var hasNanOrInf = false
            var maxAbs = 0f
            for (v in r.finalOutput) {
                if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                val a = kotlin.math.abs(v)
                if (a > maxAbs) maxAbs = a
            }
            Log.i(
                TAG,
                "full sequential UNet: downMs=${r.downMs} midMs=${r.midMs} upMs=${r.upMs} " +
                    "totalMs=${r.downMs + r.midMs + r.upMs}, " +
                    "rssAfterDown=${r.rssAfterDownKb / 1024}MB, rssAfterMid=${r.rssAfterMidKb / 1024}MB, " +
                    "rssAfterUp=${r.rssAfterUpKb / 1024}MB, peakRSS=${peakRssKb / 1024}MB, " +
                    "hasNanOrInf=$hasNanOrInf, maxAbs=$maxAbs, output[0..4]=${r.finalOutput.take(5)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== full sequential down->mid->up diagnostic: FAILED ===", e)
        }
    }

    /**
     * Holds all THREE stage CompiledModels open SIMULTANEOUSLY (create down, create mid, create
     * up -- none closed until the very end) instead of sequentially closing each before opening
     * the next (see [runFullSequential]). A real multi-step diffusion loop needs every stage on
     * every step; reloading ~2-3GB files from storage 20-50 times per image would dominate
     * runtime, so the realistic design keeps all three resident for the whole loop. The open
     * question: does concurrent residency of three separately-compiled files hit the same
     * DMA-buf cascade the single 36-op/5.39GB merge did (~13GB dmabuf_rss, fatal), or does it
     * stay bounded near the sum of their individual standalone peaks (~6.1GB: down 2.0 + mid 1.1
     * + up 3.0), since each is still its own CompiledModel instance rather than one giant
     * dispatch table? Logs RSS after each create() and each run() to localize where any blowup
     * happens.
     */
    fun runConcurrentResidency(context: Context) {
        val downFile = File(context.filesDir, "down_merged.tflite")
        val midFile = File(context.filesDir, "mid_merged.tflite")
        val upFile = File(context.filesDir, "up_merged.tflite")
        if (!downFile.exists() || !midFile.exists() || !upFile.exists()) {
            Log.w(TAG, "down/mid/up_merged.tflite not all found in ${context.filesDir}, skipping")
            return
        }

        Log.i(TAG, "=== concurrent residency (down+mid+up all open at once) diagnostic starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val skip0 = sinArray(1 * 320 * 128 * 128)
            val emb = sinArray(1 * 1280)
            val ehs = sinArray(1 * 77 * 2048)

            var finalOutput: FloatArray = FloatArray(0)
            var hasNanOrInf = false
            var maxAbs = 0f
            val rss = linkedMapOf<String, Long>()

            val (_, peakRssKb) = withPeakRss {
                rss["start"] = currentRssKb()

                val downModel = CompiledModel.create(downFile.absolutePath, options, env)
                rss["afterCreateDown"] = currentRssKb()
                val midModel = CompiledModel.create(midFile.absolutePath, options, env)
                rss["afterCreateMid"] = currentRssKb()
                val upModel = CompiledModel.create(upFile.absolutePath, options, env)
                rss["afterCreateUp"] = currentRssKb()

                val downInputs = downModel.createInputBuffers()
                val downOutputs = downModel.createOutputBuffers()
                downInputs[0].writeFloat(skip0)
                downInputs[1].writeFloat(emb)
                downInputs[2].writeFloat(ehs)
                downModel.run(downInputs, downOutputs)
                val downOuts = downOutputs.map { it.readFloat() }
                rss["afterRunDown"] = currentRssKb()

                val skip3 = downOuts[0]
                val skip1 = downOuts[1]
                val skip2 = downOuts[2]
                val skip4 = downOuts[3]
                val skip5 = downOuts[4]
                val skip6 = downOuts[5]
                val skip7 = downOuts[6]
                val skip8 = downOuts[7]

                val midInputs = midModel.createInputBuffers()
                val midOutputs = midModel.createOutputBuffers()
                midInputs[0].writeFloat(skip8)
                midInputs[1].writeFloat(emb)
                midInputs[2].writeFloat(ehs)
                midModel.run(midInputs, midOutputs)
                val midFinal = midOutputs[0].readFloat()
                rss["afterRunMid"] = currentRssKb()

                val upInputs = upModel.createInputBuffers()
                val upOutputs = upModel.createOutputBuffers()
                val upInputArrays = listOf(midFinal, skip8, emb, ehs, skip7, skip6, skip5, skip4, skip3, skip0, skip1, skip2)
                upInputArrays.forEachIndexed { i, arr -> upInputs[i].writeFloat(arr) }
                upModel.run(upInputs, upOutputs)
                finalOutput = upOutputs[0].readFloat()
                rss["afterRunUp"] = currentRssKb()

                downModel.close()
                midModel.close()
                upModel.close()
                rss["afterCloseAll"] = currentRssKb()

                for (v in finalOutput) {
                    if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                    val a = kotlin.math.abs(v)
                    if (a > maxAbs) maxAbs = a
                }
            }

            val rssLine = rss.entries.joinToString(", ") { (k, v) -> "$k=${v / 1024}MB" }
            Log.i(
                TAG,
                "concurrent residency: peakRSS=${peakRssKb / 1024}MB, hasNanOrInf=$hasNanOrInf, " +
                    "maxAbs=$maxAbs, output[0..4]=${finalOutput.take(5)} | $rssLine",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== concurrent residency diagnostic: FAILED ===", e)
        }
    }

    private data class SequentialResult(
        val finalOutput: FloatArray,
        val downMs: Long,
        val midMs: Long,
        val upMs: Long,
        val rssAfterDownKb: Long,
        val rssAfterMidKb: Long,
        val rssAfterUpKb: Long,
    )

    /**
     * Measures REAL per-cycle create()/run()/close() overhead for a single small piece
     * (mid_resnet0.tflite, 64MB fp32), repeated many times within ONE continuous process --
     * no adb relaunch, no app cold-start, no full-merged-model load contaminating the number.
     * Directly answers: is the "close each piece before creating the next, to actually free its
     * dma-buf scratch buffer" architecture's per-diffusion-step overhead (36 pieces x cycle
     * cost) a rounding error or a dealbreaker against the current merged-file's ~17s/step
     * compute time? Also logs RSS after each close() -- if it doesn't return close to baseline,
     * that's evidence against the "close() actually frees the scratch buffer" theory too.
     */
    fun measureCycleOverhead(context: Context, iterations: Int = 30) {
        val modelFile = File(context.filesDir, "mid_resnet0.tflite")
        if (!modelFile.exists()) {
            Log.w(TAG, "mid_resnet0.tflite not found in ${context.filesDir}, skipping")
            return
        }
        Log.i(TAG, "=== cycle overhead measurement starting ($iterations iterations) ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)
            val sample = FloatArray(1 * 1280 * 32 * 32) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val emb = FloatArray(1 * 1280) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }

            val baselineRssKb = currentRssKb()
            var totalCreateMs = 0L
            var totalRunMs = 0L
            var totalCloseMs = 0L
            var firstResult: FloatArray? = null
            var maxAbsDiffSeen = 0.0
            var mismatchCycles = 0

            for (i in 0 until iterations) {
                val t0 = System.currentTimeMillis()
                val model = CompiledModel.create(modelFile.absolutePath, options, env)
                val t1 = System.currentTimeMillis()

                val inputs = model.createInputBuffers()
                val outputs = model.createOutputBuffers()
                inputs[0].writeFloat(sample)
                inputs[1].writeFloat(emb)
                model.run(inputs, outputs)
                val result = outputs[0].readFloat()
                val t2 = System.currentTimeMillis()

                model.close()
                val t3 = System.currentTimeMillis()

                val createMs = t1 - t0
                val runMs = t2 - t1
                val closeMs = t3 - t2
                totalCreateMs += createMs
                totalRunMs += runMs
                totalCloseMs += closeMs

                var hasNanOrInf = false
                for (v in result) if (v.isNaN() || v.isInfinite()) hasNanOrInf = true

                // Same synthetic input every cycle -> output must be numerically identical
                // (or at worst bit-noise-close) across cycles if the NPU dispatch is properly
                // synchronous. Any real divergence here is direct evidence of the async
                // dispatch race (run() returning before the NPU finished writing) previously
                // seen with multi-piece chains -- this checks whether it also affects a single
                // isolated create->run->close cycle.
                val first = firstResult
                var cycleMaxAbsDiff = 0.0
                if (first == null) {
                    firstResult = result.copyOf()
                } else {
                    for (j in result.indices) {
                        val diff = kotlin.math.abs((result[j] - first[j]).toDouble())
                        if (diff > cycleMaxAbsDiff) cycleMaxAbsDiff = diff
                    }
                    if (cycleMaxAbsDiff > maxAbsDiffSeen) maxAbsDiffSeen = cycleMaxAbsDiff
                    if (cycleMaxAbsDiff > 1e-6) mismatchCycles++
                }

                Log.i(
                    TAG,
                    "CYCLE $i: createMs=$createMs runMs=$runMs closeMs=$closeMs " +
                        "rssKb=${currentRssKb()} hasNanOrInf=$hasNanOrInf " +
                        "out0..2=${result[0]},${result[1]},${result[2]} maxAbsDiffVsFirst=$cycleMaxAbsDiff",
                )
            }

            Log.i(
                TAG,
                "CYCLE_SUMMARY iterations=$iterations " +
                    "avgCreateMs=${totalCreateMs / iterations} " +
                    "avgRunMs=${totalRunMs / iterations} " +
                    "avgCloseMs=${totalCloseMs / iterations} " +
                    "baselineRssKb=$baselineRssKb finalRssKb=${currentRssKb()} " +
                    "mismatchCycles=$mismatchCycles maxAbsDiffSeen=$maxAbsDiffSeen",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== cycle overhead measurement: FAILED ===", e)
        }
    }

    private fun readFloatFile(file: File): FloatArray {
        val bytes = file.readBytes()
        val buf = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val out = FloatArray(buf.remaining())
        buf.get(out)
        return out
    }

    private fun writeFloatFile(file: File, data: FloatArray) {
        val buf = java.nio.ByteBuffer.allocate(data.size * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        buf.asFloatBuffer().put(data)
        file.writeBytes(buf.array())
    }

    /**
     * One real diffusion step of the full merged UNet, driven from an external (desktop Python)
     * scheduler loop -- reads 5 real input tensors (not fixed sin() test data) from
     * `context.filesDir/unet_step_in/`, runs the model, writes the noise-prediction output to
     * `context.filesDir/unet_step_out.bin`. One Kotlin/app-relaunch invocation per diffusion
     * step (reloads the 5GB model each time, ~3-4s overhead on top of the ~13s compute) --
     * simpler and more robust than keeping one process alive across steps with file-polling
     * signaling, at the cost of some speed. Used for real image-generation validation of the
     * int8 mixed-precision UNet, not just synthetic dispatch-resolves-without-crashing checks.
     */
    fun runFullUnetStep(context: Context) {
        val modelFile = File(context.filesDir, "unet_merged_full.tflite")
        val inDir = File(context.filesDir, "unet_step_in")
        val outFile = File(context.filesDir, "unet_step_out.bin")
        if (!modelFile.exists() || !inDir.exists()) {
            Log.w(TAG, "unet_merged_full.tflite or unet_step_in/ not found, skipping")
            return
        }
        Log.i(TAG, "=== full UNet single-step (from files) starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)
            val model = CompiledModel.create(modelFile.absolutePath, options, env)
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()

            inputs[0].writeFloat(readFloatFile(File(inDir, "t_emb.bin")))
            inputs[1].writeFloat(readFloatFile(File(inDir, "text_embeds.bin")))
            inputs[2].writeFloat(readFloatFile(File(inDir, "time_ids.bin")))
            inputs[3].writeFloat(readFloatFile(File(inDir, "sample.bin")))
            inputs[4].writeFloat(readFloatFile(File(inDir, "ehs.bin")))

            val start = System.currentTimeMillis()
            model.run(inputs, outputs)
            val elapsedMs = System.currentTimeMillis() - start

            val result = outputs[0].readFloat()
            model.close()
            writeFloatFile(outFile, result)
            Log.i(TAG, "STEP_DONE ${elapsedMs}ms outputSize=${result.size} output[0..2]=${result.take(3)}")
        } catch (e: Throwable) {
            Log.e(TAG, "=== full UNet single-step: FAILED ===", e)
        }
    }

    /**
     * Same real-diffusion-step contract as [runFullUnetStep] (reads 5 real input tensors from
     * `context.filesDir/unet_step_in/`, writes noise-prediction output to
     * `context.filesDir/unet_step_out.bin`, logs STEP_DONE) -- but runs the UNet as the 36
     * separate RESHAPE-wrapped CompiledModel instances ([UNET_PIECES]/[runFullUnetSeparate])
     * instead of one merged file. This is the actual stress test: does a REAL multi-step
     * diffusion loop (each step = a fresh app process = a fresh 36-piece create/run/close chain)
     * survive where the merged-file approach died at step 6 with a real lowmemorykiller kill
     * (dmabuf_rss=12702716kB)? A single clean run already succeeded ([runFullUnetSeparate],
     * ~733MB self-RSS) but self-reported RSS undersold the real cost before -- only repeated
     * real invocations can tell us whether it's actually fixed.
     */
    fun runFullUnetSeparateStep(context: Context) {
        // The current per-model production bundle location. This diagnostic deliberately uses
        // the same batch-1 AOT pieces as the shipped conditional-only NPU path.
        val dir = File(context.filesDir, "npu-unet/pureTukanoNSFW-xl")
        val inDir = File(context.filesDir, "unet_step_in")
        val outFile = File(context.filesDir, "unet_step_out.bin")
        val missing = UNET_PIECES.filter { !File(dir, it.fileName).exists() }
        if (missing.isNotEmpty() || !inDir.exists()) {
            Log.w(TAG, "npu-unet/pureTukanoNSFW-xl/ (missing ${missing.size}) or unet_step_in/ not found, skipping")
            return
        }
        Log.i(TAG, "=== full UNet single-step, separate-instances (from files) starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val roleMap = HashMap<String, FloatArray>()
            roleMap["t_emb"] = readFloatFile(File(inDir, "t_emb.bin"))
            roleMap["text_embeds"] = readFloatFile(File(inDir, "text_embeds.bin"))
            roleMap["time_ids"] = readFloatFile(File(inDir, "time_ids.bin"))
            roleMap["sample"] = readFloatFile(File(inDir, "sample.bin"))
            roleMap["ehs"] = readFloatFile(File(inDir, "ehs.bin"))

            val lastConsumerIndex = HashMap<String, Int>()
            UNET_PIECES.forEachIndexed { i, piece -> piece.inputRoles.forEach { role -> lastConsumerIndex[role] = i } }

            val start = System.currentTimeMillis()
            UNET_PIECES.forEachIndexed { pieceIdx, piece ->
                val model = CompiledModel.create(File(dir, piece.fileName).absolutePath, options, env)
                val inputs = model.createInputBuffers()
                val outputs = model.createOutputBuffers()
                piece.inputRoles.forEachIndexed { i, role ->
                    val arr = roleMap[role] ?: error("missing role '$role' for piece ${piece.name}")
                    inputs[i].writeFloat(arr)
                }
                piece.inputRoles.forEach { role -> if (lastConsumerIndex[role] == pieceIdx) roleMap.remove(role) }
                model.run(inputs, outputs)
                val results = outputs.map { it.readFloat() }
                model.close()
                piece.outputRoles.forEachIndexed { i, role -> roleMap[role] = results[i] }
            }
            val elapsedMs = System.currentTimeMillis() - start

            val result = roleMap["final_output"] ?: error("final_output never produced")
            writeFloatFile(outFile, result)
            Log.i(TAG, "STEP_DONE ${elapsedMs}ms outputSize=${result.size} output[0..2]=${result.take(3)}")
        } catch (e: Throwable) {
            Log.e(TAG, "=== full UNet single-step, separate-instances: FAILED ===", e)
        }
    }

    /**
     * Like [runFullUnet], but holds the CompiledModel open (run complete, not yet closed) for a
     * pause so an external `adb shell dumpsys meminfo <pkg>` snapshot can be taken. Self-reported
     * VmRSS (what [withPeakRss] polls) only counts memory mapped into THIS process's page
     * tables -- an NPU/DMA-BUF buffer the dispatch driver holds via fd/IOMMU without mapping it
     * into our address space wouldn't show up there at all, even though it's real resident memory
     * the kernel's lowmemorykiller does account against us (that's how the original all-fp32
     * merge's ~13GB "dmabuf_rss" figure that triggered a real OOM kill was measured -- NOT via
     * this app's own VmRSS). Needed to get a real, comparable number for the mixed-precision
     * merge instead of trusting a possibly-understated self-reported 4.1GB RSS figure.
     */
    fun runFullUnetWithPause(context: Context, pauseMs: Long = 25000) {
        val modelFile = File(context.filesDir, "unet_merged_full.tflite")
        if (!modelFile.exists()) {
            Log.w(TAG, "unet_merged_full.tflite not found in ${context.filesDir}, skipping")
            return
        }
        Log.i(TAG, "=== full UNet with pause diagnostic starting (for external memory inspection) ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)
            val model = CompiledModel.create(modelFile.absolutePath, options, env)
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()
            val tEmb = FloatArray(1 * 320) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val textEmbeds = FloatArray(1 * 1280) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val timeIds = floatArrayOf(1024f, 1024f, 0f, 0f, 1024f, 1024f)
            val sample = FloatArray(1 * 4 * 128 * 128) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            val ehs = FloatArray(1 * 77 * 2048) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            inputs[0].writeFloat(tEmb)
            inputs[1].writeFloat(textEmbeds)
            inputs[2].writeFloat(timeIds)
            inputs[3].writeFloat(sample)
            inputs[4].writeFloat(ehs)
            model.run(inputs, outputs)
            val result = outputs[0].readFloat()
            Log.i(
                TAG,
                "runFullUnetWithPause: run complete, output[0..4]=${result.take(5)}, " +
                    "PAUSING ${pauseMs}ms for external inspection NOW",
            )
            Thread.sleep(pauseMs)
            Log.i(TAG, "runFullUnetWithPause: pause complete, closing")
            model.close()
        } catch (e: Throwable) {
            Log.e(TAG, "=== full UNet with pause diagnostic: FAILED ===", e)
        }
    }

    /**
     * The full SDXL UNet: 36 chained DISPATCH_OP pieces (front embed/conv_in, down_blocks[0,1,2],
     * mid_block, up_blocks[0,1,2]) spliced into one file via merge_full_unet.py. Inputs are
     * arbitrary non-zero values -- checks that dispatch resolves end-to-end and produces finite
     * output, not numeric correctness against a PyTorch reference.
     */
    fun runFullUnet(context: Context) {
        val modelFile = File(context.filesDir, "unet_merged_full.tflite")
        if (!modelFile.exists()) {
            Log.w(TAG, "unet_merged_full.tflite not found in ${context.filesDir}, skipping")
            return
        }

        Log.i(TAG, "=== full UNet NPU dispatch diagnostic starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val (runResult, peakRssKb) = withPeakRss {
                val model = CompiledModel.create(modelFile.absolutePath, options, env)
                val inputs = model.createInputBuffers()
                val outputs = model.createOutputBuffers()

                // Actual external input order, per merge_full_unet.py's own logged
                // `external inputs: ['t_emb', 'text_embeds', 'time_ids', 'sample', 'ehs']`
                // (the comment this replaced assumed sample-first, which doesn't match --
                // confirmed by a real on-device TensorBuffer size-mismatch crash).
                val sample = FloatArray(1 * 4 * 128 * 128) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
                val tEmb = FloatArray(1 * 320) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
                val textEmbeds = FloatArray(1 * 1280) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
                val timeIds = floatArrayOf(1024f, 1024f, 0f, 0f, 1024f, 1024f)
                val ehs = FloatArray(1 * 77 * 2048) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
                inputs[0].writeFloat(tEmb)
                inputs[1].writeFloat(textEmbeds)
                inputs[2].writeFloat(timeIds)
                inputs[3].writeFloat(sample)
                inputs[4].writeFloat(ehs)

                val start = System.currentTimeMillis()
                model.run(inputs, outputs)
                val elapsedMs = System.currentTimeMillis() - start

                val result = outputs[0].readFloat()
                model.close()
                result to elapsedMs
            }
            val (out, ms) = runResult

            var hasNanOrInf = false
            var maxAbs = 0f
            for (v in out) {
                if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                val a = kotlin.math.abs(v)
                if (a > maxAbs) maxAbs = a
            }
            Log.i(
                TAG,
                "full UNet NPU: ${ms}ms, peakRSS=${peakRssKb / 1024}MB, outputSize=${out.size}, " +
                    "hasNanOrInf=$hasNanOrInf, maxAbs=$maxAbs, output[0..4]=${out.take(5)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== full UNet NPU dispatch diagnostic: FAILED ===", e)
        }
    }

    private data class PieceSpec(
        val name: String,
        val fileName: String,
        val inputRoles: List<String>,
        val outputRoles: List<String>,
    )

    // Generated from full_unet_pieces.json (build_full_unet_wrapped.py) -- the full 36-piece
    // plain-fp32 UNet topology (see merge_full_unet.py), each piece RESHAPE-wrapped
    // (build_reshape_wrapped_piece.py) so it's numerically correct as a standalone CompiledModel
    // (see project memory: bit-exact vs the merged reference across the full mid_block chain).
    // Already in dependency order.
    private val UNET_PIECES = listOf(
        PieceSpec("embed", "u_embed_wrapped.tflite", listOf("t_emb", "text_embeds", "time_ids"), listOf("emb")),
        PieceSpec("conv_in", "u_conv_in_wrapped.tflite", listOf("sample"), listOf("skip0")),
        PieceSpec("down0", "u_down0_wrapped.tflite", listOf("skip0", "emb"), listOf("skip1", "skip2", "skip3")),
        PieceSpec("down1_resnet0", "u_down1_resnet0_wrapped.tflite", listOf("skip3", "emb"), listOf("d1r0")),
        PieceSpec("down1_attn0", "u_down1_attn0_wrapped.tflite", listOf("d1r0", "ehs"), listOf("skip4")),
        PieceSpec("down1_resnet1", "u_down1_resnet1_wrapped.tflite", listOf("skip4", "emb"), listOf("d1r1")),
        PieceSpec("down1_attn1", "u_down1_attn1_wrapped.tflite", listOf("d1r1", "ehs"), listOf("skip5")),
        PieceSpec("down1_downsample", "u_down1_downsample_wrapped.tflite", listOf("skip5"), listOf("skip6")),
        PieceSpec("down2_resnet0", "u_down2_resnet0_wrapped.tflite", listOf("skip6", "emb"), listOf("d2r0")),
        PieceSpec("down2_attn0_a", "u_down2_attn0_a_wrapped.tflite", listOf("d2r0", "ehs"), listOf("d2a0_hidden")),
        PieceSpec("down2_attn0_b", "u_down2_attn0_b_wrapped.tflite", listOf("d2a0_hidden", "d2r0", "ehs"), listOf("skip7")),
        PieceSpec("down2_resnet1", "u_down2_resnet1_wrapped.tflite", listOf("skip7", "emb"), listOf("d2r1")),
        PieceSpec("down2_attn1_a", "u_down2_attn1_a_wrapped.tflite", listOf("d2r1", "ehs"), listOf("d2a1_hidden")),
        PieceSpec("down2_attn1_b", "u_down2_attn1_b_wrapped.tflite", listOf("d2a1_hidden", "d2r1", "ehs"), listOf("skip8")),
        PieceSpec("mid_resnet0", "u_mid_resnet0_wrapped.tflite", listOf("skip8", "emb"), listOf("mid_r0")),
        PieceSpec("mid_attn0", "u_mid_attn0_wrapped.tflite", listOf("mid_r0", "ehs"), listOf("mid_hidden")),
        PieceSpec("mid_attn1", "u_mid_attn1_wrapped.tflite", listOf("mid_hidden", "mid_r0", "ehs"), listOf("mid_attn_out")),
        PieceSpec("mid_resnet1", "u_mid_resnet1_wrapped.tflite", listOf("mid_attn_out", "emb"), listOf("mid_final")),
        PieceSpec("up0_resnet0", "u_up0_resnet0_wrapped.tflite", listOf("mid_final", "skip8", "emb"), listOf("u0r0")),
        PieceSpec("up0_attn0_a", "u_up0_attn0_a_wrapped.tflite", listOf("u0r0", "ehs"), listOf("u0a0_hidden")),
        PieceSpec("up0_attn0_b", "u_up0_attn0_b_wrapped.tflite", listOf("u0a0_hidden", "u0r0", "ehs"), listOf("u0_after0")),
        PieceSpec("up0_resnet1", "u_up0_resnet1_wrapped.tflite", listOf("u0_after0", "skip7", "emb"), listOf("u0r1")),
        PieceSpec("up0_attn1_a", "u_up0_attn1_a_wrapped.tflite", listOf("u0r1", "ehs"), listOf("u0a1_hidden")),
        PieceSpec("up0_attn1_b", "u_up0_attn1_b_wrapped.tflite", listOf("u0a1_hidden", "u0r1", "ehs"), listOf("u0_after1")),
        PieceSpec("up0_resnet2", "u_up0_resnet2_wrapped.tflite", listOf("u0_after1", "skip6", "emb"), listOf("u0r2")),
        PieceSpec("up0_attn2_a", "u_up0_attn2_a_wrapped.tflite", listOf("u0r2", "ehs"), listOf("u0a2_hidden")),
        PieceSpec("up0_attn2_b", "u_up0_attn2_b_wrapped.tflite", listOf("u0a2_hidden", "u0r2", "ehs"), listOf("u0_after2")),
        PieceSpec("up0_upsample", "u_up0_upsample_wrapped.tflite", listOf("u0_after2"), listOf("up0_out")),
        PieceSpec("up1_resnet0", "u_up1_resnet0_wrapped.tflite", listOf("up0_out", "skip5", "emb"), listOf("u1r0")),
        PieceSpec("up1_attn0", "u_up1_attn0_wrapped.tflite", listOf("u1r0", "ehs"), listOf("u1_after0")),
        PieceSpec("up1_resnet1", "u_up1_resnet1_wrapped.tflite", listOf("u1_after0", "skip4", "emb"), listOf("u1r1")),
        PieceSpec("up1_attn1", "u_up1_attn1_wrapped.tflite", listOf("u1r1", "ehs"), listOf("u1_after1")),
        PieceSpec("up1_resnet2", "u_up1_resnet2_wrapped.tflite", listOf("u1_after1", "skip3", "emb"), listOf("u1r2")),
        PieceSpec("up1_attn2", "u_up1_attn2_wrapped.tflite", listOf("u1r2", "ehs"), listOf("u1_after2")),
        PieceSpec("up1_upsample", "u_up1_upsample_wrapped.tflite", listOf("u1_after2"), listOf("up1_out")),
        PieceSpec("up2", "u_up2_wrapped.tflite", listOf("up1_out", "emb", "skip0", "skip1", "skip2"), listOf("final_output")),
    )

    /**
     * The full 36-piece UNet run as SEPARATE CompiledModel instances (create -> run -> close per
     * piece, each RESHAPE-wrapped for correctness -- see [compareMidMergedVsReshapeWrapped] and
     * project memory), wired via a role->FloatArray map matching [UNET_PIECES]'s dependency
     * order. This is the actual point of the whole RESHAPE-wrap investigation: does the real
     * per-call memory cost (previously ~12-13GB dmabuf_rss with the single 73-op merged file,
     * confirmed via a real lowmemorykiller kill) actually drop when every piece is closed before
     * the next one loads, now that correctness is no longer in question?
     */
    fun runFullUnetSeparate(context: Context) {
        val dir = File(context.filesDir, "unet_wrapped")
        val missing = UNET_PIECES.filter { !File(dir, it.fileName).exists() }
        if (missing.isNotEmpty()) {
            Log.w(TAG, "missing ${missing.size} piece file(s) in $dir, e.g. ${missing.first().fileName}, skipping")
            return
        }
        Log.i(TAG, "=== full UNet separate-instances (36 RESHAPE-wrapped pieces) run starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val roleMap = HashMap<String, FloatArray>()
            roleMap["sample"] = FloatArray(1 * 4 * 128 * 128) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            roleMap["t_emb"] = FloatArray(1 * 320) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            roleMap["text_embeds"] = FloatArray(1 * 1280) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            roleMap["time_ids"] = FloatArray(1 * 6) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }
            roleMap["ehs"] = FloatArray(1 * 77 * 2048) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }

            // Without this, roleMap accumulates EVERY intermediate tensor ever produced (skip
            // connections, attention hidden states, ...) for the whole 36-piece run and blows
            // the 256MB Java heap (hit in practice: OutOfMemoryError inside TensorBuffer.
            // readFloat() around piece #29 of 36). Free a role's array right after its LAST
            // consuming piece reads it -- mirrors a real pipeline's actual liveness, not a
            // dmabuf/NPU-side cost.
            val lastConsumerIndex = HashMap<String, Int>()
            UNET_PIECES.forEachIndexed { i, piece -> piece.inputRoles.forEach { role -> lastConsumerIndex[role] = i } }

            val start = System.currentTimeMillis()
            var peakRssKb = currentRssKb()
            UNET_PIECES.forEachIndexed { pieceIdx, piece ->
                val pieceStart = System.currentTimeMillis()
                val model = CompiledModel.create(File(dir, piece.fileName).absolutePath, options, env)
                val createMs = System.currentTimeMillis()
                val inputs = model.createInputBuffers()
                val outputs = model.createOutputBuffers()
                val buffersMs = System.currentTimeMillis()
                piece.inputRoles.forEachIndexed { i, role ->
                    val arr = roleMap[role] ?: error("missing role '$role' for piece ${piece.name}")
                    inputs[i].writeFloat(arr)
                }
                val writeMs = System.currentTimeMillis()
                piece.inputRoles.forEach { role -> if (lastConsumerIndex[role] == pieceIdx) roleMap.remove(role) }
                model.run(inputs, outputs)
                val runMs = System.currentTimeMillis()
                val results = outputs.map { it.readFloat() }
                val readMs = System.currentTimeMillis()
                model.close()
                val closeMs = System.currentTimeMillis()
                piece.outputRoles.forEachIndexed { i, role -> roleMap[role] = results[i] }
                val rssKb = currentRssKb()
                if (rssKb > peakRssKb) peakRssKb = rssKb
                Log.i(
                    TAG,
                    "  phases ${piece.name}: create=${createMs - pieceStart}ms " +
                        "buffers=${buffersMs - createMs}ms write=${writeMs - buffersMs}ms " +
                        "run=${runMs - writeMs}ms read=${readMs - runMs}ms " +
                        "close=${closeMs - readMs}ms rssKb=$rssKb",
                )
            }
            val totalMs = System.currentTimeMillis() - start

            val finalOutput = roleMap["final_output"] ?: error("final_output never produced")
            var hasNanOrInf = false
            var maxAbs = 0f
            for (v in finalOutput) {
                if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                val a = kotlin.math.abs(v)
                if (a > maxAbs) maxAbs = a
            }
            Log.i(
                TAG,
                "FULL_UNET_SEPARATE: totalMs=$totalMs peakSelfRssKb=$peakRssKb " +
                    "outputSize=${finalOutput.size} hasNanOrInf=$hasNanOrInf maxAbs=$maxAbs " +
                    "output[0..4]=${finalOutput.take(5)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== full UNet separate-instances run: FAILED ===", e)
        }
    }

    // Reads a raw little-endian float32 binary (no header -- produced on the desktop via
    // numpy's ndarray.tofile(), NOT .npy, which has a variable-length header this doesn't skip).
    private fun readFloatBin(file: File): FloatArray {
        val bytes = file.readBytes()
        val buf = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        val out = FloatArray(bytes.size / 4)
        buf.asFloatBuffer().get(out)
        return out
    }

    /**
     * THROWAWAY one-off: does the FLUX.2 [klein] single_blocks.0 AOT-compiled-for-Tensor-G5
     * artifact (~/code/litert-torch/scratch/models/flux2_klein_probe/single0_full_realtoks_noflags_aot/
     * single0_full_realtoks_Google_Tensor_G5.tflite -- 87/87 ops, 1 NPU partition, compiled WITHOUT
     * google_tensor_enable_large_model_support/google_tensor_sharding_intensity, see
     * docs/flux2-klein-conversion.md) actually run on real Tensor G5 hardware at the real 1024px
     * production token count (4608 = 512 text + 4096 image tokens), and does its output match the
     * PyTorch reference (already confirmed correct on desktop CPU/XNNPACK, maxAbsDiff=0.000626 vs
     * meanAbsRef=6.029 -- see scratch/compute_klein_single0_torch_reference.py /
     * scratch/run_klein_single0_tflite_and_compare.py)?
     *
     * Expects these raw float32 .bin files (numpy .tofile(), not .npy) pushed to
     * context.filesDir/klein_single0/: x.bin, pe.bin, mod_shift.bin, mod_scale.bin, mod_gate.bin,
     * torch_out.bin -- plus the AOT .tflite itself at
     * context.filesDir/klein_single0/single0_full_realtoks_Google_Tensor_G5.tflite.
     *
     * Buffer lifecycle fix vs. prior run: outputs are read before model.close(), and input/output
     * TensorBuffers are explicitly closed in the correct order (inputs, then outputs, then model)
     * after the read is complete. The prior run called model.close() before outputs.forEach
     * { it.close() } was ever invoked, which is the Kotlin-side analogue of the C++ destruction-
     * order bug that caused the SDXL batch=2 crash (see docs/npu-unet-conversion.md). Whether
     * this matters for the Kotlin LiteRT API is unknown, but it was never done correctly before
     * and is the lowest-risk change to try first.
     */
    fun runKleinSingle0(context: Context) {
        val dir = File(context.filesDir, "klein_single0")
        val modelFile = File(dir, "single0_full_realtoks_Google_Tensor_G5.tflite")
        val inputNames = listOf("x", "pe", "mod_shift", "mod_scale", "mod_gate")
        val inputFiles = inputNames.map { File(dir, "$it.bin") }
        val refFile = File(dir, "torch_out.bin")
        if (!modelFile.exists() || inputFiles.any { !it.exists() } || !refFile.exists()) {
            Log.w(TAG, "klein_single0 model/input/reference files not all found in $dir, skipping")
            return
        }

        Log.i(TAG, "=== Klein single_blocks.0 NPU diagnostic starting (real 1024px token shape) ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val model = CompiledModel.create(modelFile.absolutePath, options, env)
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()
            inputFiles.forEachIndexed { i, f -> inputs[i].writeFloat(readFloatBin(f)) }

            val start = System.currentTimeMillis()
            model.run(inputs, outputs)
            val elapsedMs = System.currentTimeMillis() - start

            // Read all outputs BEFORE closing anything. Then close in order: inputs, outputs,
            // model -- so the backing dmabuf is not unmapped while we are still reading from it.
            val npuOut = outputs[0].readFloat()
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
            model.close()

            val refOut = readFloatBin(refFile)
            if (npuOut.size != refOut.size) {
                Log.e(TAG, "KLEIN_SINGLE0: size mismatch npuOut=${npuOut.size} refOut=${refOut.size}")
                return
            }
            var maxAbsDiff = 0f
            var sumAbsDiff = 0.0
            var sumAbsRef = 0.0
            var hasNanOrInf = false
            for (i in npuOut.indices) {
                val v = npuOut[i]
                if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                val diff = kotlin.math.abs(v - refOut[i])
                if (diff > maxAbsDiff) maxAbsDiff = diff
                sumAbsDiff += diff
                sumAbsRef += kotlin.math.abs(refOut[i])
            }
            val meanAbsDiff = sumAbsDiff / npuOut.size
            val meanAbsRef = sumAbsRef / npuOut.size
            Log.i(
                TAG,
                "KLEIN_SINGLE0: runMs=$elapsedMs hasNanOrInf=$hasNanOrInf " +
                    "maxAbsDiff=$maxAbsDiff meanAbsDiff=$meanAbsDiff meanAbsRef=$meanAbsRef " +
                    "npuOut[0..4]=${npuOut.take(5)} ref[0..4]=${refOut.take(5)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== Klein single_blocks.0 NPU diagnostic: FAILED ===", e)
        }
    }

    /**
     * THROWAWAY one-off: same question as [runKleinSingle0] but using the smaller probe-shape
     * artifact compiled at 80 tokens (16 text + 64 image) --
     * single0_noflags_aot/single0_Google_Tensor_G5.tflite (also 87/87 ops, 1 NPU partition,
     * compiled at the same time as the real-shape artifact).
     *
     * Motivation (docs/flux2-klein-conversion.md step 6): isolate whether the real-shape firmware
     * fault is token-count/buffer-size dependent. The DIVE's mcause=0xb (RISC-V STORE_ACCESS_FAULT)
     * could mean the firmware tried to write to a DMA buffer address that falls outside the range
     * the DIVE's MMU will accept. At 4608 tokens the largest intermediate activation is the
     * linear1 pre-attention output: 4608 × 9216 × 4 bytes ≈ 170 MB -- at 80 tokens it's only
     * ~3 MB. If the 80-token artifact runs clean while the 4608-token one crashes, buffer-range
     * is a real variable. If it crashes too, size is not the variable and the investigation should
     * look elsewhere (specific op patterns, alignment, dispatch payload structure, etc).
     *
     * Uses synthetic sine-wave inputs (no I/O files needed) since we have no saved PyTorch
     * reference at the 80-token shape. Only checks: does it run without crashing the NPU, and
     * is the output finite and non-zero?
     *
     * Expects: context.filesDir/klein_single0/single0_Google_Tensor_G5.tflite
     */
    fun runKleinSingle0SmallShape(context: Context) {
        val dir = File(context.filesDir, "klein_single0")
        val modelFile = File(dir, "single0_Google_Tensor_G5.tflite")
        if (!modelFile.exists()) {
            Log.w(TAG, "klein_single0/single0_Google_Tensor_G5.tflite not found in $dir, skipping")
            return
        }

        // 80-token probe shape: 16 text + 64 image tokens (flux2_klein_probe.py defaults).
        // x: [1, 80, 3072], pe: [1, 1, 80, 64, 2, 2], mod_*: [1, 1, 3072].
        val tokenCount = 80
        val hiddenSize = 3072
        val ropeComplexPairs = 64 // head_dim=128 -> 64 complex pairs per the RoPE convention

        Log.i(TAG, "=== Klein single_blocks.0 small-shape (80-token) NPU diagnostic starting ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val model = CompiledModel.create(modelFile.absolutePath, options, env)
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()

            // x: [1, tokenCount, hiddenSize]
            inputs[0].writeFloat(FloatArray(1 * tokenCount * hiddenSize) { i -> kotlin.math.sin(i.toFloat()) * 0.1f })
            // pe: [1, 1, tokenCount, ropeComplexPairs, 2, 2]
            inputs[1].writeFloat(FloatArray(1 * 1 * tokenCount * ropeComplexPairs * 2 * 2) { i -> kotlin.math.sin(i.toFloat()) * 0.1f })
            // mod_shift, mod_scale, mod_gate: each [1, 1, hiddenSize]
            for (k in 2..4) {
                inputs[k].writeFloat(FloatArray(1 * 1 * hiddenSize) { i -> kotlin.math.sin(i.toFloat()) * 0.1f })
            }

            val start = System.currentTimeMillis()
            model.run(inputs, outputs)
            val elapsedMs = System.currentTimeMillis() - start

            // Same lifecycle discipline as runKleinSingle0: read before closing.
            val npuOut = outputs[0].readFloat()
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
            model.close()

            var hasNanOrInf = false
            var maxAbs = 0f
            var allZero = true
            for (v in npuOut) {
                if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                val a = kotlin.math.abs(v)
                if (a > maxAbs) maxAbs = a
                if (a > 1e-9f) allZero = false
            }
            Log.i(
                TAG,
                "KLEIN_SINGLE0_SMALL: runMs=$elapsedMs hasNanOrInf=$hasNanOrInf " +
                    "maxAbs=$maxAbs allZero=$allZero outputSize=${npuOut.size} " +
                    "npuOut[0..4]=${npuOut.take(5)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== Klein single_blocks.0 small-shape NPU diagnostic: FAILED ===", e)
        }
    }

    /**
     * THROWAWAY generalized token-count probe: runs any AOT-compiled Klein single-stream block
     * artifact with synthetic sin-wave inputs at a specified token count, without needing any
     * reference .bin files on-device. Used for the binary-search of the Darwinn DMA address-range
     * ceiling (DIVE mcause=0xb STORE_ACCESS_FAULT at 4608 tokens, clean at 80 tokens).
     *
     * Invoked via intent extras (all required):
     *   - "klein_token_probe_file"   : filename (not full path) of the .tflite inside
     *                                  context.filesDir/klein_single0/
     *   - "klein_token_probe_tokens" : total token count (text + image) to use for synthetic inputs
     *
     * Example adb command for 512 tokens:
     *   adb shell am force-stop com.pockettavern.app
     *   adb shell am start -n com.pockettavern.app/.MainActivity \
     *       --ez run_klein_token_probe true \
     *       --es klein_token_probe_file single0_full_T512tok_Google_Tensor_G5.tflite \
     *       --ei klein_token_probe_tokens 512
     *
     * Checks: does it crash the NPU, and is the output finite/non-zero?
     * Logs as: KLEIN_TOKEN_PROBE[<tokenCount>]: runMs=... hasNanOrInf=... allZero=... ...
     */
    fun runKleinTokenProbe(context: Context, fileName: String, tokenCount: Int) {
        val dir = File(context.filesDir, "klein_single0")
        val modelFile = File(dir, fileName)
        if (!modelFile.exists()) {
            Log.w(TAG, "klein_token_probe: $modelFile not found, skipping")
            return
        }

        val hiddenSize = 3072
        val ropeComplexPairs = 64 // head_dim=128 -> 64 complex pairs

        Log.i(TAG, "=== Klein token probe: file=$fileName tokens=$tokenCount ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val model = CompiledModel.create(modelFile.absolutePath, options, env)
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()

            // x: [1, tokenCount, hiddenSize]
            inputs[0].writeFloat(FloatArray(1 * tokenCount * hiddenSize) { i -> kotlin.math.sin(i.toFloat()) * 0.1f })
            // pe: [1, 1, tokenCount, ropeComplexPairs, 2, 2]
            inputs[1].writeFloat(FloatArray(1 * 1 * tokenCount * ropeComplexPairs * 2 * 2) { i -> kotlin.math.sin(i.toFloat()) * 0.1f })
            // mod_shift, mod_scale, mod_gate: each [1, 1, hiddenSize]
            for (k in 2..4) {
                inputs[k].writeFloat(FloatArray(1 * 1 * hiddenSize) { i -> kotlin.math.sin(i.toFloat()) * 0.1f })
            }

            val start = System.currentTimeMillis()
            model.run(inputs, outputs)
            val elapsedMs = System.currentTimeMillis() - start

            val npuOut = outputs[0].readFloat()
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
            model.close()

            var hasNanOrInf = false
            var maxAbs = 0f
            var allZero = true
            for (v in npuOut) {
                if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                val a = kotlin.math.abs(v)
                if (a > maxAbs) maxAbs = a
                if (a > 1e-9f) allZero = false
            }
            // linear1 output size (largest intermediate) as reference for the DMA limit search
            val linear1MiB = tokenCount.toLong() * 9216L * 4L / (1024L * 1024L)
            Log.i(
                TAG,
                "KLEIN_TOKEN_PROBE[$tokenCount]: runMs=$elapsedMs hasNanOrInf=$hasNanOrInf " +
                    "maxAbs=$maxAbs allZero=$allZero outputSize=${npuOut.size} " +
                    "linear1_intermediate_MiB=$linear1MiB " +
                    "npuOut[0..4]=${npuOut.take(5)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== Klein token probe ($fileName tokens=$tokenCount): FAILED ===", e)
        }
    }

    /**
     * THROWAWAY diagnostic for the chunked/flash-attention pivot (docs/flux2-klein-conversion.md
     * step 7): does the `flash_step` kernel -- one dispatch of the online-softmax recurrence over a
     * single (Q-chunk, K/V-chunk) pair -- actually run on real Tensor G5 hardware without hitting
     * the DIVE STORE_ACCESS_FAULT or the superlinear performance cliff that motivated this pivot
     * away from one monolithic full-sequence SDPA dispatch?
     *
     * Signature matches scratch/klein_flash_step_export.py's `FlashStep` module exactly:
     *   inputs:  q_blk, k_blk, v_blk [1, 24, 1024, 128]; running_max, running_sum [1, 24, 1024, 1];
     *            running_out [1, 24, 1024, 128]
     *   outputs: running_max, running_sum, running_out (updated), same shapes as the state inputs
     *
     * Uses synthetic sine-wave inputs -- no reference bins needed, since this only checks whether
     * the NPU dispatch itself is sound (finite, non-zero, no crash), not numerical correctness
     * against the CPU reference (already proven in klein_flash_step_export.py / step 7's CPU work).
     *
     * Expects: context.filesDir/klein_single0/flash_step_probe_1024_Google_Tensor_G5.tflite
     */
    fun runKleinFlashStepProbe(context: Context) {
        val dir = File(context.filesDir, "klein_single0")
        val modelFile = File(dir, "flash_step_probe_1024_Google_Tensor_G5.tflite")
        if (!modelFile.exists()) {
            Log.w(TAG, "klein_single0/flash_step_probe_1024_Google_Tensor_G5.tflite not found in $dir, skipping")
            return
        }

        val chunk = 1024
        val heads = 24
        val headDim = 128

        Log.i(TAG, "=== Klein flash_step NPU diagnostic starting (chunk=$chunk) ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val model = CompiledModel.create(modelFile.absolutePath, options, env)
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()

            // q_blk, k_blk, v_blk: [1, heads, chunk, headDim]
            for (k in 0..2) {
                inputs[k].writeFloat(FloatArray(heads * chunk * headDim) { i -> kotlin.math.sin(i.toFloat()) * 0.1f })
            }
            // running_max, running_sum: [1, heads, chunk, 1]
            for (k in 3..4) {
                inputs[k].writeFloat(FloatArray(heads * chunk) { i -> kotlin.math.sin(i.toFloat()) * 0.1f })
            }
            // running_out: [1, heads, chunk, headDim]
            inputs[5].writeFloat(FloatArray(heads * chunk * headDim) { i -> kotlin.math.sin(i.toFloat()) * 0.1f })

            val start = System.currentTimeMillis()
            model.run(inputs, outputs)
            val elapsedMs = System.currentTimeMillis() - start

            val newMax = outputs[0].readFloat()
            val newSum = outputs[1].readFloat()
            val newOut = outputs[2].readFloat()
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
            model.close()

            fun summarize(name: String, arr: FloatArray): String {
                var hasNanOrInf = false
                var maxAbs = 0f
                var allZero = true
                for (v in arr) {
                    if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                    val a = kotlin.math.abs(v)
                    if (a > maxAbs) maxAbs = a
                    if (a > 1e-9f) allZero = false
                }
                return "$name(hasNanOrInf=$hasNanOrInf maxAbs=$maxAbs allZero=$allZero size=${arr.size})"
            }

            Log.i(
                TAG,
                "KLEIN_FLASH_STEP: runMs=$elapsedMs " +
                    "${summarize("running_max", newMax)} ${summarize("running_sum", newSum)} " +
                    "${summarize("running_out", newOut)} newOut[0..4]=${newOut.take(5)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== Klein flash_step NPU diagnostic: FAILED ===", e)
        }
    }

    /**
     * THROWAWAY diagnostic for pass 1 of the chunked-block dispatch design
     * (docs/flux2-klein-conversion.md step 7): does the QKV-projection piece (LayerNorm + linear1 +
     * QKNorm + RoPE, run on one 1024-token query-chunk) run on real Tensor G5 hardware?
     *
     * Signature matches scratch/klein_qkv_proj_export.py's `QkvProjChunk` module exactly:
     *   inputs:  x_chunk [1, 1024, 3072]; pe_chunk [1, 1, 1024, 64, 2, 2];
     *            mod_shift, mod_scale [1, 1, 3072]
     *   outputs: q, k, v [1, 24, 1024, 128]; mlp [1, 1024, 18432]
     *
     * Expects: context.filesDir/klein_single0/qkv_proj_probe_Google_Tensor_G5.tflite
     */
    fun runKleinQkvProjProbe(context: Context) {
        val dir = File(context.filesDir, "klein_single0")
        val modelFile = File(dir, "qkv_proj_probe_Google_Tensor_G5.tflite")
        if (!modelFile.exists()) {
            Log.w(TAG, "klein_single0/qkv_proj_probe_Google_Tensor_G5.tflite not found in $dir, skipping")
            return
        }

        val chunk = 1024
        val hiddenSize = 3072
        val ropeComplexPairs = 64

        Log.i(TAG, "=== Klein qkv_proj NPU diagnostic starting (chunk=$chunk) ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val model = CompiledModel.create(modelFile.absolutePath, options, env)
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()

            // x_chunk: [1, chunk, hiddenSize]
            inputs[0].writeFloat(FloatArray(chunk * hiddenSize) { i -> kotlin.math.sin(i.toFloat()) * 0.1f })
            // pe_chunk: [1, 1, chunk, ropeComplexPairs, 2, 2]
            inputs[1].writeFloat(FloatArray(chunk * ropeComplexPairs * 2 * 2) { i -> kotlin.math.sin(i.toFloat()) * 0.1f })
            // mod_shift, mod_scale: [1, 1, hiddenSize]
            for (k in 2..3) {
                inputs[k].writeFloat(FloatArray(hiddenSize) { i -> kotlin.math.sin(i.toFloat()) * 0.1f })
            }

            val start = System.currentTimeMillis()
            model.run(inputs, outputs)
            val elapsedMs = System.currentTimeMillis() - start

            val outArrays = outputs.map { it.readFloat() }
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
            model.close()

            fun summarize(name: String, arr: FloatArray): String {
                var hasNanOrInf = false
                var maxAbs = 0f
                var allZero = true
                for (v in arr) {
                    if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                    val a = kotlin.math.abs(v)
                    if (a > maxAbs) maxAbs = a
                    if (a > 1e-9f) allZero = false
                }
                return "$name(hasNanOrInf=$hasNanOrInf maxAbs=$maxAbs allZero=$allZero size=${arr.size})"
            }

            val names = listOf("q", "k", "v", "mlp")
            val summaries = names.zip(outArrays).joinToString(" ") { (n, a) -> summarize(n, a) }
            Log.i(TAG, "KLEIN_QKV_PROJ: runMs=$elapsedMs $summaries")
        } catch (e: Throwable) {
            Log.e(TAG, "=== Klein qkv_proj NPU diagnostic: FAILED ===", e)
        }
    }

    /**
     * THROWAWAY diagnostic for pass 3 of the chunked-block dispatch design
     * (docs/flux2-klein-conversion.md step 7): does the output-projection piece (linear2 over
     * concat(attn, SiLU(mlp)) + gated residual, run on one 1024-token query-chunk) run on real
     * Tensor G5 hardware?
     *
     * Signature matches scratch/klein_out_proj_export.py's `OutProjChunk` module exactly:
     *   inputs:  x_chunk, attn_chunk [1, 1024, 3072]; mlp_chunk [1, 1024, 18432]; mod_gate [1, 1, 3072]
     *   outputs: out [1, 1024, 3072]
     *
     * Expects: context.filesDir/klein_single0/out_proj_probe_Google_Tensor_G5.tflite
     */
    fun runKleinOutProjProbe(context: Context) {
        val dir = File(context.filesDir, "klein_single0")
        val modelFile = File(dir, "out_proj_probe_Google_Tensor_G5.tflite")
        if (!modelFile.exists()) {
            Log.w(TAG, "klein_single0/out_proj_probe_Google_Tensor_G5.tflite not found in $dir, skipping")
            return
        }

        val chunk = 1024
        val hiddenSize = 3072
        val mlpHiddenDim = 18432

        Log.i(TAG, "=== Klein out_proj NPU diagnostic starting (chunk=$chunk) ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            val model = CompiledModel.create(modelFile.absolutePath, options, env)
            val inputs = model.createInputBuffers()
            val outputs = model.createOutputBuffers()

            // x_chunk, attn_chunk: [1, chunk, hiddenSize]
            for (k in 0..1) {
                inputs[k].writeFloat(FloatArray(chunk * hiddenSize) { i -> kotlin.math.sin(i.toFloat()) * 0.1f })
            }
            // mlp_chunk: [1, chunk, mlpHiddenDim]
            inputs[2].writeFloat(FloatArray(chunk * mlpHiddenDim) { i -> kotlin.math.sin(i.toFloat()) * 0.1f })
            // mod_gate: [1, 1, hiddenSize]
            inputs[3].writeFloat(FloatArray(hiddenSize) { i -> kotlin.math.sin(i.toFloat()) * 0.1f })

            val start = System.currentTimeMillis()
            model.run(inputs, outputs)
            val elapsedMs = System.currentTimeMillis() - start

            val npuOut = outputs[0].readFloat()
            inputs.forEach { it.close() }
            outputs.forEach { it.close() }
            model.close()

            var hasNanOrInf = false
            var maxAbs = 0f
            var allZero = true
            for (v in npuOut) {
                if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                val a = kotlin.math.abs(v)
                if (a > maxAbs) maxAbs = a
                if (a > 1e-9f) allZero = false
            }
            Log.i(
                TAG,
                "KLEIN_OUT_PROJ: runMs=$elapsedMs hasNanOrInf=$hasNanOrInf " +
                    "maxAbs=$maxAbs allZero=$allZero outputSize=${npuOut.size} " +
                    "npuOut[0..4]=${npuOut.take(5)}",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== Klein out_proj NPU diagnostic: FAILED ===", e)
        }
    }

    /**
     * THE real correctness test for the chunked/flash-attention pivot (docs/flux2-klein-conversion.md
     * step 7): chains all 3 piece-types (qkv_proj, flash_step, out_proj) together, exactly as the
     * NPU dispatch design intends, to compute one full single_blocks.0 forward at the real
     * production shape (4608 tokens) -- then diffs the assembled result against the real PyTorch
     * reference (`torch_out.bin`, same file `runKleinSingle0` already uses). Individually-clean
     * pieces (see runKleinQkvProjProbe / runKleinFlashStepProbe / runKleinOutProjProbe) don't prove
     * the *chained* result is correct -- this does.
     *
     * Uses chunk=1152 (4608 / 4, no ragged last chunk -- see scratch/klein_*_export.py --chunk 1152)
     * so this is the *first* on-device validation at chunk=1152, not just 1024; a clean result here
     * both proves the chaining logic and re-confirms chunk size isn't uniquely magic at 1024.
     *
     * Dispatch count: 4 qkv_proj + 4×4=16 flash_step + 4 out_proj = 24 CompiledModel runs, each
     * created/run/closed individually (same create->run->close lifetime discipline as every other
     * diagnostic in this file, not one persistent CompiledModel per piece-type).
     *
     * Expects in context.filesDir/klein_single0/:
     *   qkv_proj_probe_1152_Google_Tensor_G5.tflite, flash_step_probe_1152_Google_Tensor_G5.tflite,
     *   out_proj_probe_1152_Google_Tensor_G5.tflite, x.bin, pe.bin, mod_shift.bin, mod_scale.bin,
     *   mod_gate.bin, torch_out.bin (the last 6 are the same files runKleinSingle0 uses).
     */
    fun runKleinChunkedBlockProbe(context: Context) {
        val dir = File(context.filesDir, "klein_single0")
        val qkvModelFile = File(dir, "qkv_proj_probe_1152_Google_Tensor_G5.tflite")
        val flashModelFile = File(dir, "flash_step_probe_1152_Google_Tensor_G5.tflite")
        val flashInitModelFile = File(dir, "flash_step_init_probe_Google_Tensor_G5.tflite")
        val outModelFile = File(dir, "out_proj_probe_1152_Google_Tensor_G5.tflite")
        val xFile = File(dir, "x.bin")
        val peFile = File(dir, "pe.bin")
        val modShiftFile = File(dir, "mod_shift.bin")
        val modScaleFile = File(dir, "mod_scale.bin")
        val modGateFile = File(dir, "mod_gate.bin")
        val refFile = File(dir, "torch_out.bin")
        if (!qkvModelFile.exists() || !flashModelFile.exists() || !flashInitModelFile.exists() || !outModelFile.exists() ||
            !xFile.exists() || !peFile.exists() || !modShiftFile.exists() ||
            !modScaleFile.exists() || !modGateFile.exists() || !refFile.exists()
        ) {
            Log.w(TAG, "runKleinChunkedBlockProbe: model/input/reference files not all found in $dir, skipping")
            return
        }

        val tokenCount = 4608
        val chunk = 1152
        val numChunks = tokenCount / chunk
        val hiddenSize = 3072
        val ropeComplexPairs = 64
        val heads = 24
        val headDim = 128
        val peStride = ropeComplexPairs * 2 * 2

        // q/k/v/mlp/attn are all kept in memory now (with eager per-chunk release, same fix as
        // runKleinDoubleChunkedBlockProbe). mlp alone is 4x85 MiB =~ 340 MiB if all chunks were held
        // simultaneously -- previously this exceeded the default ~256 MiB Dalvik heap and forced
        // mlp to stay on disk; with `android:largeHeap="true"` now enabled (2026-08-29, see
        // AndroidManifest.xml), the full in-memory design fits and disk I/O is eliminated entirely.
        Log.i(TAG, "=== Klein chunked-block NPU diagnostic starting (chunk=$chunk, numChunks=$numChunks) ===")
        lateinit var qkvModel: CompiledModel
        lateinit var flashModel: CompiledModel
        lateinit var flashInitModel: CompiledModel
        lateinit var outModel: CompiledModel
        var tmpDir: File? = null
        val createdModels = ArrayList<CompiledModel>(4)
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            fun open(file: File): CompiledModel =
                CompiledModel.create(file.absolutePath, options, env).also { createdModels.add(it) }

            qkvModel = open(qkvModelFile)
            flashModel = open(flashModelFile)
            flashInitModel = open(flashInitModelFile)
            outModel = open(outModelFile)

            fun runPiece(model: CompiledModel, vararg inputArrays: FloatArray): List<FloatArray> {
                val inputs = model.createInputBuffers()
                val outputs = model.createOutputBuffers()
                inputArrays.forEachIndexed { i, arr -> inputs[i].writeFloat(arr) }
                model.run(inputs, outputs)
                val result = outputs.map { it.readFloat() }
                inputs.forEach { it.close() }
                outputs.forEach { it.close() }
                return result
            }

            // Reads floatCount floats starting at floatOffset (not byte offset) from a raw float32
            // bin, without loading the rest of the file -- avoids holding the full 4608-token x/pe/
            // reference tensors (each tens of MB) in the Dalvik heap at once.
            fun readFloatRange(file: File, floatOffset: Long, floatCount: Int): FloatArray {
                java.io.RandomAccessFile(file, "r").use { raf ->
                    raf.seek(floatOffset * 4)
                    val bytes = ByteArray(floatCount * 4)
                    raf.readFully(bytes)
                    val out = FloatArray(floatCount)
                    java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
                    return out
                }
            }

            // mlp stays on disk: all 4 chunks resident simultaneously would be ~340 MiB, and that's
            // unavoidably concurrent with k/v needing to stay resident through all of pass 2
            // (~113 MiB) plus attn (~57 MiB) -- ~510 MiB total, which OOM'd even with
            // android:largeHeap="true" doubling the ceiling to 512 MiB (confirmed empirically,
            // 2026-08-29: "Failed to allocate a 84934672 byte allocation ... growth limit
            // 536870912"). Unlike q/k/v/attn (which fit even at chunk=1152), mlp's own total size is
            // the hard constraint here, not a fixable lifetime/ordering issue -- keeping it on disk
            // is the practical optimum for this chunk size, not a regression.
            fun writeFloatBin(file: File, data: FloatArray) {
                java.io.DataOutputStream(java.io.BufferedOutputStream(java.io.FileOutputStream(file))).use { out ->
                    for (v in data) out.writeFloat(v)
                }
            }

            fun readFloatBinStreamed(file: File): FloatArray {
                val count = (file.length() / 4).toInt()
                val out = FloatArray(count)
                java.io.DataInputStream(java.io.BufferedInputStream(java.io.FileInputStream(file))).use { input ->
                    for (i in 0 until count) out[i] = input.readFloat()
                }
                return out
            }

            tmpDir = File(context.cacheDir, "klein_chunked_block_tmp").apply { deleteRecursively(); mkdirs() }

            val modShift = readFloatBin(modShiftFile)
            val modScale = readFloatBin(modScaleFile)
            val modGate = readFloatBin(modGateFile)

            // q/k/v/attn kept in memory; mlp stays on disk (see note above).
            val qChunks = arrayOfNulls<FloatArray>(numChunks)
            val kChunks = arrayOfNulls<FloatArray>(numChunks)
            val vChunks = arrayOfNulls<FloatArray>(numChunks)
            val attnChunks = arrayOfNulls<FloatArray>(numChunks)

            val startTotal = System.currentTimeMillis()

            // Pass 1: QKV projection, one dispatch per query-chunk.
            for (c in 0 until numChunks) {
                val xChunk = readFloatRange(xFile, c.toLong() * chunk * hiddenSize, chunk * hiddenSize)
                val peChunk = readFloatRange(peFile, c.toLong() * chunk * peStride, chunk * peStride)
                val result = runPiece(qkvModel, xChunk, peChunk, modShift, modScale)
                if (c == 0) {
                    Log.i(
                        TAG,
                        "KLEIN_CHUNKED_BLOCK_DEBUG: q0[0:5]=${result[0].take(5)} k0[0:5]=${result[1].take(5)} " +
                            "v0[0:5]=${result[2].take(5)} mlp0[0:5]=${result[3].take(5)}",
                    )
                }
                qChunks[c] = result[0]
                kChunks[c] = result[1]
                vChunks[c] = result[2]
                writeFloatBin(File(tmpDir!!, "mlp_$c.bin"), result[3])
            }

            // Pass 2: flash attention -- one dispatch per (Q-chunk, K/V-chunk) pair, running-softmax
            // state threaded through host-side, exactly the design from klein_flash_step_prototype.py.
            val stateSize = heads * chunk
            val qkvSize = heads * chunk * headDim
            for (qi in 0 until numChunks) {
                val qChunk = qChunks[qi]!!
                // ki=0 uses flash_step_init (no external running-state input at all) instead of
                // flash_step with an artificial "empty" sentinel state. Root cause (2026-08-29):
                // feeding flash_step a sentinel initial state (-Infinity, then -1e30, then -30000 --
                // all identical wrong results) produced numerically WRONG output specifically after
                // Tensor G5 AOT compilation, even though the exact same recurrence on the exact same
                // real data was proven correct via desktop XNNPACK on the *uncompiled* exported
                // graph -- an AOT/NPU-compiler bug tied to this kernel+sentinel pattern, not a math
                // or wiring error. flash_step_init sidesteps it entirely by never constructing that
                // state: it computes the first K/V chunk's local softmax directly (see
                // klein_flash_step_init_export.py). flash_step (with the max()/correction merge) is
                // only used for ki=1..N-1, where runningMax is always genuine data from a real
                // previous chunk, never a sentinel.
                val init = runPiece(flashInitModel, qChunk, kChunks[0]!!, vChunks[0]!!)
                var runningMax = init[0]
                var runningSum = init[1]
                var runningOut = init[2]
                for (ki in 1 until numChunks) {
                    val result = runPiece(flashModel, qChunk, kChunks[ki]!!, vChunks[ki]!!, runningMax, runningSum, runningOut)
                    runningMax = result[0]
                    runningSum = result[1]
                    runningOut = result[2]
                }
                // Normalize (host-side, cheap) then transpose+reshape [1,H,chunk,D] -> [1,chunk,H*D]
                // to match _out's expected attn layout (same as SingleBlockAttention's eager path).
                val attn = FloatArray(chunk * hiddenSize)
                for (h in 0 until heads) {
                    for (t in 0 until chunk) {
                        val denom = runningSum[h * chunk + t]
                        for (d in 0 until headDim) {
                            attn[t * hiddenSize + h * headDim + d] = runningOut[(h * chunk + t) * headDim + d] / denom
                        }
                    }
                }
                if (qi == 0) {
                    Log.i(TAG, "KLEIN_CHUNKED_BLOCK_DEBUG: attn0[0:5]=${attn.take(5)}")
                }
                attnChunks[qi] = attn
                // Free this chunk's q as soon as its own outer iteration is done (only ever needed
                // within this one iteration, unlike k/v which every qi needs) -- same fix as
                // runKleinDoubleChunkedBlockProbe's OOM.
                qChunks[qi] = null
            }
            // k/v no longer needed once pass 2 completes (pass 3 only needs attnChunks + mlp).
            for (i in 0 until numChunks) {
                kChunks[i] = null
                vChunks[i] = null
            }

            // Pass 3: output projection, one dispatch per query-chunk. Diff each chunk against the
            // reference incrementally instead of materializing the full 4608-token output/reference.
            var maxAbsDiff = 0f
            var sumAbsDiff = 0.0
            var sumAbsRef = 0.0
            var hasNanOrInf = false
            var firstFive: List<Float>? = null
            var firstFiveRef: List<Float>? = null
            for (c in 0 until numChunks) {
                val xChunk = readFloatRange(xFile, c.toLong() * chunk * hiddenSize, chunk * hiddenSize)
                val attnChunk = attnChunks[c]!!
                val mlpChunk = readFloatBinStreamed(File(tmpDir!!, "mlp_$c.bin"))
                val result = runPiece(outModel, xChunk, attnChunk, mlpChunk, modGate)
                val outChunk = result[0]
                val refChunk = readFloatRange(refFile, c.toLong() * chunk * hiddenSize, chunk * hiddenSize)
                if (c == 0) {
                    firstFive = outChunk.take(5)
                    firstFiveRef = refChunk.take(5)
                }
                var chunkMaxAbsDiff = 0f
                var chunkSumAbsDiff = 0.0
                var chunkMaxAbsDiffIdx = -1
                for (i in outChunk.indices) {
                    val v = outChunk[i]
                    if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                    val diff = kotlin.math.abs(v - refChunk[i])
                    if (diff > maxAbsDiff) maxAbsDiff = diff
                    if (diff > chunkMaxAbsDiff) {
                        chunkMaxAbsDiff = diff
                        chunkMaxAbsDiffIdx = i
                    }
                    sumAbsDiff += diff
                    chunkSumAbsDiff += diff
                    sumAbsRef += kotlin.math.abs(refChunk[i])
                }
                val tokenOfMax = if (chunkMaxAbsDiffIdx >= 0) chunkMaxAbsDiffIdx / hiddenSize else -1
                Log.i(
                    TAG,
                    "KLEIN_CHUNKED_BLOCK_DEBUG: chunk=$c chunkMaxAbsDiff=$chunkMaxAbsDiff " +
                        "chunkMeanAbsDiff=${chunkSumAbsDiff / outChunk.size} atLocalToken=$tokenOfMax " +
                        "npuAtMax=${outChunk.getOrNull(chunkMaxAbsDiffIdx)} refAtMax=${refChunk.getOrNull(chunkMaxAbsDiffIdx)}",
                )
            }

            val elapsedTotalMs = System.currentTimeMillis() - startTotal
            tmpDir?.deleteRecursively()

            val totalElems = tokenCount.toLong() * hiddenSize
            val meanAbsDiff = sumAbsDiff / totalElems
            val meanAbsRef = sumAbsRef / totalElems
            Log.i(
                TAG,
                "KLEIN_CHUNKED_BLOCK: totalMs=$elapsedTotalMs numChunks=$numChunks dispatches=${numChunks + numChunks * numChunks + numChunks} " +
                    "hasNanOrInf=$hasNanOrInf maxAbsDiff=$maxAbsDiff meanAbsDiff=$meanAbsDiff meanAbsRef=$meanAbsRef " +
                    "npuOut[0..4]=$firstFive ref[0..4]=$firstFiveRef",
            )
        } catch (e: Throwable) {
            tmpDir?.deleteRecursively()
            Log.e(TAG, "=== Klein chunked-block NPU diagnostic: FAILED ===", e)
        } finally {
            createdModels.forEach { it.close() }
        }
    }

    /**
     * Correctness test for the DOUBLE-stream chunked dispatch design
     * (docs/flux2-klein-conversion.md step 8): chains all 99 dispatches (9 QKV-proj + 9x9=81
     * flash-attention + 9 out-proj) into one full `double_blocks.0` forward at the real production
     * shape, then diffs against the real PyTorch reference. Mirrors [runKleinChunkedBlockProbe]'s
     * structure and lessons (streamed temp files to bound Java heap, flash_step_init for the first
     * K/V chunk to avoid the sentinel-state AOT-compiler bug) but with 9 UNEVEN-origin chunks: index
     * 0 is the whole 512-token txt stream (its own qkv/out-proj weights), indices 1..8 are 512-token
     * img chunks (a different set of weights, reused across all 8). This uniform chunk=512 avoids
     * needing multiple compiled flash_step shapes for mismatched txt/img chunk sizes -- see
     * scratch/klein_double0_probe.py.
     *
     * Expects in context.filesDir/klein_single0/: double0_img_qkv_proj_probe_Google_Tensor_G5.tflite,
     * double0_txt_qkv_proj_probe_Google_Tensor_G5.tflite, double0_img_out_proj_probe_Google_Tensor_G5
     * .tflite, double0_txt_out_proj_probe_Google_Tensor_G5.tflite,
     * flash_step_probe_512_Google_Tensor_G5.tflite, flash_step_init_probe_512_Google_Tensor_G5.tflite,
     * and the real reference IO (img, txt, pe, pe_ctx, the various mod bins, img_out.bin,
     * txt_out.bin) -- see
     * scratch/push_klein_double0_chunked_block.sh.
     */
    fun runKleinDoubleChunkedBlockProbe(context: Context) {
        val dir = File(context.filesDir, "klein_single0")
        // img chunked at 1024 (not 512) to cut dispatch count: 5 total chunks (1 txt@512 + 4 img
        // @1024) instead of 9, and 25 flash-attention dispatches (5x5) instead of 81 (9x9) -- see
        // docs/flux2-klein-conversion.md's dispatch-count-vs-overhead discussion. Needs a small set
        // of ASYMMETRIC flash_step/flash_step_init shapes since txt (Q or K/V = 512) and img (Q or
        // K/V = 1024) chunks now differ in size -- klein_flash_step_export.py /
        // klein_flash_step_init_export.py gained --q-chunk/--kv-chunk for this.
        val imgQkvModelFile = File(dir, "double0_img_qkv_proj_probe_1024_Google_Tensor_G5.tflite")
        val txtQkvModelFile = File(dir, "double0_txt_qkv_proj_probe_Google_Tensor_G5.tflite")
        val imgOutModelFile = File(dir, "double0_img_out_proj_probe_1024_Google_Tensor_G5.tflite")
        val txtOutModelFile = File(dir, "double0_txt_out_proj_probe_Google_Tensor_G5.tflite")
        // flash_step_init KV is always chunk 0 (txt, 512) -- two Q-size variants.
        val flashInitQ512Kv512ModelFile = File(dir, "flash_step_init_probe_512_Google_Tensor_G5.tflite")
        val flashInitQ1024Kv512ModelFile = File(dir, "flash_step_init_probe_q1024_kv512_Google_Tensor_G5.tflite")
        // flash_step KV is always an img chunk (1024) for ki=1..N-1 -- two Q-size variants.
        val flashQ512Kv1024ModelFile = File(dir, "flash_step_probe_q512_kv1024_Google_Tensor_G5.tflite")
        val flashQ1024Kv1024ModelFile = File(dir, "flash_step_probe_1024_Google_Tensor_G5.tflite")

        val refNames = listOf(
            "img", "txt", "pe", "pe_ctx",
            "img_mod1_shift", "img_mod1_scale", "img_mod1_gate",
            "img_mod2_shift", "img_mod2_scale", "img_mod2_gate",
            "txt_mod1_shift", "txt_mod1_scale", "txt_mod1_gate",
            "txt_mod2_shift", "txt_mod2_scale", "txt_mod2_gate",
            "img_out", "txt_out",
        )
        val refFiles = refNames.associateWith { File(dir, "$it.bin") }
        val modelFiles = listOf(
            imgQkvModelFile, txtQkvModelFile, imgOutModelFile, txtOutModelFile,
            flashInitQ512Kv512ModelFile, flashInitQ1024Kv512ModelFile, flashQ512Kv1024ModelFile, flashQ1024Kv1024ModelFile,
        )
        if (modelFiles.any { !it.exists() } || refFiles.values.any { !it.exists() }) {
            Log.w(
                TAG,
                "runKleinDoubleChunkedBlockProbe: model/input/reference files not all found in $dir " +
                    "(missing: ${modelFiles.filter { !it.exists() }.map { it.name }}), skipping",
            )
            return
        }

        val txtTokens = 512
        val imgTokens = 4096
        val imgChunkSize = 1024
        val numImgChunks = imgTokens / imgChunkSize
        val numChunks = 1 + numImgChunks // chunk 0 = txt@512, 1..numImgChunks = img@1024
        val chunkSizes = IntArray(numChunks) { if (it == 0) txtTokens else imgChunkSize }
        val hiddenSize = 3072
        val ropeComplexPairs = 64
        val heads = 24
        val headDim = 128
        val peStride = ropeComplexPairs * 2 * 2

        Log.i(TAG, "=== Klein DOUBLE chunked-block NPU diagnostic starting (numChunks=$numChunks, chunkSizes=${chunkSizes.toList()}) ===")
        // Persistent CompiledModel instances, created once and reused across every dispatch to that
        // same piece (not create->run->close per call). Per-piece profiling
        // (runKleinProfileDispatchOverhead) measured a real ~20-40% per-dispatch speedup from this,
        // mostly from a lower model.run() time itself, not just skipping create(). Total resident
        // model size here (~58+190+58+190+~6 MiB =~ 500 MiB) is well within budget now that pieces
        // are already small/chunked for the dispatch-count-reduction work above -- unlike the SDXL
        // create->run->close-per-dispatch lesson, which was about a single much-larger dmabuf, not
        // several small/medium models held open simultaneously.
        lateinit var imgQkvModel: CompiledModel
        lateinit var txtQkvModel: CompiledModel
        lateinit var imgOutModel: CompiledModel
        lateinit var txtOutModel: CompiledModel
        lateinit var flashInitQ512Kv512Model: CompiledModel
        lateinit var flashInitQ1024Kv512Model: CompiledModel
        lateinit var flashQ512Kv1024Model: CompiledModel
        lateinit var flashQ1024Kv1024Model: CompiledModel
        val createdModels = ArrayList<CompiledModel>(8)
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            fun open(file: File): CompiledModel =
                CompiledModel.create(file.absolutePath, options, env).also { createdModels.add(it) }

            imgQkvModel = open(imgQkvModelFile)
            txtQkvModel = open(txtQkvModelFile)
            imgOutModel = open(imgOutModelFile)
            txtOutModel = open(txtOutModelFile)
            flashInitQ512Kv512Model = open(flashInitQ512Kv512ModelFile)
            flashInitQ1024Kv512Model = open(flashInitQ1024Kv512ModelFile)
            flashQ512Kv1024Model = open(flashQ512Kv1024ModelFile)
            flashQ1024Kv1024Model = open(flashQ1024Kv1024ModelFile)

            fun runPiece(model: CompiledModel, vararg inputArrays: FloatArray): List<FloatArray> {
                val inputs = model.createInputBuffers()
                val outputs = model.createOutputBuffers()
                inputArrays.forEachIndexed { i, arr -> inputs[i].writeFloat(arr) }
                model.run(inputs, outputs)
                val result = outputs.map { it.readFloat() }
                inputs.forEach { it.close() }
                outputs.forEach { it.close() }
                return result
            }

            fun readFloatRange(file: File, floatOffset: Long, floatCount: Int): FloatArray {
                java.io.RandomAccessFile(file, "r").use { raf ->
                    raf.seek(floatOffset * 4)
                    val bytes = ByteArray(floatCount * 4)
                    raf.readFully(bytes)
                    val out = FloatArray(floatCount)
                    java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
                    return out
                }
            }

            val mods = refNames.filter { it.contains("mod") }.associateWith { readFloatBin(refFiles.getValue(it)) }

            // Intermediate q/k/v/attn chunks kept in memory (not round-tripped through disk like the
            // earlier chunk=512-uniform version needed) -- double-stream's out_proj computes its MLP
            // internally (unlike single-stream, which has a separate ~85 MiB mlp intermediate per
            // chunk), so total resident size here is just q+k+v+attn across 5 chunks (~220 MiB),
            // comfortably within the default ~256 MiB Dalvik heap. This tests whether eliminating the
            // per-float DataOutputStream/DataInputStream round-trip (suspected dominant cost per the
            // profiling update above) meaningfully speeds up the chain beyond the ~5% persistent-
            // model-reuse alone gave.
            val qChunks = arrayOfNulls<FloatArray>(numChunks)
            val kChunks = arrayOfNulls<FloatArray>(numChunks)
            val vChunks = arrayOfNulls<FloatArray>(numChunks)
            val attnChunks = arrayOfNulls<FloatArray>(numChunks)

            val startTotal = System.currentTimeMillis()

            // Pass 1: QKV projection. Chunk 0 = whole txt stream (txt weights + pe_ctx, chunk=512);
            // chunks 1..numImgChunks = img chunks (img weights + pe, chunk=1024 each).
            run {
                val txtChunk = readFloatRange(refFiles.getValue("txt"), 0, txtTokens * hiddenSize)
                val peCtxChunk = readFloatRange(refFiles.getValue("pe_ctx"), 0, txtTokens * peStride)
                val result = runPiece(
                    txtQkvModel, txtChunk, peCtxChunk,
                    mods.getValue("txt_mod1_shift"), mods.getValue("txt_mod1_scale"),
                )
                qChunks[0] = result[0]
                kChunks[0] = result[1]
                vChunks[0] = result[2]
            }
            for (ic in 0 until numImgChunks) {
                val c = ic + 1
                val imgChunk = readFloatRange(refFiles.getValue("img"), ic.toLong() * imgChunkSize * hiddenSize, imgChunkSize * hiddenSize)
                val peChunk = readFloatRange(refFiles.getValue("pe"), ic.toLong() * imgChunkSize * peStride, imgChunkSize * peStride)
                val result = runPiece(
                    imgQkvModel, imgChunk, peChunk,
                    mods.getValue("img_mod1_shift"), mods.getValue("img_mod1_scale"),
                )
                qChunks[c] = result[0]
                kChunks[c] = result[1]
                vChunks[c] = result[2]
            }

            // Pass 2: flash attention over the combined 5-chunk (txt+img) K/V list. K/V for ki=0 is
            // always txt (512); for ki=1..N-1 always img (1024) -- so flash_step_init only varies by
            // Q size (KV fixed at 512), and flash_step only varies by Q size too (KV fixed at 1024).
            for (qi in 0 until numChunks) {
                val tq = chunkSizes[qi]
                val qChunk = qChunks[qi]!!
                val flashInitModel = if (tq == txtTokens) flashInitQ512Kv512Model else flashInitQ1024Kv512Model
                val init = runPiece(flashInitModel, qChunk, kChunks[0]!!, vChunks[0]!!)
                var runningMax = init[0]
                var runningSum = init[1]
                var runningOut = init[2]
                val flashModel = if (tq == txtTokens) flashQ512Kv1024Model else flashQ1024Kv1024Model
                for (ki in 1 until numChunks) {
                    val result = runPiece(flashModel, qChunk, kChunks[ki]!!, vChunks[ki]!!, runningMax, runningSum, runningOut)
                    runningMax = result[0]
                    runningSum = result[1]
                    runningOut = result[2]
                }
                val attn = FloatArray(tq * hiddenSize)
                for (h in 0 until heads) {
                    for (t in 0 until tq) {
                        val denom = runningSum[h * tq + t]
                        for (d in 0 until headDim) {
                            attn[t * hiddenSize + h * headDim + d] = runningOut[(h * tq + t) * headDim + d] / denom
                        }
                    }
                }
                attnChunks[qi] = attn
                // Free this chunk's q as soon as its own outer iteration is done -- q is only ever
                // needed within the single qi iteration that produced it, unlike k/v (needed by
                // every qi). Keeping all 5 q chunks resident for the whole pass peaked memory right
                // at the ~256 MiB Dalvik heap ceiling (OutOfMemoryError observed); releasing each one
                // immediately after use keeps at most one "extra" q chunk (<=12.6 MiB) resident.
                qChunks[qi] = null
            }
            // k/v are no longer needed once pass 2 is done (pass 3 only uses attnChunks) -- release
            // them too rather than holding ~113 MiB dead weight into pass 3.
            for (i in 0 until numChunks) {
                kChunks[i] = null
                vChunks[i] = null
            }

            // Pass 3: output projection. Chunk 0 -> txt_out (whole stream, 1 dispatch). Chunks
            // 1..numImgChunks -> img_out pieces, diffed incrementally against the real reference.
            var hasNanOrInf = false
            var maxAbsDiffImg = 0f
            var sumAbsDiffImg = 0.0
            var sumAbsRefImg = 0.0
            var maxAbsDiffTxt = 0f
            var sumAbsDiffTxt = 0.0
            var sumAbsRefTxt = 0.0

            run {
                val txtChunk = readFloatRange(refFiles.getValue("txt"), 0, txtTokens * hiddenSize)
                val attnChunk = attnChunks[0]!!
                val result = runPiece(
                    txtOutModel, txtChunk, attnChunk,
                    mods.getValue("txt_mod1_gate"), mods.getValue("txt_mod2_shift"),
                    mods.getValue("txt_mod2_scale"), mods.getValue("txt_mod2_gate"),
                )
                val txtOutChunk = result[0]
                // txt_out.bin is little-endian (desktop numpy .tofile()) -- must use the class's
                // little-endian readFloatBin (this is real reference data read fresh from disk each
                // time, not one of the in-memory intermediate chunks above).
                val refChunk = readFloatBin(refFiles.getValue("txt_out"))
                fun stats(name: String, arr: FloatArray): String {
                    var mx = 0f; var mn = Float.MAX_VALUE; var nanInf = false; var sum = 0.0
                    for (v in arr) {
                        if (v.isNaN() || v.isInfinite()) nanInf = true
                        val a = kotlin.math.abs(v)
                        if (a > mx) mx = a
                        if (a < mn) mn = a
                        sum += a
                    }
                    return "$name(maxAbs=$mx minAbs=$mn meanAbs=${sum / arr.size} nanInf=$nanInf size=${arr.size})"
                }
                Log.i(
                    TAG,
                    "KLEIN_DOUBLE_CHUNKED_BLOCK_DEBUG: ${stats("txtChunk", txtChunk)} ${stats("attnChunk0", attnChunk)} " +
                        "${stats("txtOutChunk", txtOutChunk)} ${stats("refChunk", refChunk)}",
                )
                for (i in txtOutChunk.indices) {
                    val v = txtOutChunk[i]
                    if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                    val diff = kotlin.math.abs(v - refChunk[i])
                    if (diff > maxAbsDiffTxt) maxAbsDiffTxt = diff
                    sumAbsDiffTxt += diff
                    sumAbsRefTxt += kotlin.math.abs(refChunk[i])
                }
            }
            for (ic in 0 until numImgChunks) {
                val c = ic + 1
                val imgChunk = readFloatRange(refFiles.getValue("img"), ic.toLong() * imgChunkSize * hiddenSize, imgChunkSize * hiddenSize)
                val attnChunk = attnChunks[c]!!
                val result = runPiece(
                    imgOutModel, imgChunk, attnChunk,
                    mods.getValue("img_mod1_gate"), mods.getValue("img_mod2_shift"),
                    mods.getValue("img_mod2_scale"), mods.getValue("img_mod2_gate"),
                )
                val imgOutChunk = result[0]
                val refChunk = readFloatRange(refFiles.getValue("img_out"), ic.toLong() * imgChunkSize * hiddenSize, imgChunkSize * hiddenSize)
                for (i in imgOutChunk.indices) {
                    val v = imgOutChunk[i]
                    if (v.isNaN() || v.isInfinite()) hasNanOrInf = true
                    val diff = kotlin.math.abs(v - refChunk[i])
                    if (diff > maxAbsDiffImg) maxAbsDiffImg = diff
                    sumAbsDiffImg += diff
                    sumAbsRefImg += kotlin.math.abs(refChunk[i])
                }
            }

            val elapsedTotalMs = System.currentTimeMillis() - startTotal

            val meanAbsDiffImg = sumAbsDiffImg / (imgTokens.toLong() * hiddenSize)
            val meanAbsRefImg = sumAbsRefImg / (imgTokens.toLong() * hiddenSize)
            val meanAbsDiffTxt = sumAbsDiffTxt / (txtTokens.toLong() * hiddenSize)
            val meanAbsRefTxt = sumAbsRefTxt / (txtTokens.toLong() * hiddenSize)
            Log.i(
                TAG,
                "KLEIN_DOUBLE_CHUNKED_BLOCK: totalMs=$elapsedTotalMs numChunks=$numChunks " +
                    "dispatches=${numChunks + numChunks * numChunks + numChunks} hasNanOrInf=$hasNanOrInf " +
                    "img: maxAbsDiff=$maxAbsDiffImg meanAbsDiff=$meanAbsDiffImg meanAbsRef=$meanAbsRefImg " +
                    "txt: maxAbsDiff=$maxAbsDiffTxt meanAbsDiff=$meanAbsDiffTxt meanAbsRef=$meanAbsRefTxt",
            )
        } catch (e: Throwable) {
            Log.e(TAG, "=== Klein DOUBLE chunked-block NPU diagnostic: FAILED ===", e)
        } finally {
            createdModels.forEach { it.close() }
        }
    }

    /**
     * THROWAWAY profiling diagnostic: how much of the chunked-block chain's per-dispatch wall-clock
     * cost (observed ~1.3-1.5s/dispatch for single-stream, ~360ms/dispatch for double-stream -- see
     * docs/flux2-klein-conversion.md's dispatch-count-vs-overhead discussion) is `CompiledModel
     * .create()` overhead specifically, vs. actual NPU `model.run()` compute? Compares two patterns
     * for repeated calls to the SAME compiled artifact:
     *   A) create -> run -> close, every single call (today's pattern, per the SDXL-derived
     *      create->run->close-to-bound-dmabuf-memory lesson)
     *   B) create ONCE, run N times (fresh input/output buffers per run, model stays open), close once
     *
     * Tests both a small/cheap piece (flash_step, called many times per block: 16-81x depending on
     * block type) and a large piece (qkv_proj, called several times per block: 4-9x), using synthetic
     * sin-wave inputs (timing only, correctness not in question here). N=8 iterations each, matching
     * roughly the reuse count of one double-stream img/txt piece.
     *
     * Expects in context.filesDir/klein_single0/: flash_step_probe_1152_Google_Tensor_G5.tflite,
     * qkv_proj_probe_1152_Google_Tensor_G5.tflite (both already pushed from the single-stream work).
     */
    fun runKleinProfileDispatchOverhead(context: Context) {
        val dir = File(context.filesDir, "klein_single0")
        val flashModel = File(dir, "flash_step_probe_1152_Google_Tensor_G5.tflite")
        val qkvModel = File(dir, "qkv_proj_probe_1152_Google_Tensor_G5.tflite")
        if (!flashModel.exists() || !qkvModel.exists()) {
            Log.w(TAG, "runKleinProfileDispatchOverhead: model files not found in $dir, skipping")
            return
        }

        val chunk = 1152
        val heads = 24
        val headDim = 128
        val hiddenSize = 3072
        val ropeComplexPairs = 64
        val iterations = 8

        Log.i(TAG, "=== Klein dispatch-overhead profiling starting (iterations=$iterations) ===")
        try {
            val npuProvider = BuiltinNpuAcceleratorProvider(context)
            val env = Environment.create(context, npuProvider)
            val options = CompiledModel.Options(Accelerator.NPU)

            fun sinArray(size: Int) = FloatArray(size) { i -> kotlin.math.sin(i.toFloat()) * 0.1f }

            // Precomputed ONCE, outside every timed region -- the previous attempt regenerated
            // these via repeated sin() calls *inside* the loop (e.g. flash_step's q/k/v/out arrays
            // are ~3.5M elements each), which could easily cost hundreds of ms of pure CPU time and
            // get misattributed to "NPU dispatch overhead". This run isolates create/write/run/
            // read/close from data generation.
            val flashQ = sinArray(heads * chunk * headDim)
            val flashK = sinArray(heads * chunk * headDim)
            val flashV = sinArray(heads * chunk * headDim)
            val flashMax = sinArray(heads * chunk)
            val flashSum = sinArray(heads * chunk)
            val flashOut = sinArray(heads * chunk * headDim)
            val qkvX = sinArray(chunk * hiddenSize)
            val qkvPe = sinArray(chunk * ropeComplexPairs * 2 * 2)
            val qkvModShift = sinArray(hiddenSize)
            val qkvModScale = sinArray(hiddenSize)

            // Separately timed phases within one dispatch: create, write-inputs, run, read-outputs,
            // close. Returns (createMs, writeMs, runMs, readMs, closeMs).
            fun timedFlashDispatch(model: CompiledModel): LongArray {
                var t = System.currentTimeMillis()
                val inputs = model.createInputBuffers()
                val outputs = model.createOutputBuffers()
                val bufMs = System.currentTimeMillis() - t
                t = System.currentTimeMillis()
                inputs[0].writeFloat(flashQ)
                inputs[1].writeFloat(flashK)
                inputs[2].writeFloat(flashV)
                inputs[3].writeFloat(flashMax)
                inputs[4].writeFloat(flashSum)
                inputs[5].writeFloat(flashOut)
                val writeMs = System.currentTimeMillis() - t
                t = System.currentTimeMillis()
                model.run(inputs, outputs)
                val runMs = System.currentTimeMillis() - t
                t = System.currentTimeMillis()
                outputs.forEach { it.readFloat() }
                val readMs = System.currentTimeMillis() - t
                t = System.currentTimeMillis()
                inputs.forEach { it.close() }
                outputs.forEach { it.close() }
                val closeMs = System.currentTimeMillis() - t
                return longArrayOf(bufMs, writeMs, runMs, readMs, closeMs)
            }

            fun timedQkvDispatch(model: CompiledModel): LongArray {
                var t = System.currentTimeMillis()
                val inputs = model.createInputBuffers()
                val outputs = model.createOutputBuffers()
                val bufMs = System.currentTimeMillis() - t
                t = System.currentTimeMillis()
                inputs[0].writeFloat(qkvX)
                inputs[1].writeFloat(qkvPe)
                inputs[2].writeFloat(qkvModShift)
                inputs[3].writeFloat(qkvModScale)
                val writeMs = System.currentTimeMillis() - t
                t = System.currentTimeMillis()
                model.run(inputs, outputs)
                val runMs = System.currentTimeMillis() - t
                t = System.currentTimeMillis()
                outputs.forEach { it.readFloat() }
                val readMs = System.currentTimeMillis() - t
                t = System.currentTimeMillis()
                inputs.forEach { it.close() }
                outputs.forEach { it.close() }
                val closeMs = System.currentTimeMillis() - t
                return longArrayOf(bufMs, writeMs, runMs, readMs, closeMs)
            }

            fun summarize(label: String, phaseTotals: LongArray, createTotalMs: Long) {
                Log.i(
                    TAG,
                    "KLEIN_PROFILE_OVERHEAD: $label createMs=$createTotalMs (${createTotalMs / iterations}/call) " +
                        "bufMs=${phaseTotals[0]} (${phaseTotals[0] / iterations}/call) " +
                        "writeMs=${phaseTotals[1]} (${phaseTotals[1] / iterations}/call) " +
                        "runMs=${phaseTotals[2]} (${phaseTotals[2] / iterations}/call) " +
                        "readMs=${phaseTotals[3]} (${phaseTotals[3] / iterations}/call) " +
                        "closeMs=${phaseTotals[4]} (${phaseTotals[4] / iterations}/call) " +
                        "totalMs=${createTotalMs + phaseTotals.sum()}",
                )
            }

            // Pattern A: create -> dispatch -> close, every iteration.
            run {
                val phaseTotals = LongArray(5)
                var createTotalMs = 0L
                repeat(iterations) {
                    val tc = System.currentTimeMillis()
                    val model = CompiledModel.create(flashModel.absolutePath, options, env)
                    createTotalMs += System.currentTimeMillis() - tc
                    val phases = timedFlashDispatch(model)
                    for (i in phases.indices) phaseTotals[i] += phases[i]
                    val tclose = System.currentTimeMillis()
                    model.close()
                    createTotalMs += System.currentTimeMillis() - tclose // model.close() folded into "create" bucket
                }
                summarize("flash_step[recreate-each-call]", phaseTotals, createTotalMs)
            }

            // Pattern B: create once, dispatch N times, close once.
            run {
                val tc = System.currentTimeMillis()
                val model = CompiledModel.create(flashModel.absolutePath, options, env)
                val createOnceMs = System.currentTimeMillis() - tc
                val phaseTotals = LongArray(5)
                repeat(iterations) {
                    val phases = timedFlashDispatch(model)
                    for (i in phases.indices) phaseTotals[i] += phases[i]
                }
                model.close()
                summarize("flash_step[reuse-model]", phaseTotals, createOnceMs)
            }

            // Same two patterns for the large qkv_proj piece.
            run {
                val phaseTotals = LongArray(5)
                var createTotalMs = 0L
                repeat(iterations) {
                    val tc = System.currentTimeMillis()
                    val model = CompiledModel.create(qkvModel.absolutePath, options, env)
                    createTotalMs += System.currentTimeMillis() - tc
                    val phases = timedQkvDispatch(model)
                    for (i in phases.indices) phaseTotals[i] += phases[i]
                    val tclose = System.currentTimeMillis()
                    model.close()
                    createTotalMs += System.currentTimeMillis() - tclose
                }
                summarize("qkv_proj[recreate-each-call]", phaseTotals, createTotalMs)
            }
            run {
                val tc = System.currentTimeMillis()
                val model = CompiledModel.create(qkvModel.absolutePath, options, env)
                val createOnceMs = System.currentTimeMillis() - tc
                val phaseTotals = LongArray(5)
                repeat(iterations) {
                    val phases = timedQkvDispatch(model)
                    for (i in phases.indices) phaseTotals[i] += phases[i]
                }
                model.close()
                summarize("qkv_proj[reuse-model]", phaseTotals, createOnceMs)
            }
        } catch (e: Throwable) {
            Log.e(TAG, "=== Klein dispatch-overhead profiling: FAILED ===", e)
        }
    }
}
