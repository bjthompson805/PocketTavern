package com.pockettavern.app.data.local.inference

import android.content.Context
import android.util.Base64
import com.pockettavern.app.GenerationService
import com.pockettavern.app.util.DebugLogger
import com.pockettavern.app.util.OnDeviceImageGenerationScreenState
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * On-device FLUX.2 [klein] image generation: prompt -> on-device Qwen3-4B text encoding (CPU,
 * fp16) -> 4-step NPU denoising -> staged NPU VAE decode -> PNG, via [KleinDiffusionBridge].
 * Mirrors [MnnDiffusionEngine]'s shape closely, but with no persistent native handle -- every
 * call reloads everything (matches this pipeline's existing diagnostic architecture), so there is
 * no `ensureLoaded`/`unload` handle lifecycle to manage, only a generation mutex.
 *
 * Registers with [OnDeviceMemoryManager] under [OnDeviceMemoryManager.Slot.KLEIN] purely to force
 * other engines (SDXL/LLM) to unload before a run starts -- the ~8GB fp16 encoder plus NPU
 * buffers cannot coexist with another resident model. The unload callback is a no-op: nothing
 * from a Klein generation stays resident between calls, the native engine is stack-scoped inside
 * one blocking JNI call.
 */
@Singleton
class KleinDiffusionEngine @Inject constructor(
    @ApplicationContext private val context: Context,
    private val memoryManager: OnDeviceMemoryManager,
    private val modelManager: KleinModelManager,
) {
    sealed class Progress {
        object Started : Progress()
        data class Step(val percent: Int) : Progress()
        data class Done(val imageBase64: String) : Progress()
        data class Error(val message: String) : Progress()
    }

    private val genMutex = Mutex()

    init {
        memoryManager.register(OnDeviceMemoryManager.Slot.KLEIN) { /* nothing persists between calls */ }
    }

    /**
     * Generates one Klein image. Confirmed on real hardware: the text encoder alone takes
     * 120-350s depending on precision (fp16 in production, per
     * docs/flux2-klein-conversion.md's decision), plus the 4-step NPU denoise and staged VAE
     * decode -- expect several minutes total. Runs under [GenerationService] the whole time
     * (same reasoning as [MnnDiffusionEngine.generate]: keeps the CPU governor boosted and holds
     * a wake lock; its 10-minute wake-lock safety net is renewed by [heartbeat] every 5 minutes,
     * important here since a Klein run is likely to exceed 10 minutes).
     *
     * There is no native cancellation -- cancelling the returned [Flow] only detaches this side
     * from listening; the blocking native call keeps running to completion regardless.
     */
    fun generate(prompt: String, seed: Int): Flow<Progress> = callbackFlow {
        GenerationService.start(context, "Generating image on-device (FLUX.2 klein)…")

        val heartbeat = launch(Dispatchers.Default) {
            while (isActive) {
                delay(5 * 60 * 1000L)
                GenerationService.start(context, "Generating image on-device (FLUX.2 klein)…")
            }
        }

        val job = launch(Dispatchers.IO) {
            var outputFile: File? = null
            var keepsScreenOn = false
            try {
                genMutex.withLock {
                    trySend(Progress.Started)
                    if (!modelManager.isReady()) {
                        trySend(Progress.Error(
                            "FLUX.2 [klein] model files aren't staged -- see Settings for the " +
                                "required paths"
                        ))
                        return@withLock
                    }

                    val modelBytes = modelManager.qwenDir.walkTopDown().filter { it.isFile }.sumOf { it.length() }
                    memoryManager.prepareLoad(OnDeviceMemoryManager.Slot.KLEIN, modelBytes)
                    OnDeviceImageGenerationScreenState.begin()
                    keepsScreenOn = true

                    val file = File(context.cacheDir, "klein_gen_${System.currentTimeMillis()}.png")
                    outputFile = file
                    val actualSeed = if (seed < 0) (0..Int.MAX_VALUE).random() else seed

                    val mmapCacheDir = File(context.cacheDir, "klein_mmap").apply { mkdirs() }

                    DebugLogger.log("KleinDiffusionEngine: generating seed=$actualSeed, prompt=\"$prompt\"")
                    val ok = KleinDiffusionBridge.nativeGenerate(
                        modelManager.npuDir.absolutePath,
                        context.applicationInfo.nativeLibraryDir,
                        File(modelManager.qwenDir, "config.json").absolutePath,
                        mmapCacheDir.absolutePath,
                        prompt,
                        file.absolutePath,
                        actualSeed,
                    ) { percent ->
                        DebugLogger.log("KleinDiffusionEngine: progress=$percent%")
                        trySend(Progress.Step(percent))
                    }

                    if (!ok || !file.exists() || file.length() == 0L) {
                        trySend(Progress.Error("FLUX.2 [klein] generation failed; see logcat"))
                    } else {
                        val base64 = Base64.encodeToString(file.readBytes(), Base64.NO_WRAP)
                        trySend(Progress.Done(base64))
                    }
                }
            } catch (e: Exception) {
                DebugLogger.logError("KleinDiffusionEngine", "generation failed", e)
                trySend(Progress.Error(e.message ?: "On-device FLUX.2 [klein] generation failed"))
            } finally {
                if (keepsScreenOn) OnDeviceImageGenerationScreenState.end()
                outputFile?.delete()
                heartbeat.cancel()
                GenerationService.stop(context)
                close()
            }
        }

        awaitClose {
            job.cancel()
            heartbeat.cancel()
        }
    }
}
