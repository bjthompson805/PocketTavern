package com.pockettavern.app.data.local.inference

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

/**
 * FLUX.2 [klein]'s model files are custom NPU-converted artifacts (~20GB total: an 8GB fp16 Qwen
 * text encoder + ~140 Tensor-G5-compiled `.tflite` transformer/VAE pieces) with no public host --
 * unlike [SdxlModelManager], there is no download-by-URL flow here. Files are staged manually via
 * `adb push` into the two fixed directories below (the same convention `NpuDiagnostic.kt`'s
 * existing Klein diagnostic calls already use), matching [SdxlModelManager]'s own precedent for
 * its NPU bundle directory ("not downloadable yet ... only reachable via manual adb push").
 *
 * Presence checks here are best-effort (matches [SdxlModelManager.hasNpuBundle]'s philosophy) --
 * real per-file validation happens natively when the pipeline actually tries to load a component.
 */
@Singleton
class KleinModelManager @Inject constructor(
    @ApplicationContext private val context: Context,
) {
    val qwenDir: File by lazy { File(context.filesDir, "flux2_klein_qwen") }
    val npuDir: File by lazy { File(context.filesDir, "flux2_klein_npu") }

    private fun isQwenComplete(): Boolean =
        QWEN_REQUIRED_FILES.all { File(qwenDir, it).let { f -> f.exists() && f.length() > 0 } }

    private fun isNpuPresent(): Boolean =
        npuDir.isDirectory && (npuDir.list()?.isNotEmpty() == true)

    fun isReady(): Boolean = isQwenComplete() && isNpuPresent()

    data class Status(
        val qwenReady: Boolean,
        val npuReady: Boolean,
        val qwenPath: String,
        val npuPath: String,
    )

    fun status(): Status = Status(
        qwenReady = isQwenComplete(),
        npuReady = isNpuPresent(),
        qwenPath = qwenDir.absolutePath,
        npuPath = npuDir.absolutePath,
    )

    companion object {
        // Matches the fp16 export's file set (config.json/llm.mnn/llm.mnn.weight/llm_config.json/
        // tokenizer.mtok) — see docs/flux2-klein-conversion.md's 2026-08-30 fp16-decision entry.
        val QWEN_REQUIRED_FILES = listOf(
            "config.json",
            "llm.mnn",
            "llm.mnn.weight",
            "llm_config.json",
            "tokenizer.mtok",
        )
    }
}
