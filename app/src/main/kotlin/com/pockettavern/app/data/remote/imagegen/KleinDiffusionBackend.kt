package com.pockettavern.app.data.remote.imagegen

import com.pockettavern.app.data.local.inference.KleinDiffusionEngine
import com.pockettavern.app.data.local.inference.KleinModelManager
import com.pockettavern.app.domain.model.ForgeGenerationParams
import com.pockettavern.app.domain.model.GenerationState
import com.pockettavern.app.domain.model.ImageGenBackendType
import com.pockettavern.app.domain.model.ImageGenCapabilities
import com.pockettavern.app.domain.model.Result
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

/**
 * On-device FLUX.2 [klein] generation via [KleinDiffusionEngine]. Klein is a guidance-distilled,
 * fixed-4-step model with no CFG term in the reference implementation (see
 * `KleinDiffusionEngine.cpp`'s schedule, and `docs/flux2-klein-conversion.md`) -- capabilities
 * deliberately expose only prompt + seed, not steps/CFG/negative-prompt, to avoid offering
 * controls that would be no-ops for this backend.
 */
class KleinDiffusionBackend(
    private val engine: KleinDiffusionEngine,
    private val modelManager: KleinModelManager,
) : ImageGenBackend {

    override val type = ImageGenBackendType.LOCAL_FLUX_KLEIN

    override val capabilities = ImageGenCapabilities(
        supportsSeed = true,
        // Fixed 1024x1024 output -- the NPU VAE decode bundle is a fixed-shape compilation, same
        // reasoning as MnnDiffusionBackend's SDXL UNet.
        supportsResolutionPresets = false,
        supportsProgress = true,
        // Not a real URL requirement -- like MnnDiffusionBackend, this reuses the settings
        // screen's existing "Connection" section slot (gated on requiresUrl || requiresApiKey)
        // to show KleinModelSection's manual-staging status instead of adding a new capability
        // flag + UI gate just for this backend.
        requiresUrl = true,
        requiresApiKey = false,
        // No native cancellation -- see KleinDiffusionEngine.generate()'s doc.
        supportsCancel = false,
    )

    override suspend fun testConnection(): Result<Boolean> {
        val status = modelManager.status()
        return when {
            status.qwenReady && status.npuReady -> Result.Success(true)
            !status.qwenReady && !status.npuReady -> Result.Error(Exception(
                "FLUX.2 [klein] model files aren't staged -- push the text encoder to " +
                    "${status.qwenPath} and the NPU bundle to ${status.npuPath}"
            ))
            !status.qwenReady -> Result.Error(Exception(
                "FLUX.2 [klein] text encoder isn't staged at ${status.qwenPath}"
            ))
            else -> Result.Error(Exception(
                "FLUX.2 [klein] NPU bundle isn't staged at ${status.npuPath}"
            ))
        }
    }

    // Klein's schedule is fixed (see class doc) -- nothing to query natively.
    override suspend fun getSamplers(): Result<List<String>> = Result.Success(emptyList())

    override suspend fun getSchedulers(): Result<List<String>> = Result.Success(emptyList())

    override suspend fun getModels(): Result<List<String>> = Result.Success(emptyList())

    override fun generate(params: ForgeGenerationParams): Flow<GenerationState> = flow {
        emit(GenerationState.Starting)
        if (!modelManager.isReady()) {
            emit(GenerationState.Error(
                "FLUX.2 [klein] model files aren't staged -- see Settings for the required paths"
            ))
            return@flow
        }

        engine.generate(prompt = params.prompt, seed = params.seed).collect { progress ->
            when (progress) {
                is KleinDiffusionEngine.Progress.Started -> emit(GenerationState.Starting)
                is KleinDiffusionEngine.Progress.Step ->
                    emit(GenerationState.InProgress(progress = progress.percent / 100f, eta = 0f, previewImage = null))
                is KleinDiffusionEngine.Progress.Done ->
                    emit(GenerationState.Complete(imageBase64 = progress.imageBase64))
                is KleinDiffusionEngine.Progress.Error -> emit(GenerationState.Error(progress.message))
            }
        }
    }

    override suspend fun interrupt(): Result<Unit> = Result.Success(Unit)
}
