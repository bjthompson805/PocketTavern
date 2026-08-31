package com.pockettavern.app

import android.content.Context
import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.Surface
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.pockettavern.app.ui.audio.ThemeAudioManager
import com.pockettavern.app.ui.navigation.SillyTavernNavGraph
import com.pockettavern.app.ui.theme.SillyTavernTheme
import com.pockettavern.app.ui.theme.ThemeManager
import com.pockettavern.app.util.LocaleHelper
import com.pockettavern.app.util.NpuDiagnostic
import com.pockettavern.app.util.OnDeviceImageGenerationScreenState
import dagger.hilt.android.AndroidEntryPoint
import javax.inject.Inject
import kotlinx.coroutines.launch

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    @Inject lateinit var themeManager: ThemeManager
    @Inject lateinit var themeAudioManager: ThemeAudioManager

    override fun attachBaseContext(newBase: Context) {
        super.attachBaseContext(LocaleHelper.wrap(newBase))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        // Explicit debug-only command hook used by scratch/run_npu_diffusion.py. This is not an
        // automatic diagnostic: it runs only when adb supplies the extra below, and allows the
        // desktop script to drive real parallel batch-1 CFG image generation with a fixed seed.
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_npu_unet_parallel_cfg_step", false)) {
            Thread { NpuDiagnostic.runParallelBatch1FileStep(applicationContext) }.start()
        }
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_npu_text_encoder_smoke", false)) {
            Thread { NpuDiagnostic.runNativeTextEncoderEngineSmoke(applicationContext) }.start()
        }
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_sdxl_npu_e2e_test", false)) {
            Thread { NpuDiagnostic.runEndToEndSdxlNpuGenerationTest(applicationContext) }.start()
        }
        // Explicit debug-only command hook: does the FLUX.2 [klein] single_blocks.0 AOT-compiled
        // artifact actually run on real Tensor G5 hardware, and match the PyTorch reference? See
        // NpuDiagnostic.runKleinSingle0 and docs/flux2-klein-conversion.md.
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_klein_single0_npu_diagnostic", false)) {
            Thread { NpuDiagnostic.runKleinSingle0(applicationContext) }.start()
        }
        // Explicit debug-only command hook: same as run_klein_single0_npu_diagnostic but using the
        // 80-token probe-shape artifact (single0_Google_Tensor_G5.tflite) with synthetic inputs --
        // isolates whether the real-shape Darwinn fault is token-count/buffer-size dependent.
        // See NpuDiagnostic.runKleinSingle0SmallShape and docs/flux2-klein-conversion.md step 6.
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_klein_single0_small_shape_diagnostic", false)) {
            Thread { NpuDiagnostic.runKleinSingle0SmallShape(applicationContext) }.start()
        }
        // Generalized token-count probe for binary-searching the Darwinn DMA limit.
        // Requires extras: --es klein_token_probe_file <filename> --ei klein_token_probe_tokens <N>
        // See NpuDiagnostic.runKleinTokenProbe and scratch/build_klein_token_bisect.sh.
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_klein_token_probe", false)) {
            val file = intent.getStringExtra("klein_token_probe_file") ?: ""
            val tokens = intent.getIntExtra("klein_token_probe_tokens", -1)
            if (file.isNotBlank() && tokens > 0) {
                Thread { NpuDiagnostic.runKleinTokenProbe(applicationContext, file, tokens) }.start()
            } else {
                android.util.Log.w("MainActivity", "run_klein_token_probe: missing or invalid file/tokens extras")
            }
        }
        // Chunked/flash-attention pivot diagnostic (docs/flux2-klein-conversion.md step 7): does the
        // flash_step kernel (one online-softmax dispatch over a 1024-token Q/K/V chunk) run on Tensor
        // G5 without the DIVE fault or the performance cliff? See NpuDiagnostic.runKleinFlashStepProbe.
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_klein_flash_step_probe", false)) {
            Thread { NpuDiagnostic.runKleinFlashStepProbe(applicationContext) }.start()
        }
        // Chunked-block dispatch design passes 1 and 3 (QKV-projection, output-projection). See
        // NpuDiagnostic.runKleinQkvProjProbe / runKleinOutProjProbe.
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_klein_qkv_proj_probe", false)) {
            Thread { NpuDiagnostic.runKleinQkvProjProbe(applicationContext) }.start()
        }
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_klein_out_proj_probe", false)) {
            Thread { NpuDiagnostic.runKleinOutProjProbe(applicationContext) }.start()
        }
        // The real correctness test: chains all 3 piece-types into one full single_blocks.0 forward
        // at production shape (chunk=1152) and diffs against the real PyTorch reference.
        // See NpuDiagnostic.runKleinChunkedBlockProbe.
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_klein_chunked_block_probe", false)) {
            Thread { NpuDiagnostic.runKleinChunkedBlockProbe(applicationContext) }.start()
        }
        // Same correctness test but for a DOUBLE-stream block (double_blocks.0, cross-attention).
        // See NpuDiagnostic.runKleinDoubleChunkedBlockProbe.
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_klein_double_chunked_block_probe", false)) {
            Thread { NpuDiagnostic.runKleinDoubleChunkedBlockProbe(applicationContext) }.start()
        }
        // Profiling: how much per-dispatch wall-clock cost is CompiledModel.create() overhead vs
        // actual NPU compute? See NpuDiagnostic.runKleinProfileDispatchOverhead.
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_klein_profile_dispatch_overhead", false)) {
            Thread { NpuDiagnostic.runKleinProfileDispatchOverhead(applicationContext) }.start()
        }
        // Native (C++/LiteRT-C-API) perf/correctness confirmation for the chunked/flash-attention
        // design -- no JVM heap involved, unlike run_klein_chunked_block_probe. See
        // NpuDiagnostic.runNativeKleinSingleBlockSmoke.
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_native_klein_single_block_smoke", false)) {
            Thread { NpuDiagnostic.runNativeKleinSingleBlockSmoke(applicationContext) }.start()
        }
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_native_klein_double_block_smoke", false)) {
            android.util.Log.i("MainActivity", "starting native Klein double-block smoke test")
            Thread { NpuDiagnostic.runNativeKleinDoubleBlockSmoke(applicationContext) }.start()
        }
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_native_klein_zero_copy_qkv_flash", false)) {
            Thread { NpuDiagnostic.runNativeKleinZeroCopyQkvFlashProbe(applicationContext) }.start()
        }
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_native_klein_time_in_smoke", false)) {
            Thread { NpuDiagnostic.runNativeKleinTimeInSmoke(applicationContext) }.start()
        }
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_native_klein_component_boundary_smoke", false)) {
            Thread { NpuDiagnostic.runNativeKleinComponentBoundarySmoke(applicationContext) }.start()
        }
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_native_klein_vae_up3_smoke", false)) {
            Thread { NpuDiagnostic.runNativeKleinVaeUp3Smoke(applicationContext) }.start()
        }
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_native_klein_transformer_smoke", false)) {
            Thread { NpuDiagnostic.runNativeKleinTransformerSmoke(applicationContext) }.start()
        }
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_native_klein_one_step_reference", false)) {
            Thread { NpuDiagnostic.runNativeKleinOneStepReference(applicationContext) }.start()
        }
        if (intent.getBooleanExtra("run_native_klein_remaining_reference_steps", false)) {
            android.util.Log.i("MainActivity", "starting remaining Klein reference steps")
            Thread { NpuDiagnostic.runNativeKleinRemainingReferenceSteps(applicationContext) }.start()
        }
        if (BuildConfig.DEBUG && intent.getBooleanExtra("run_native_klein_vae_decode", false)) {
            Thread { NpuDiagnostic.decodeKleinReferenceLatent(applicationContext) }.start()
        }
        if (BuildConfig.DEBUG && intent.hasExtra("run_native_qwen_text_encoder")) {
            val prompt = intent.getStringExtra("run_native_qwen_text_encoder").orEmpty()
            Thread { NpuDiagnostic.runNativeQwenTextEncoder(applicationContext, prompt) }.start()
        }
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.RESUMED) {
                try {
                    OnDeviceImageGenerationScreenState.active.collect { active ->
                        if (active) {
                            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                        } else {
                            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                        }
                    }
                } finally {
                    // The app is no longer visible. Do not keep the display lit in the background.
                    window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                }
            }
        }
        enableEdgeToEdge()
        setContent {
            val themeColors by themeManager.colors.collectAsStateWithLifecycle()
            val particleEffect by themeManager.particleEffect.collectAsStateWithLifecycle()
            val themeAssets by themeManager.themeAssets.collectAsStateWithLifecycle()
            SillyTavernTheme(colors = themeColors, particleEffect = particleEffect, themeAssets = themeAssets) {
                Surface(
                    modifier = Modifier
                        .fillMaxSize()
                        .safeDrawingPadding()
                ) {
                    SillyTavernNavGraph(themeAudioManager = themeAudioManager)
                }
            }
        }
    }
}
