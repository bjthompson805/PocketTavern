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
