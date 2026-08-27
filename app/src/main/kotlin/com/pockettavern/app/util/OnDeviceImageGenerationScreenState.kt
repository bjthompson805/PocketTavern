package com.pockettavern.app.util

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Process-local signal from the on-device SDXL worker to the visible activity. Keeping the
 * display interactive avoids power-management slowdowns during either CPU or Tensor-NPU image
 * generation. The activity clears its window flag whenever it is no longer visible.
 */
object OnDeviceImageGenerationScreenState {
    private val _active = MutableStateFlow(false)
    val active = _active.asStateFlow()

    fun begin() {
        _active.value = true
    }

    fun end() {
        _active.value = false
    }
}
