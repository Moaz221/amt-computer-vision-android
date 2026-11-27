package com.amt.vision.vm

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.amt.vision.ml.Detection
import com.amt.vision.ml.ObjectDetector
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class DetectionState(
    val bitmap: Bitmap? = null,
    val detections: List<Detection> = emptyList(),
    val modelReady: Boolean = false,
    val running: Boolean = false,
    val threshold: Float = 0.20f,
    val status: String = "STANDBY"
)

class DetectionViewModel(app: Application) : AndroidViewModel(app) {
    private val detector = ObjectDetector(app)
    private val _state = MutableStateFlow(
        DetectionState(
            modelReady = detector.isReady(),
            status = if (detector.isReady()) "SYSTEM ONLINE" else "DETECTOR MODEL NOT INSTALLED"
        )
    )
    val state: StateFlow<DetectionState> = _state

    fun onImagePicked(uri: Uri) {
        viewModelScope.launch(Dispatchers.Default) {
            val bmp = getApplication<Application>().contentResolver.openInputStream(uri)?.use { input ->
                com.amt.vision.util.BitmapLoader.decode(input.readBytes(), 1280)
            }
            if (bmp == null) {
                _state.update { it.copy(status = "FAILED TO LOAD IMAGE") }
                return@launch
            }
            _state.update { it.copy(bitmap = bmp, detections = emptyList(), status = "IMAGE LOADED") }
            runDetection(bmp)
        }
    }

    fun onThreshold(value: Float) {
        _state.update { it.copy(threshold = value) }
    }

    fun onRotate() {
        val bmp = _state.value.bitmap ?: run {
            _state.update { it.copy(status = "NO IMAGE LOADED") }
            return
        }
        viewModelScope.launch(Dispatchers.Default) {
            val m = android.graphics.Matrix().apply { postRotate(90f) }
            val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
            _state.update { it.copy(bitmap = rotated, detections = emptyList(), status = "ROTATED 90\u00B0") }
            runDetection(rotated)
        }
    }

    fun onDetect() {
        val bmp = _state.value.bitmap ?: run {
            _state.update { it.copy(status = "NO IMAGE LOADED") }
            return
        }
        viewModelScope.launch(Dispatchers.Default) { runDetection(bmp) }
    }

    private fun runDetection(bmp: Bitmap) {
        if (!detector.isReady()) {
            _state.update { it.copy(running = false, status = "DETECTOR MODEL NOT INSTALLED") }
            return
        }
        _state.update { it.copy(running = true, status = "DETECTING...") }
        try {
            val found = detector.detect(bmp, _state.value.threshold)
            _state.update {
                it.copy(
                    running = false,
                    detections = found,
                    status = if (found.isEmpty()) "NO OBJECTS FOUND" else "${found.size} OBJECT(S) FOUND"
                )
            }
        } catch (e: Exception) {
            _state.update { it.copy(running = false, status = "DETECTION ERROR: ${e.message}") }
        }
    }
}
