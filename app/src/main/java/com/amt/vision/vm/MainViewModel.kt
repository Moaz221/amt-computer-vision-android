package com.amt.vision.vm

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.amt.vision.ml.Canny
import com.amt.vision.ml.ModelNotReadyException
import com.amt.vision.ml.ModelRegistry
import com.amt.vision.ui.PredictionUi
import com.amt.vision.ui.UiState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

class MainViewModel(app: Application) : AndroidViewModel(app) {

    companion object {
        const val ENSEMBLE = "ENSEMBLE (ALL MODELS)"
    }

    private val registry = ModelRegistry(app)

    private val _state = MutableStateFlow(
        UiState(
            models = registry.modelNames() + ENSEMBLE,
            selectedModel = registry.modelNames().firstOrNull() ?: ENSEMBLE
        )
    )
    val state: StateFlow<UiState> = _state

    fun onSelectModel(name: String) {
        _state.update { it.copy(selectedModel = name, prediction = null) }
    }

    fun onImagePicked(uri: Uri) {
        viewModelScope.launch(Dispatchers.Default) {
            val resolver = getApplication<Application>().contentResolver
            val bitmap = resolver.openInputStream(uri)?.use { input ->
                val bytes = input.readBytes()
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
                var sample = 1
                while (bounds.outWidth / (sample * 2) >= 1024 && bounds.outHeight / (sample * 2) >= 1024) sample *= 2
                val opts = BitmapFactory.Options().apply { inSampleSize = sample }
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
            }
            if (bitmap == null) {
                _state.update { it.copy(status = "FAILED TO LOAD IMAGE") }
                return@launch
            }
            val name = resolver.query(uri, null, null, null, null)?.use { c ->
                val idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (c.moveToFirst() && idx >= 0) c.getString(idx) else null
            } ?: uri.lastPathSegment ?: uri.toString()
            _state.update {
                it.copy(
                    bitmap = bitmap,
                    cannyBitmap = null,
                    cannyOn = false,
                    prediction = null,
                    imageInfo = "${bitmap.width} x ${bitmap.height} x 3",
                    filePath = name,
                    status = "IMAGE LOADED"
                )
            }
        }
    }

    fun onToggleCanny() {
        val current = _state.value
        val src = current.bitmap
        if (src == null) {
            _state.update { it.copy(status = "NO IMAGE LOADED") }
            return
        }
        if (current.cannyOn) {
            _state.update { it.copy(cannyOn = false, status = "RGB MODE") }
            return
        }
        val cached = current.cannyBitmap
        if (cached != null) {
            _state.update { it.copy(cannyOn = true, status = "CANNY EDGE MAP ACTIVE") }
            return
        }
        _state.update { it.copy(status = "COMPUTING EDGES...") }
        viewModelScope.launch(Dispatchers.Default) {
            val edges = Canny.detect(src)
            _state.update { it.copy(cannyBitmap = edges, cannyOn = true, status = "CANNY EDGE MAP ACTIVE") }
        }
    }

    fun onToggleInfo() {
        _state.update { it.copy(showInfo = !it.showInfo) }
    }

    fun onPredict() {
        val current = _state.value
        val src = current.bitmap
        if (src == null) {
            _state.update { it.copy(status = "NO IMAGE LOADED") }
            return
        }
        _state.update { it.copy(predicting = true, status = "RUNNING INFERENCE...") }
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val (probs, note) = if (current.selectedModel == ENSEMBLE) {
                    val (p, used, skipped) = registry.classifyEnsemble(src)
                    val note = buildString {
                        append("ensemble of ${used.size}: ${used.joinToString()}")
                        if (skipped.isNotEmpty()) append(" | skipped: ${skipped.joinToString()}")
                    }
                    p to note
                } else {
                    val spec = registry.spec(current.selectedModel)
                    registry.classify(current.selectedModel, src) to
                        "model: ${spec.name} | ${spec.preprocess} -> ${spec.output}"
                }
                val fullNote = if (registry.configWarnings.isEmpty()) note
                else registry.configWarnings.joinToString(" | ") + " | " + note
                val ranked = probs.indices.sortedByDescending { probs[it] }
                val label = registry.classes.getOrElse(ranked.first()) { "class ${ranked.first()}" }
                val top = ranked.take(3).map { (registry.classes.getOrElse(it) { "class $it" }) to probs[it] }
                val sceneProbs = registry.sceneHead?.predict(probs)
                val sceneRank = sceneProbs?.indices?.sortedByDescending { sceneProbs[it] }
                val ui = PredictionUi(
                    label, probs[ranked.first()], top, fullNote,
                    scene = sceneRank?.first()?.let { if (sceneProbs[it] < 0.5f) "no clear scene" else registry.sceneHead!!.classes[it] },
                    sceneConfidence = sceneRank?.first()?.let { sceneProbs[it] } ?: 0f,
                    sceneTop = sceneRank?.take(3)?.map { registry.sceneHead!!.classes[it] to sceneProbs[it] } ?: emptyList()
                )
                _state.update {
                    it.copy(
                        predicting = false,
                        prediction = ui,
                        status = "PREDICTION COMPLETE"
                    )
                }
            } catch (e: ModelNotReadyException) {
                _state.update {
                    it.copy(
                        predicting = false,
                        status = "MISSING WEIGHTS"
                    )
                }
            } catch (e: Exception) {
                _state.update { it.copy(predicting = false, status = "INFERENCE ERROR: ${e.message}") }
            }
        }
    }
}
