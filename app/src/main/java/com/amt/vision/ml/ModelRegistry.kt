package com.amt.vision.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel

class ModelNotReadyException(val modelFile: String) : Exception("Model file missing: models/$modelFile")

/**
 * One entry of models_config.json.
 *
 * [preprocess] is how the bitmap becomes the input tensor and [output] is what
 * the last layer produces - "softmax" (probabilities, used as they are) or
 * "logits" (raw scores, softmaxed here). Both are written by
 * tools/convert_to_tflite.py, which reads them off the Keras model itself.
 */
data class ModelSpec(
    val name: String,
    val file: String,
    val inputW: Int,
    val inputH: Int,
    val preprocess: String,
    val output: String
)

/** Loads models_config.json from assets, runs single-model and ensemble predictions with TFLite. */
class ModelRegistry(private val context: Context) {

    private val specs: List<ModelSpec>
    val classes: List<String>

    /** Config problems found while loading, surfaced in the UI instead of failing silently. */
    val configWarnings: List<String>

    private val interpreters = mutableMapOf<String, Interpreter>()

    init {
        val warnings = mutableListOf<String>()
        val config = JSONObject(context.assets.open("models_config.json").bufferedReader().use { it.readText() })
        specs = config.getJSONArray("models").let { arr ->
            (0 until arr.length()).map { i ->
                val o = arr.getJSONObject(i)
                val size = o.getJSONArray("input_size")
                val name = o.getString("name")
                ModelSpec(
                    name = name,
                    file = o.getString("file"),
                    inputW = size.getInt(0),
                    inputH = size.getInt(1),
                    preprocess = resolve(name, "preprocess", o.optString("preprocess", ""), warnings) {
                        Preprocessing.normalizeMode(it)
                    },
                    output = resolve(name, "output", o.optString("output", ""), warnings) {
                        Postprocessing.normalizeOutput(it)
                    }
                )
            }
        }
        classes = context.assets.open(config.optString("classes_file", "classes.txt"))
            .bufferedReader().readLines().map { it.trim() }.filter { it.isNotEmpty() }
        configWarnings = warnings
        warnings.forEach { Log.w(TAG, it) }
    }

    private fun resolve(
        modelName: String,
        key: String,
        raw: String,
        warnings: MutableList<String>,
        normalize: (String) -> String
    ): String {
        return try {
            normalize(raw)
        } catch (e: IllegalArgumentException) {
            val fallback = if (key == "output") "softmax" else Preprocessing.DEFAULT_MODE
            warnings += "$modelName: ${e.message}, using $fallback"
            fallback
        }
    }

    fun modelNames(): List<String> = specs.map { it.name }

    fun spec(name: String): ModelSpec = specs.first { it.name == name }

    fun availableModels(): List<String> {
        val assets = context.assets.list("models")?.toSet() ?: emptySet()
        return specs.filter { it.file in assets }.map { it.name }
    }

    /**
     * Runs one model and returns probabilities over [classes].
     * A "logits" model is softmaxed here; a "softmax" model is returned as-is,
     * because softmaxing probabilities again flattens every score toward 1/N.
     */
    fun classify(modelName: String, bitmap: Bitmap): FloatArray {
        val spec = spec(modelName)
        val assets = context.assets.list("models")?.toSet() ?: emptySet()
        if (spec.file !in assets) throw ModelNotReadyException(spec.file)
        val interpreter = interpreters.getOrPut(spec.name) {
            val fd = context.assets.openFd("models/${spec.file}")
            val buffer = FileInputStream(fd.fileDescriptor).channel.map(
                FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
            Interpreter(buffer, Interpreter.Options().setNumThreads(4))
        }

        val scaled = Bitmap.createScaledBitmap(bitmap, spec.inputW, spec.inputH, true)
        val px = IntArray(spec.inputW * spec.inputH)
        scaled.getPixels(px, 0, spec.inputW, 0, 0, spec.inputW, spec.inputH)
        val input = Preprocessing.toInput(spec.preprocess, px, spec.inputW, spec.inputH)

        val outputCount = interpreter.getOutputTensor(0).shape().last()
        if (outputCount != classes.size) {
            throw IllegalStateException(
                "${spec.name} outputs $outputCount values but classes.txt has " +
                    "${classes.size} classes"
            )
        }
        val output = Array(1) { FloatArray(outputCount) }
        interpreter.run(input, output)
        val scores = output[0]
        return if (spec.output == "logits") Postprocessing.softmax(scores) else scores
    }

    /** Averages probabilities over every model whose .tflite file is present. Returns (probs, used, skipped). */
    fun classifyEnsemble(bitmap: Bitmap): Triple<FloatArray, List<String>, List<String>> {
        val available = availableModels()
        val skipped = specs.map { it.name } - available.toSet()
        require(available.isNotEmpty()) { "No model files found in assets/models/" }
        val acc = FloatArray(classes.size)
        available.forEach { name ->
            val probs = classify(name, bitmap)
            for (i in acc.indices) { acc[i] = acc[i] + probs[i] }
        }
        for (i in acc.indices) { acc[i] = acc[i] / available.size }
        return Triple(acc, available, skipped)
    }

    private companion object {
        const val TAG = "ModelRegistry"
    }
}
