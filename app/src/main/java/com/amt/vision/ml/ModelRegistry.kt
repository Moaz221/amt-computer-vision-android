package com.amt.vision.ml

import android.content.Context
import android.graphics.Bitmap
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel
import kotlin.math.exp

class ModelNotReadyException(val modelFile: String) : Exception("Model file missing: models/$modelFile")

data class ModelSpec(
    val name: String,
    val file: String,
    val inputW: Int,
    val inputH: Int,
    val preprocess: String
)

/** Loads models_config.json from assets, runs single-model and ensemble predictions with TFLite. */
class ModelRegistry(private val context: Context) {

    private val specs: List<ModelSpec>
    val classes: List<String>
    private val interpreters = mutableMapOf<String, Interpreter>()

    init {
        val config = JSONObject(context.assets.open("models_config.json").bufferedReader().use { it.readText() })
        specs = config.getJSONArray("models").let { arr ->
            (0 until arr.length()).map { i ->
                val o = arr.getJSONObject(i)
                val size = o.getJSONArray("input_size")
                ModelSpec(
                    name = o.getString("name"),
                    file = o.getString("file"),
                    inputW = size.getInt(0),
                    inputH = size.getInt(1),
                    preprocess = o.optString("preprocess", "rescale")
                )
            }
        }
        classes = context.assets.open(config.optString("classes_file", "classes.txt"))
            .bufferedReader().readLines().map { it.trim() }.filter { it.isNotEmpty() }
    }

    fun modelNames(): List<String> = specs.map { it.name }

    fun spec(name: String): ModelSpec = specs.first { it.name == name }

    fun availableModels(): List<String> {
        val assets = context.assets.list("models")?.toSet() ?: emptySet()
        return specs.filter { it.file in assets }.map { it.name }
    }

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
        val input = preprocess(bitmap, spec)
        val output = Array(1) { FloatArray(classes.size) }
        interpreter.run(input, output)
        return softmax(output[0])
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

    private fun preprocess(bitmap: Bitmap, spec: ModelSpec): Array<Array<Array<FloatArray>>> {
        val scaled = Bitmap.createScaledBitmap(bitmap, spec.inputW, spec.inputH, true)
        val px = IntArray(spec.inputW * spec.inputH)
        scaled.getPixels(px, 0, spec.inputW, 0, 0, spec.inputW, spec.inputH)
        val input = Array(1) { Array(spec.inputH) { Array(spec.inputW) { FloatArray(3) } } }
        for (y in 0 until spec.inputH) {
            for (x in 0 until spec.inputW) {
                val p = px[y * spec.inputW + x]
                val r = ((p shr 16) and 0xFF).toFloat()
                val g = ((p shr 8) and 0xFF).toFloat()
                val b = (p and 0xFF).toFloat()
                val out = input[0][y][x]
                when (spec.preprocess) {
                    "caffe" -> { out[0] = b - 103.939f; out[1] = g - 116.779f; out[2] = r - 123.68f }
                    "tf" -> { out[0] = r / 127.5f - 1f; out[1] = g / 127.5f - 1f; out[2] = b / 127.5f - 1f }
                    else -> { out[0] = r / 255f; out[1] = g / 255f; out[2] = b / 255f }
                }
            }
        }
        return input
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()
        val exps = FloatArray(logits.size) { exp((logits[it] - max).toDouble()).toFloat() }
        val sum = exps.sum()
        return FloatArray(logits.size) { exps[it] / sum }
    }
}
