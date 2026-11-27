package com.amt.vision.ml

/**
 * Turns a picked bitmap into the tensor a model was trained on.
 *
 * This is the Kotlin side of the single spec in `common/preprocess.py`; the
 * training pipeline applies exactly the same four modes. Keep the two in step.
 *
 * Modes (input is RGB at the model's input size):
 *  - raw      RGB unchanged, 0..255. The model graph normalises internally.
 *  - rescale  RGB / 255            -> 0..1
 *  - tf       RGB / 127.5 - 1      -> -1..1
 *  - caffe    RGB -> BGR, minus [103.939, 116.779, 123.68]
 *
 * No Android types are used here, so `PreprocessingTest` can check these
 * numbers on a plain JVM against the Python reference.
 */
object Preprocessing {

    val CAFFE_MEAN_BGR = floatArrayOf(103.939f, 116.779f, 123.68f)

    val MODES = listOf("raw", "rescale", "tf", "caffe")

    const val DEFAULT_MODE = "rescale"

    /** Maps a models_config.json string to a mode; blank means [DEFAULT_MODE]. */
    fun normalizeMode(mode: String?): String {
        val key = (mode ?: "").trim().lowercase()
        return when (key) {
            "" -> DEFAULT_MODE
            "raw", "none", "identity" -> "raw"
            "rescale", "div255", "0..1" -> "rescale"
            "tf", "-1..1", "tf_1_1" -> "tf"
            "caffe", "bgr" -> "caffe"
            else -> throw IllegalArgumentException(
                "unknown preprocess mode '$mode' (supported: ${MODES.joinToString()})"
            )
        }
    }

    /**
     * Builds the NHWC input tensor from packed ARGB pixels
     * (`Bitmap.getPixels` output) of a [width] x [height] image.
     */
    fun toInput(mode: String, pixels: IntArray, width: Int, height: Int): Array<Array<Array<FloatArray>>> {
        require(pixels.size >= width * height) {
            "need ${width * height} pixels, got ${pixels.size}"
        }
        val resolved = normalizeMode(mode)
        val input = Array(1) { Array(height) { Array(width) { FloatArray(3) } } }
        for (y in 0 until height) {
            val row = y * width
            for (x in 0 until width) {
                val p = pixels[row + x]
                val r = ((p shr 16) and 0xFF).toFloat()
                val g = ((p shr 8) and 0xFF).toFloat()
                val b = (p and 0xFF).toFloat()
                val out = input[0][y][x]
                when (resolved) {
                    "raw" -> { out[0] = r; out[1] = g; out[2] = b }
                    "rescale" -> { out[0] = r / 255f; out[1] = g / 255f; out[2] = b / 255f }
                    "tf" -> {
                        out[0] = r / 127.5f - 1f
                        out[1] = g / 127.5f - 1f
                        out[2] = b / 127.5f - 1f
                    }
                    // caffe: BGR channel order, then per-channel mean subtraction
                    else -> {
                        out[0] = b - CAFFE_MEAN_BGR[0]
                        out[1] = g - CAFFE_MEAN_BGR[1]
                        out[2] = r - CAFFE_MEAN_BGR[2]
                    }
                }
            }
        }
        return input
    }
}

/** What the model's numbers mean on the way out. */
object Postprocessing {

    /** Values a model can produce: already-probabilities, or raw scores. */
    val OUTPUTS = listOf("softmax", "logits")

    /** Maps a models_config.json string to an output mode; blank means softmax. */
    fun normalizeOutput(output: String?): String {
        val key = (output ?: "").trim().lowercase()
        return when (key) {
            "", "softmax", "probabilities", "probs" -> "softmax"
            "logits", "scores" -> "logits"
            else -> throw IllegalArgumentException(
                "unknown output mode '$output' (supported: ${OUTPUTS.joinToString()})"
            )
        }
    }

    fun softmax(logits: FloatArray): FloatArray {
        if (logits.isEmpty()) return logits
        val max = logits.max()
        val exps = FloatArray(logits.size) { Math.exp((logits[it] - max).toDouble()).toFloat() }
        val sum = exps.sum()
        return FloatArray(logits.size) { exps[it] / sum }
    }
}
