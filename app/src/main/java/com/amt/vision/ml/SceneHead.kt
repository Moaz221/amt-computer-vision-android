package com.amt.vision.ml

import org.json.JSONObject
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * "Photo category" of the original course app: buildings, forest, glacier,
 * mountain, sea or street.
 *
 * It is a tiny softmax-regression layer on top of the 1000 ImageNet probabilities
 * of whichever model (or the ensemble) just ran. Weights come from
 * tools/train_scene_head.py, trained on the Intel Image Classification photos.
 */
class SceneHead(
    val classes: List<String>,
    private val weights: List<FloatArray>,
    private val bias: FloatArray
) {
    /** Probabilities over [classes] for one ImageNet probability vector. */
    fun predict(imagenetProbs: FloatArray): FloatArray {
        val logits = FloatArray(classes.size) { k ->
            val w = weights[k]
            var s = bias[k]
            val n = minOf(w.size, imagenetProbs.size)
            for (i in 0 until n) s += w[i] * sqrt(imagenetProbs[i].coerceAtLeast(0f))
            s
        }
        val max = logits.max()
        val e = FloatArray(logits.size) { exp(logits[it] - max) }
        val sum = e.sum()
        return FloatArray(e.size) { e[it] / sum }
    }

    companion object {
        fun fromJson(text: String): SceneHead {
            val o = JSONObject(text)
            val cls = o.getJSONArray("classes").let { a -> (0 until a.length()).map { a.getString(it) } }
            val w = o.getJSONArray("weights").let { a ->
                (0 until a.length()).map { k ->
                    val row = a.getJSONArray(k)
                    FloatArray(row.length()) { row.getDouble(it).toFloat() }
                }
            }
            val b = o.getJSONArray("bias").let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } }
            return SceneHead(cls, w, b)
        }
    }
}
