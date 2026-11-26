package com.amt.vision.ml

import android.graphics.Bitmap
import kotlin.math.hypot
import kotlin.math.roundToInt

/**
 * Canny edge detection in pure Kotlin:
 * grayscale -> gaussian blur (5x5) -> Sobel -> non-max suppression ->
 * double threshold -> hysteresis. Edges are tinted with the HUD accent color.
 */
object Canny {

    fun detect(src: Bitmap, maxSize: Int = 512): Bitmap {
        val scale = minOf(1f, maxSize.toFloat() / maxOf(src.width, src.height))
        val w = maxOf(1, (src.width * scale).roundToInt())
        val h = maxOf(1, (src.height * scale).roundToInt())
        val scaled = if (scale < 1f) Bitmap.createScaledBitmap(src, w, h, true) else src
        val px = IntArray(w * h)
        scaled.getPixels(px, 0, w, 0, 0, w, h)
        val edges = detectPixels(px, w, h)
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        out.setPixels(edges, 0, w, 0, 0, w, h)
        return out
    }

    /** Returns ARGB pixels: accent-colored edges on near-black. Pure Kotlin, unit-testable. */
    fun detectPixels(px: IntArray, w: Int, h: Int, lowRatio: Float = 0.08f, highRatio: Float = 0.2f): IntArray {
        val gray = FloatArray(w * h)
        for (i in px.indices) {
            val p = px[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            gray[i] = 0.299f * r + 0.587f * g + 0.114f * b
        }

        // Gaussian blur 5x5 (sigma ~1.4)
        val k = floatArrayOf(2f, 4f, 5f, 4f, 2f, 4f, 9f, 12f, 9f, 4f, 5f, 12f, 15f, 12f, 5f, 4f, 9f, 12f, 9f, 4f, 2f, 4f, 5f, 4f, 2f)
        val blurred = FloatArray(w * h)
        for (y in 2 until h - 2) {
            for (x in 2 until w - 2) {
                var acc = 0f
                var ki = 0
                for (dy in -2..2) {
                    val row = (y + dy) * w + x
                    for (dx in -2..2) acc += gray[row + dx] * k[ki++]
                }
                blurred[y * w + x] = acc / 159f
            }
        }

        // Sobel magnitude + direction
        val mag = FloatArray(w * h)
        val dir = FloatArray(w * h)
        var maxMag = 0f
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val i = y * w + x
                val gx = -blurred[i - w - 1] - 2 * blurred[i - 1] - blurred[i + w - 1] +
                        blurred[i - w + 1] + 2 * blurred[i + 1] + blurred[i + w + 1]
                val gy = -blurred[i - w - 1] - 2 * blurred[i - w] - blurred[i - w + 1] +
                        blurred[i + w - 1] + 2 * blurred[i + w] + blurred[i + w + 1]
                val m = hypot(gx, gy)
                mag[i] = m
                if (m > maxMag) maxMag = m
                dir[i] = Math.toDegrees(kotlin.math.atan2(gy, gx).toDouble()).toFloat()
            }
        }
        if (maxMag <= 0f) maxMag = 1f
        val high = maxMag * highRatio
        val low = maxMag * lowRatio

        // Non-maximum suppression
        val nms = FloatArray(w * h)
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val i = y * w + x
                var angle = dir[i] % 180f
                if (angle < 0) angle += 180f
                val m = mag[i]
                val (n1, n2) = when {
                    angle < 22.5f || angle >= 157.5f -> mag[i - 1] to mag[i + 1]
                    angle < 67.5f -> mag[i - w + 1] to mag[i + w - 1]
                    angle < 112.5f -> mag[i - w] to mag[i + w]
                    else -> mag[i - w - 1] to mag[i + w + 1]
                }
                if (m >= n1 && m >= n2) nms[i] = m
            }
        }

        // Double threshold + hysteresis
        val strong = 2
        val weak = 1
        val label = IntArray(w * h)
        val stack = IntArray(w * h)
        var top = 0
        for (i in nms.indices) {
            when {
                nms[i] >= high -> { label[i] = strong; stack[top++] = i }
                nms[i] >= low -> label[i] = weak
            }
        }
        while (top > 0) {
            val i = stack[--top]
            val x = i % w
            val y = i / w
            for (dy in -1..1) for (dx in -1..1) {
                val nx = x + dx
                val ny = y + dy
                if (nx in 1 until w - 1 && ny in 1 until h - 1) {
                    val ni = ny * w + nx
                    if (label[ni] == weak) {
                        label[ni] = strong
                        stack[top++] = ni
                    }
                }
            }
        }

        val edgeColor = 0xFF00E5FF.toInt()
        val bgColor = 0xFF05080F.toInt()
        val out = IntArray(w * h)
        for (i in out.indices) out[i] = if (label[i] == strong) edgeColor else bgColor
        return out
    }
}
