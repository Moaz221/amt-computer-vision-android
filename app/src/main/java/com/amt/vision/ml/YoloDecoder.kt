package com.amt.vision.ml

import kotlin.math.max
import kotlin.math.min

/** One detected box in the coordinates of the ORIGINAL image (pixels). */
data class Detection(
    val label: String,
    val classId: Int,
    val score: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

/**
 * Pure-Kotlin post-processing for a YOLOv5 TFLite export (shape [1, N, 5 + nc]:
 * x, y, w, h, objectness, class scores). No Android classes, so it is unit tested on the JVM.
 *
 * Boxes come out normalized (0..1) from the YOLOv5 TFLite export; if the model was exported with
 * pixel coordinates (values above ~1.5) they are used as they are.
 */
object YoloDecoder {

    /** How the original image was fitted into the square model input (letterbox). */
    class Letterbox(val scale: Float, val padX: Float, val padY: Float)

    fun letterbox(srcW: Int, srcH: Int, size: Int): Letterbox {
        val scale = min(size.toFloat() / srcW, size.toFloat() / srcH)
        val padX = (size - srcW * scale) / 2f
        val padY = (size - srcH * scale) / 2f
        return Letterbox(scale, padX, padY)
    }

    fun decode(
        rows: Array<FloatArray>,
        labels: List<String>,
        inputSize: Int,
        srcW: Int,
        srcH: Int,
        confThreshold: Float = 0.25f,
        iouThreshold: Float = 0.45f,
        classMinScore: Map<Int, Float> = emptyMap(),
        minBoxFraction: Float = 0f
    ): List<Detection> {
        if (rows.isEmpty()) return emptyList()
        val lb = letterbox(srcW, srcH, inputSize)
        // Decide the unit from the box CENTERS of confident rows only. Junk rows can have a width or
        // height above 1.5 even in a normalized export, which used to flip this to "pixels" and
        // draw every box as a tiny/huge rectangle in the top-left corner.
        val confident = rows.filter { it[4] >= confThreshold }
        val normalized = confident.isEmpty() || confident.maxOf { max(it[0], it[1]) } <= 1.5f
        val k = if (normalized) inputSize.toFloat() else 1f
        val found = ArrayList<Detection>()
        for (r in rows) {
            val obj = r[4]
            if (obj < confThreshold) continue
            var best = 0
            for (c in 1 until r.size - 5) if (r[5 + c] > r[5 + best]) best = c
            val score = obj * r[5 + best]
            if (score < max(confThreshold, classMinScore[best] ?: 0f)) continue
            val cx = r[0] * k; val cy = r[1] * k; val w = r[2] * k; val h = r[3] * k
            val l = ((cx - w / 2f - lb.padX) / lb.scale).coerceIn(0f, srcW.toFloat())
            val t = ((cy - h / 2f - lb.padY) / lb.scale).coerceIn(0f, srcH.toFloat())
            val rr = ((cx + w / 2f - lb.padX) / lb.scale).coerceIn(0f, srcW.toFloat())
            val b = ((cy + h / 2f - lb.padY) / lb.scale).coerceIn(0f, srcH.toFloat())
            if (rr <= l || b <= t) continue
            if (rr - l < minBoxFraction * max(srcW, srcH) || b - t < minBoxFraction * max(srcW, srcH)) continue
            found += Detection(labels.getOrElse(best) { "class $best" }, best, score, l, t, rr, b)
        }
        return dropContainedOtherClass(nms(found, iouThreshold))
    }

    /**
     * A weaker box of another class sitting (mostly) inside a stronger box is almost always a
     * false positive (for example a "mask" on glasses inside a "no mask" face).
     */
    fun dropContainedOtherClass(dets: List<Detection>): List<Detection> {
        val sorted = dets.sortedByDescending { it.score }
        val kept = ArrayList<Detection>()
        for (d in sorted) {
            val inside = kept.any {
                it.classId != d.classId && containedFraction(d, it) > 0.8f
            }
            if (!inside) kept += d
        }
        return kept
    }

    private fun containedFraction(inner: Detection, outer: Detection): Float {
        val iw = min(inner.right, outer.right) - max(inner.left, outer.left)
        val ih = min(inner.bottom, outer.bottom) - max(inner.top, outer.top)
        if (iw <= 0f || ih <= 0f) return 0f
        val area = (inner.right - inner.left) * (inner.bottom - inner.top)
        return if (area <= 0f) 0f else iw * ih / area
    }

    /** Per-class non-maximum suppression, highest score first. */
    fun nms(dets: List<Detection>, iouThreshold: Float): List<Detection> {
        val kept = ArrayList<Detection>()
        for (d in dets.sortedByDescending { it.score }) {
            if (kept.none { it.classId == d.classId && iou(it, d) > iouThreshold }) kept += d
        }
        return kept
    }

    fun iou(a: Detection, b: Detection): Float {
        val iw = min(a.right, b.right) - max(a.left, b.left)
        val ih = min(a.bottom, b.bottom) - max(a.top, b.top)
        if (iw <= 0f || ih <= 0f) return 0f
        val inter = iw * ih
        val union = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - inter
        return if (union <= 0f) 0f else inter / union
    }
}
