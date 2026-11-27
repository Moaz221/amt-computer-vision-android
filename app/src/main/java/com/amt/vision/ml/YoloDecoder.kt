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
        iouThreshold: Float = 0.45f
    ): List<Detection> {
        if (rows.isEmpty()) return emptyList()
        val lb = letterbox(srcW, srcH, inputSize)
        val normalized = rows.maxOf { max(max(it[0], it[1]), max(it[2], it[3])) } <= 1.5f
        val k = if (normalized) inputSize.toFloat() else 1f
        val found = ArrayList<Detection>()
        for (r in rows) {
            val obj = r[4]
            if (obj < confThreshold) continue
            var best = 0
            for (c in 1 until r.size - 5) if (r[5 + c] > r[5 + best]) best = c
            val score = obj * r[5 + best]
            if (score < confThreshold) continue
            val cx = r[0] * k; val cy = r[1] * k; val w = r[2] * k; val h = r[3] * k
            val l = ((cx - w / 2f - lb.padX) / lb.scale).coerceIn(0f, srcW.toFloat())
            val t = ((cy - h / 2f - lb.padY) / lb.scale).coerceIn(0f, srcH.toFloat())
            val rr = ((cx + w / 2f - lb.padX) / lb.scale).coerceIn(0f, srcW.toFloat())
            val b = ((cy + h / 2f - lb.padY) / lb.scale).coerceIn(0f, srcH.toFloat())
            if (rr <= l || b <= t) continue
            found += Detection(labels.getOrElse(best) { "class $best" }, best, score, l, t, rr, b)
        }
        return nms(found, iouThreshold)
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
