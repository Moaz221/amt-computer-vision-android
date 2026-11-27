package com.amt.vision.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel

/**
 * YOLOv5 object detection with TFLite. Needs two files in assets/models/ (written by
 * tools/yolo_training.ipynb): detector.tflite and detector_classes.txt (one class per line).
 * Until they exist, [isReady] is false and the Detection screen says so instead of crashing.
 */
class ObjectDetector(private val context: Context) {

    companion object {
        const val MODEL_FILE = "detector.tflite"
        const val CLASSES_FILE = "detector_classes.txt"
        const val MASK_MIN_SCORE = 0.45f
        const val MIN_BOX_FRACTION = 0.03f
        private const val DIR = "models"
    }

    private var interpreter: Interpreter? = null

    val classes: List<String> = try {
        context.assets.open("$DIR/$CLASSES_FILE").bufferedReader().readLines()
            .map { it.trim() }.filter { it.isNotEmpty() }
    } catch (e: Exception) {
        emptyList()
    }

    fun isReady(): Boolean =
        classes.isNotEmpty() && (context.assets.list(DIR)?.contains(MODEL_FILE) == true)

    fun detect(bitmap: Bitmap, conf: Float = 0.25f): List<Detection> {
        if (!isReady()) throw ModelNotReadyException(MODEL_FILE)
        val interp = interpreter ?: run {
            val fd = context.assets.openFd("$DIR/$MODEL_FILE")
            val buf = FileInputStream(fd.fileDescriptor).channel.map(
                FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
            Interpreter(buf, Interpreter.Options().setNumThreads(4)).also { interpreter = it }
        }
        val inShape = interp.getInputTensor(0).shape() // [1, H, W, 3]
        val size = inShape[1]
        val lb = YoloDecoder.letterbox(bitmap.width, bitmap.height, size)

        val canvasBmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(canvasBmp)
        canvas.drawColor(Color.rgb(114, 114, 114))
        val w = (bitmap.width * lb.scale).toInt().coerceAtLeast(1)
        val h = (bitmap.height * lb.scale).toInt().coerceAtLeast(1)
        canvas.drawBitmap(Bitmap.createScaledBitmap(bitmap, w, h, true), lb.padX, lb.padY, null)

        val px = IntArray(size * size)
        canvasBmp.getPixels(px, 0, size, 0, 0, size, size)
        val input = Array(1) { Array(size) { Array(size) { FloatArray(3) } } }
        for (y in 0 until size) for (x in 0 until size) {
            val p = px[y * size + x]
            input[0][y][x][0] = ((p shr 16) and 0xFF) / 255f
            input[0][y][x][1] = ((p shr 8) and 0xFF) / 255f
            input[0][y][x][2] = (p and 0xFF) / 255f
        }
        val outShape = interp.getOutputTensor(0).shape() // [1, N, 5 + nc]
        val output = Array(1) { Array(outShape[1]) { FloatArray(outShape[2]) } }
        interp.run(input, output)
        check(outShape[2] == 5 + classes.size) {
            "model has ${outShape[2] - 5} classes but $CLASSES_FILE lists ${classes.size}"
        }
        // "mask" needs a higher score than other classes: glasses and similar objects look like masks.
        val maskId = classes.indexOf("mask")
        val minScores = if (maskId >= 0) mapOf(maskId to MASK_MIN_SCORE) else emptyMap()
        return YoloDecoder.decode(
            output[0], classes, size, bitmap.width, bitmap.height, conf,
            classMinScore = minScores, minBoxFraction = MIN_BOX_FRACTION
        )
    }
}
