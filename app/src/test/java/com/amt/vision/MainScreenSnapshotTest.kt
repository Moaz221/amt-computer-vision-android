package com.amt.vision

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import app.cash.paparazzi.DeviceConfig
import app.cash.paparazzi.Paparazzi
import com.amt.vision.ml.Canny
import com.amt.vision.ui.AppTheme
import com.amt.vision.ui.MainScreen
import com.amt.vision.ui.PredictionUi
import com.amt.vision.ui.UiState
import org.junit.Rule
import org.junit.Test

class MainScreenSnapshotTest {

    @get:Rule
    val paparazzi = Paparazzi(
        deviceConfig = DeviceConfig.PIXEL_6,
        theme = "android:Theme.Black.NoTitleBar"
    )

    private val models = listOf("MobileNetV2", "InceptionV3", "ResNet50", "EfficientNet-Lite0", "EfficientNet-Lite2", "ENSEMBLE (ALL MODELS)")

    private fun sampleBitmap(): Bitmap =
        BitmapFactory.decodeResource(paparazzi.context.resources, R.drawable.sample)

    private fun screen(state: UiState) {
        paparazzi.snapshot {
            AppTheme { MainScreen(state, {}, {}, {}, {}, {}, {}) }
        }
    }

    @Test
    fun standby() {
        screen(UiState(models = models, selectedModel = "ENSEMBLE (ALL MODELS)"))
    }

    @Test
    fun prediction() {
        screen(
            UiState(
                bitmap = sampleBitmap(),
                models = models,
                selectedModel = "ENSEMBLE (ALL MODELS)",
                prediction = PredictionUi(
                    // real ensemble output for this photo (tools/predict_demo.py)
                    label = "tabby",
                    confidence = 0.447f,
                    top = listOf("tabby" to 0.447f, "tiger cat" to 0.186f, "Egyptian cat" to 0.153f),
                    note = "ensemble of 5: MobileNetV2, InceptionV3, ResNet50, EfficientNet-Lite0, EfficientNet-Lite2",
                    scene = "no clear scene", sceneConfidence = 0.37f,
                    sceneTop = listOf("sea" to 0.37f, "buildings" to 0.25f, "street" to 0.12f)
                ),
                showInfo = true,
                imageInfo = "1024 x 626 x 3",
                filePath = "IMG_20261127_cat.jpg",
                status = "PREDICTION COMPLETE"
            )
        )
    }

    @Test
    fun canny() {
        val bmp = sampleBitmap()
        screen(
            UiState(
                bitmap = bmp,
                cannyBitmap = Canny.detect(bmp),
                cannyOn = true,
                models = models,
                selectedModel = "ENSEMBLE (ALL MODELS)",
                filePath = "IMG_20261127_cat.jpg",
                status = "CANNY EDGE MAP ACTIVE"
            )
        )
    }
}
