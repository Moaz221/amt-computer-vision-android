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

    private val models = listOf("VGG-19", "InceptionV3", "AlexNet", "Custom CNN", "ENSEMBLE (ALL MODELS)")

    private fun sampleBitmap(): Bitmap =
        BitmapFactory.decodeResource(paparazzi.context.resources, R.drawable.sample)

    private fun screen(state: UiState) {
        paparazzi.snapshot {
            AppTheme { MainScreen(state, {}, {}, {}, {}, {}) }
        }
    }

    @Test
    fun standby() {
        screen(UiState(models = models, selectedModel = "VGG-19"))
    }

    @Test
    fun prediction() {
        screen(
            UiState(
                bitmap = sampleBitmap(),
                models = models,
                selectedModel = "VGG-19",
                prediction = PredictionUi(
                    label = "mountain",
                    confidence = 0.94f,
                    top = listOf("mountain" to 0.94f, "glacier" to 0.04f, "sea" to 0.01f),
                    note = "model: VGG-19"
                ),
                showInfo = true,
                imageInfo = "1280 x 853 x 3",
                filePath = "IMG_20261127_mountain.jpg",
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
                filePath = "IMG_20261127_mountain.jpg",
                status = "CANNY EDGE MAP ACTIVE"
            )
        )
    }
}
