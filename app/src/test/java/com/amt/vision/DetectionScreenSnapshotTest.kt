package com.amt.vision

import android.graphics.BitmapFactory
import app.cash.paparazzi.DeviceConfig
import app.cash.paparazzi.Paparazzi
import com.amt.vision.ml.Detection
import com.amt.vision.ui.AppTheme
import com.amt.vision.ui.DetectionScreen
import com.amt.vision.vm.DetectionState
import org.junit.Rule
import org.junit.Test

/** Layout preview only: the boxes here are made-up demo values, not model output. */
class DetectionScreenSnapshotTest {

    @get:Rule
    val paparazzi = Paparazzi(deviceConfig = DeviceConfig.PIXEL_6, theme = "android:Theme.Black.NoTitleBar")

    @Test
    fun standby() {
        paparazzi.snapshot { AppTheme { DetectionScreen(DetectionState(), {}, {}, {}, {}) } }
    }

    @Test
    fun withBoxes() {
        val bmp = BitmapFactory.decodeResource(paparazzi.context.resources, R.drawable.sample)
        val w = bmp.width.toFloat(); val h = bmp.height.toFloat()
        val dets = listOf(
            Detection("mask", 0, 0.91f, w * 0.1f, h * 0.15f, w * 0.55f, h * 0.7f),
            Detection("no mask", 1, 0.67f, w * 0.6f, h * 0.3f, w * 0.95f, h * 0.85f)
        )
        paparazzi.snapshot {
            AppTheme { DetectionScreen(DetectionState(bitmap = bmp, detections = dets, status = "2 OBJECT(S) FOUND (DEMO BOXES)"), {}, {}, {}, {}) }
        }
    }
}
