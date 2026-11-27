package com.amt.vision

import app.cash.paparazzi.DeviceConfig
import app.cash.paparazzi.Paparazzi
import com.amt.vision.ui.AboutScreen
import com.amt.vision.ui.AppTheme
import org.junit.Rule
import org.junit.Test

class AboutScreenSnapshotTest {

    @get:Rule
    val paparazzi = Paparazzi(
        deviceConfig = DeviceConfig.PIXEL_6,
        theme = "android:Theme.Black.NoTitleBar"
    )

    @Test
    fun about() {
        paparazzi.snapshot {
            AppTheme { AboutScreen(onBack = {}) }
        }
    }
}
