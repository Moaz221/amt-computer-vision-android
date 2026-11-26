package com.amt.vision.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// Unified palette: one accent family (neon cyan) on deep space dark.
val Accent = Color(0xFF00E5FF)
val AccentSoft = Color(0x3300E5FF)
val AccentFaint = Color(0x1A00E5FF)
val BgDark = Color(0xFF05080F)
val Surface = Color(0xFF0B1220)
val SurfaceAlt = Color(0xFF0E1626)
val TextMain = Color(0xFFE6F4FF)
val TextDim = Color(0xFF5B6B80)

private val scheme = darkColorScheme(
    primary = Accent,
    onPrimary = Color(0xFF031018),
    background = BgDark,
    onBackground = TextMain,
    surface = Surface,
    onSurface = TextMain,
    surfaceVariant = SurfaceAlt,
    onSurfaceVariant = TextDim,
    outline = Accent
)

@Composable
fun AppTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = scheme, content = content)
}
