package com.amt.vision.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.amt.vision.vm.DetectionState
import kotlin.math.min
import kotlin.math.roundToInt

private val Palette = listOf(Color(0xFF00E5FF), Color(0xFFFF5252), Color(0xFF69F0AE), Color(0xFFFFD740), Color(0xFFE040FB))

@Composable
fun DetectionScreen(
    state: DetectionState,
    onUpload: () -> Unit,
    onDetect: () -> Unit,
    onThreshold: (Float) -> Unit,
    onBack: () -> Unit
) {
    Column(
        Modifier.fillMaxSize().background(BgDark).statusBarsPadding().navigationBarsPadding().padding(12.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                "OBJECT DETECTION",
                fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold, fontSize = 20.sp,
                color = TextMain, modifier = Modifier.weight(1f)
            )
            Chip("BACK", onBack)
        }
        Spacer(Modifier.height(10.dp))
        Box(
            Modifier.weight(1f).fillMaxWidth()
                .clip(RoundedCornerShape(6.dp))
                .border(1.dp, Accent.copy(alpha = 0.6f), RoundedCornerShape(6.dp))
                .background(Color(0xFF030509))
        ) {
            val bmp = state.bitmap
            if (bmp == null) {
                Text(
                    "UPLOAD AN IMAGE", fontFamily = FontFamily.Monospace, fontSize = 13.sp,
                    color = TextDim, modifier = Modifier.align(Alignment.Center)
                )
            } else {
                Image(
                    bitmap = bmp.asImageBitmap(), contentDescription = null,
                    modifier = Modifier.fillMaxSize().padding(14.dp), contentScale = ContentScale.Fit
                )
                // Boxes are in image pixels; map them onto the Fit-scaled image inside the 14dp padding.
                Canvas(Modifier.fillMaxSize().padding(14.dp)) {
                    val s = min(size.width / bmp.width, size.height / bmp.height)
                    val offX = (size.width - bmp.width * s) / 2f
                    val offY = (size.height - bmp.height * s) / 2f
                    val paint = android.graphics.Paint().apply {
                        textSize = 13.sp.toPx(); isAntiAlias = true; typeface = android.graphics.Typeface.MONOSPACE
                    }
                    state.detections.forEach { d ->
                        val color = Palette[d.classId % Palette.size]
                        val tl = Offset(offX + d.left * s, offY + d.top * s)
                        drawRect(color, tl, Size((d.right - d.left) * s, (d.bottom - d.top) * s), style = Stroke(3f))
                        val text = "${d.label} ${(d.score * 100).roundToInt()}%"
                        val tw = paint.measureText(text)
                        val th = paint.textSize + 8f
                        val ty = if (tl.y - th < 0f) tl.y else tl.y - th
                        drawRect(color, Offset(tl.x, ty), Size(tw + 12f, th))
                        paint.color = android.graphics.Color.BLACK
                        drawContext.canvas.nativeCanvas.drawText(text, tl.x + 6f, ty + th - 8f, paint)
                    }
                }
            }
        }
        Spacer(Modifier.height(10.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("CONFIDENCE ${(state.threshold * 100).roundToInt()}%", fontFamily = FontFamily.Monospace, fontSize = 12.sp, color = TextDim)
            Slider(value = state.threshold, onValueChange = onThreshold, valueRange = 0.05f..0.9f, modifier = Modifier.weight(1f).padding(horizontal = 8.dp))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            Chip("UPLOAD IMAGE", onUpload, Modifier.weight(1f))
            Chip(if (state.running) "RUNNING..." else "DETECT", onDetect, Modifier.weight(1f))
        }
        Spacer(Modifier.height(8.dp))
        Text(
            "> ${state.status}", fontFamily = FontFamily.Monospace, fontSize = 12.sp,
            color = if (state.modelReady) Accent else Color(0xFFFFD740)
        )
    }
}

@Composable
private fun Chip(label: String, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Box(
        modifier.clip(RoundedCornerShape(6.dp)).background(AccentFaint)
            .border(1.dp, Accent.copy(alpha = 0.6f), RoundedCornerShape(6.dp))
            .clickable(onClick = onClick).padding(horizontal = 14.dp, vertical = 10.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(label, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold, fontSize = 13.sp, color = Accent)
    }
}
