package com.amt.vision.ui

import android.graphics.Bitmap
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.roundToInt

data class PredictionUi(
    val label: String,
    val confidence: Float,
    val top: List<Pair<String, Float>>,
    val note: String?,
    /** Photo category from the course project: buildings, forest, glacier, mountain, sea, street. */
    val scene: String? = null,
    val sceneConfidence: Float = 0f,
    val sceneTop: List<Pair<String, Float>> = emptyList()
)

data class UiState(
    val bitmap: Bitmap? = null,
    val cannyBitmap: Bitmap? = null,
    val cannyOn: Boolean = false,
    val models: List<String> = emptyList(),
    val selectedModel: String = "",
    val predicting: Boolean = false,
    val prediction: PredictionUi? = null,
    val showInfo: Boolean = false,
    val imageInfo: String? = null,
    val filePath: String? = null,
    val status: String = "SYSTEM ONLINE"
)

@Composable
fun MainScreen(
    state: UiState,
    onUpload: () -> Unit,
    onToggleCanny: () -> Unit,
    onPredict: () -> Unit,
    onToggleInfo: () -> Unit,
    onSelectModel: (String) -> Unit,
    onAbout: () -> Unit
) {
    Column(
        Modifier
            .fillMaxSize()
            .background(BgDark)
            .statusBarsPadding()
            .navigationBarsPadding()
            .padding(12.dp)
    ) {
        TopBar(onAbout)
        Spacer(Modifier.height(10.dp))
        HudViewport(state, Modifier.weight(1f).fillMaxWidth())
        Spacer(Modifier.height(10.dp))
        ControlDeck(state, onUpload, onToggleCanny, onPredict, onToggleInfo)
        Spacer(Modifier.height(8.dp))
        ModelSelector(state.models, state.selectedModel, onSelectModel)
        Spacer(Modifier.height(8.dp))
        PredictionPanel(state.prediction)
        if (state.showInfo) {
            Spacer(Modifier.height(8.dp))
            InfoRow(state.imageInfo)
        }
        Spacer(Modifier.height(8.dp))
        PathBar(state.filePath)
        Spacer(Modifier.height(8.dp))
        FooterBar(state.status)
    }
}

@Composable
private fun TopBar(onAbout: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(
                "MOAZ COMPUTER VISION",
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold,
                fontSize = 20.sp,
                color = TextMain
            )
            Spacer(Modifier.height(6.dp))
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(2.dp)
                    .background(Brush.horizontalGradient(listOf(Accent, Accent.copy(alpha = 0f))))
            )
        }
        Spacer(Modifier.width(12.dp))
        Box(
            Modifier
                .size(40.dp)
                .clip(CircleShape)
                .background(AccentFaint)
                .border(1.dp, Accent.copy(alpha = 0.6f), CircleShape)
                .clickable(onClick = onAbout),
            contentAlignment = Alignment.Center
        ) {
            Text(
                "i",
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold,
                fontSize = 18.sp,
                color = Accent
            )
        }
    }
}

@Composable
private fun HudViewport(state: UiState, modifier: Modifier = Modifier) {
    Box(
        modifier
            .clip(RoundedCornerShape(6.dp))
            .border(1.dp, Accent.copy(alpha = 0.6f), RoundedCornerShape(6.dp))
            .background(Color(0xFF030509))
    ) {
        GridBackground()
        val shown = if (state.cannyOn) state.cannyBitmap else state.bitmap
        if (shown != null) {
            Image(
                bitmap = shown.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier.fillMaxSize().padding(14.dp),
                contentScale = ContentScale.Fit
            )
        } else {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(
                    "UPLOAD AN IMAGE",
                    fontFamily = FontFamily.Monospace,
                    fontSize = 13.sp,
                    color = TextDim
                )
            }
        }
        CornerBrackets()
        ModeChip(
            "MODE: " + when {
                state.cannyOn -> "CANNY"
                state.bitmap != null -> "RGB"
                else -> "STANDBY"
            },
            Modifier.align(Alignment.TopStart).padding(10.dp)
        )
    }
}

@Composable
private fun GridBackground() {
    Canvas(Modifier.fillMaxSize()) {
        val step = 32.dp.toPx()
        val color = Accent.copy(alpha = 0.05f)
        var x = 0f
        while (x < size.width) { drawLine(color, Offset(x, 0f), Offset(x, size.height)); x += step }
        var y = 0f
        while (y < size.height) { drawLine(color, Offset(0f, y), Offset(size.width, y)); y += step }
    }
}

@Composable
private fun CornerBrackets() {
    Canvas(Modifier.fillMaxSize().padding(6.dp)) {
        val len = 22.dp.toPx()
        val stroke = 2.5f.dp.toPx()
        val c = Accent
        val pts = listOf(
            Offset(0f, 0f), Offset(size.width, 0f),
            Offset(0f, size.height), Offset(size.width, size.height)
        )
        for (p in pts) {
            val sx = if (p.x == 0f) 1f else -1f
            val sy = if (p.y == 0f) 1f else -1f
            drawLine(c, p, Offset(p.x + sx * len, p.y), strokeWidth = stroke)
            drawLine(c, p, Offset(p.x, p.y + sy * len), strokeWidth = stroke)
        }
    }
}

@Composable
private fun ModeChip(text: String, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier,
        color = AccentFaint,
        border = BorderStroke(1.dp, Accent.copy(alpha = 0.5f)),
        shape = RoundedCornerShape(3.dp)
    ) {
        Text(
            text,
            Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
            fontFamily = FontFamily.Monospace,
            fontSize = 10.sp,
            color = Accent
        )
    }
}

@Composable
private fun ControlDeck(
    state: UiState,
    onUpload: () -> Unit,
    onToggleCanny: () -> Unit,
    onPredict: () -> Unit,
    onToggleInfo: () -> Unit
) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        DeckButton("UPLOAD", Modifier.weight(1f), active = false, onClick = onUpload)
        DeckButton("CANNY", Modifier.weight(1f), active = state.cannyOn, onClick = onToggleCanny)
        DeckButton(
            if (state.predicting) "..." else "PREDICT",
            Modifier.weight(1f),
            active = state.predicting,
            onClick = onPredict
        )
        DeckButton("INFO", Modifier.weight(1f), active = state.showInfo, onClick = onToggleInfo)
    }
}

@Composable
private fun DeckButton(label: String, modifier: Modifier = Modifier, active: Boolean, onClick: () -> Unit) {
    OutlinedButton(
        onClick = onClick,
        modifier = modifier,
        shape = RoundedCornerShape(4.dp),
        border = BorderStroke(1.dp, if (active) Accent else Accent.copy(alpha = 0.4f)),
        colors = ButtonDefaults.outlinedButtonColors(
            containerColor = if (active) AccentSoft else AccentFaint,
            contentColor = Accent
        ),
        contentPadding = PaddingValues(vertical = 10.dp)
    ) {
        Text(label, fontFamily = FontFamily.Monospace, fontSize = 12.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun ModelSelector(models: List<String>, selected: String, onSelect: (String) -> Unit) {
    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        items(models) { m ->
            val isSel = m == selected
            OutlinedButton(
                onClick = { onSelect(m) },
                shape = RoundedCornerShape(4.dp),
                border = BorderStroke(1.dp, if (isSel) Accent else Accent.copy(alpha = 0.25f)),
                colors = ButtonDefaults.outlinedButtonColors(
                    containerColor = if (isSel) AccentSoft else Surface,
                    contentColor = if (isSel) Accent else TextDim
                ),
                contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp)
            ) {
                Text(m, fontFamily = FontFamily.Monospace, fontSize = 11.sp)
            }
        }
    }
}

@Composable
private fun PredictionPanel(prediction: PredictionUi?) {
    Surface(
        Modifier.fillMaxWidth(),
        color = Surface,
        border = BorderStroke(1.dp, Accent.copy(alpha = 0.25f)),
        shape = RoundedCornerShape(6.dp)
    ) {
        Column(Modifier.padding(12.dp)) {
            Text("PREDICTION READOUT", fontFamily = FontFamily.Monospace, fontSize = 10.sp, color = TextDim)
            Spacer(Modifier.height(8.dp))
            if (prediction != null) {
                if (prediction.scene != null) {
                    Text("PHOTO CATEGORY", fontFamily = FontFamily.Monospace, fontSize = 10.sp, color = TextDim)
                    Row(verticalAlignment = Alignment.Bottom) {
                        Text(
                            prediction.scene.uppercase(),
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold,
                            fontSize = 26.sp,
                            color = Accent,
                            modifier = Modifier.weight(1f)
                        )
                        Text(
                            "${(prediction.sceneConfidence * 100).roundToInt()}%",
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold,
                            fontSize = 18.sp,
                            color = Accent
                        )
                    }
                    Spacer(Modifier.height(4.dp))
                    Box(Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)).background(SurfaceAlt)) {
                        Box(
                            Modifier
                                .fillMaxWidth(prediction.sceneConfidence.coerceIn(0f, 1f))
                                .fillMaxHeight()
                                .background(Brush.horizontalGradient(listOf(Accent.copy(alpha = 0.5f), Accent)))
                        )
                    }
                    Spacer(Modifier.height(4.dp))
                    Text(
                        prediction.sceneTop.joinToString("  ") { (n, p) -> "$n ${(p * 100).roundToInt()}%" },
                        fontFamily = FontFamily.Monospace, fontSize = 10.sp, color = TextDim
                    )
                    Spacer(Modifier.height(10.dp))
                    Text("OBJECT SEEN (IMAGENET)", fontFamily = FontFamily.Monospace, fontSize = 10.sp, color = TextDim)
                }
                Row(verticalAlignment = Alignment.Bottom) {
                    Text(
                        prediction.label.uppercase(),
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold,
                        fontSize = if (prediction.scene != null) 16.sp else 22.sp,
                        color = if (prediction.scene != null) TextMain else Accent,
                        modifier = Modifier.weight(1f)
                    )
                    Text(
                        "${(prediction.confidence * 100).roundToInt()}%",
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold,
                        fontSize = 16.sp,
                        color = Accent
                    )
                }
                Spacer(Modifier.height(6.dp))
                Box(Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)).background(SurfaceAlt)) {
                    Box(
                        Modifier
                            .fillMaxWidth(prediction.confidence.coerceIn(0f, 1f))
                            .fillMaxHeight()
                            .background(Brush.horizontalGradient(listOf(Accent.copy(alpha = 0.5f), Accent)))
                    )
                }
                Spacer(Modifier.height(8.dp))
                prediction.top.forEachIndexed { i, (name, p) ->
                    Text(
                        "#${i + 1}  $name  ${(p * 100).roundToInt()}%",
                        fontFamily = FontFamily.Monospace,
                        fontSize = 11.sp,
                        color = TextDim
                    )
                }
                if (!prediction.note.isNullOrBlank()) {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        prediction.note,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 10.sp,
                        color = TextDim,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            } else {
                Text("AWAITING PREDICTION", fontFamily = FontFamily.Monospace, fontSize = 12.sp, color = TextDim)
            }
        }
    }
}

@Composable
private fun InfoRow(info: String?) {
    Surface(
        Modifier.fillMaxWidth(),
        color = Surface,
        border = BorderStroke(1.dp, Accent.copy(alpha = 0.25f)),
        shape = RoundedCornerShape(6.dp)
    ) {
        Text(
            "IMAGE INFO   SHAPE ${info ?: "-"}   DIM ${info?.substringBefore(" x ") ?: "-"}",
            Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
            fontFamily = FontFamily.Monospace,
            fontSize = 11.sp,
            color = TextDim
        )
    }
}

@Composable
private fun PathBar(path: String?) {
    Surface(
        Modifier.fillMaxWidth(),
        color = Surface,
        border = BorderStroke(1.dp, Accent.copy(alpha = 0.25f)),
        shape = RoundedCornerShape(6.dp)
    ) {
        Text(
            "FILEPATH   ${path ?: "-"}",
            Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
            fontFamily = FontFamily.Monospace,
            fontSize = 11.sp,
            color = TextDim,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@Composable
private fun FooterBar(status: String) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(8.dp).clip(CircleShape).background(Accent))
        Spacer(Modifier.width(8.dp))
        Text(status, fontFamily = FontFamily.Monospace, fontSize = 11.sp, color = TextDim)
    }
}
