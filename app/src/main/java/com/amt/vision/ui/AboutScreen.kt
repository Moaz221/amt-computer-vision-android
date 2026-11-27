package com.amt.vision.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun AboutScreen(onBack: () -> Unit) {
    BackHandler(onBack = onBack)
    Column(
        Modifier
            .fillMaxSize()
            .background(BgDark)
            .statusBarsPadding()
            .navigationBarsPadding()
            .padding(12.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text(
                    "ABOUT THIS APP",
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
                    .clickable(onClick = onBack),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    "X",
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold,
                    fontSize = 14.sp,
                    color = Accent
                )
            }
        }
        Spacer(Modifier.height(12.dp))
        Column(
            Modifier
                .weight(1f)
                .fillMaxWidth()
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            AboutSection(
                "WHAT IT DOES",
                "MOAZ Computer Vision classifies any photo, fully on-device. " +
                    "Upload a picture and the app predicts which of the 1000 ImageNet categories it belongs to, " +
                    "shows a confidence score, and lists the top 3 guesses. Works with everyday photos: animals, " +
                    "objects, food, vehicles, scenes and more."
            )
            AboutSection(
                "HOW TO USE",
                "1. UPLOAD - pick any photo from your gallery.\n" +
                    "2. Choose a model from the model row, or ENSEMBLE to combine all of them.\n" +
                    "3. PREDICT - run the classification. The result, confidence bar and top 3 appear below.\n" +
                    "4. CANNY - toggle the edge-detection view of the uploaded image.\n" +
                    "5. INFO - show the uploaded image's metadata (dimensions)."
            )
            AboutSection(
                "MODELS ON BOARD",
                "Five pretrained ImageNet models are bundled inside the app and run locally with TensorFlow Lite:\n\n" +
                    "- MobileNetV2 - small and fast, great all-rounder.\n" +
                    "- InceptionV3 - Google's 299px inception network.\n" +
                    "- ResNet50 - 50-layer residual network.\n" +
                    "- EfficientNet-Lite0 - MediaPipe mobile model.\n" +
                    "- EfficientNet-Lite2 - bigger MediaPipe mobile model.\n\n" +
                    "All are trained on ImageNet (ILSVRC 2012, 1000 classes)."
            )
            AboutSection(
                "ENSEMBLE MODE",
                "Select ENSEMBLE (ALL MODELS) and the app runs every model on your photo and averages their " +
                    "probability outputs into one combined answer. Slower than a single model, but more robust: " +
                    "it smooths out any one model's mistakes."
            )
            AboutSection(
                "PRIVACY",
                "Everything runs offline on your phone. Your photos never leave the device. " +
                    "No account, no server, no internet connection needed."
            )
            AboutSection(
                "CREDITS & LICENSES",
                "- Model weights: Keras Applications (MobileNetV2, InceptionV3, ResNet50) and Google MediaPipe " +
                    "(EfficientNet-Lite0/2), Apache License 2.0.\n" +
                    "- Class labels: ImageNet / ILSVRC 2012.\n" +
                    "- App icon: original generated artwork, free to use.\n" +
                    "- Built with Kotlin, Jetpack Compose and TensorFlow Lite.\n\n" +
                    "Version 1.3"
            )
            Spacer(Modifier.height(8.dp))
        }
    }
}

@Composable
private fun AboutSection(title: String, body: String) {
    Surface(
        Modifier.fillMaxWidth(),
        color = Surface,
        border = BorderStroke(1.dp, Accent.copy(alpha = 0.25f)),
        shape = RoundedCornerShape(6.dp)
    ) {
        Column(Modifier.padding(12.dp)) {
            Text(
                title,
                fontFamily = FontFamily.Monospace,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
                color = Accent
            )
            Spacer(Modifier.height(6.dp))
            Text(
                body,
                fontSize = 13.sp,
                lineHeight = 19.sp,
                color = TextMain
            )
        }
    }
}
