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
                "MOAZ Computer Vision looks at a photo and tells you, fully on-device:\n" +
                    "- its PHOTO CATEGORY, the six classes of the course project (buildings, forest, glacier, " +
                    "mountain, sea, street) with a confidence score, and\n" +
                    "- the OBJECT it sees (top 3 of the 1000 ImageNet classes: animals, vehicles, food, objects...).\n" +
                    "It also shows the image size and path, and a Canny edge view."
            )
            AboutSection(
                "HOW TO USE",
                "1. UPLOAD - pick any photo from your gallery.\n" +
                    "2. Choose a model from the model row, or ENSEMBLE to combine all of them.\n" +
                    "3. PREDICT - run the classification. The photo category, its confidence bar and the top 3 objects appear below.\n" +
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
                    "All are trained on ImageNet (ILSVRC 2012, 1000 classes).\n\n" +
                    "PHOTO CATEGORY: a small layer (softmax regression) trained on the Intel Image Classification " +
                    "photos sits on top of the selected model's output and maps it to buildings / forest / glacier / " +
                    "mountain / sea / street. On 501 held-out Intel test photos it scores about 87-90% depending on the model " +
                    "(ensemble 89.6%). When the top score is under 50% (for example a close-up of a cat) the app says NO CLEAR SCENE and still lists the closest categories."
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
                    "Version 1.4"
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
