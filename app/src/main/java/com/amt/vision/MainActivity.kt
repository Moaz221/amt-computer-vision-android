package com.amt.vision

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.lifecycle.viewmodel.compose.viewModel
import com.amt.vision.ui.AppTheme
import com.amt.vision.ui.MainScreen
import com.amt.vision.vm.MainViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AppTheme {
                val vm: MainViewModel = viewModel()
                val state by vm.state.collectAsState()
                val picker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
                    uri?.let { vm.onImagePicked(it) }
                }
                MainScreen(
                    state = state,
                    onUpload = { picker.launch("image/*") },
                    onToggleCanny = vm::onToggleCanny,
                    onPredict = vm::onPredict,
                    onToggleInfo = vm::onToggleInfo,
                    onSelectModel = vm::onSelectModel
                )
            }
        }
    }
}
