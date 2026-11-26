"""Convert trained Keras .h5 weights to TensorFlow Lite for the Android app.

Usage (on your PC, with TensorFlow installed):
    python tools/convert_to_tflite.py

Reads models_config.json at the repo root (same file the desktop app uses),
converts every models/*.h5 it finds to app/src/main/assets/models/*.tflite,
and updates app/src/main/assets/models_config.json + classes.txt.

Models whose .h5 file is missing are skipped and stay listed in the config;
the app reports them as "missing weights" until you add the .tflite file.
"""
import json
import os

import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "app", "src", "main", "assets")


def main():
    with open(os.path.join(ROOT, "models_config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    os.makedirs(os.path.join(ASSETS, "models"), exist_ok=True)

    for model in config["models"]:
        h5_path = os.path.join(ROOT, "models", model["file"])
        tflite_name = os.path.splitext(model["file"])[0] + ".tflite"
        tflite_path = os.path.join(ASSETS, "models", tflite_name)
        if not os.path.isfile(h5_path):
            print(f"SKIP  {model['name']}: {model['file']} not found")
            continue
        print(f"CONVERT  {model['name']}: {model['file']} -> {tflite_name}")
        keras_model = tf.keras.models.load_model(h5_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        with open(tflite_path, "wb") as out:
            out.write(converter.convert())
        model["file"] = tflite_name

    with open(os.path.join(ASSETS, "models_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    classes_src = os.path.join(ROOT, config.get("classes_file", "classes.txt"))
    if os.path.isfile(classes_src):
        with open(classes_src) as src, open(os.path.join(ASSETS, "classes.txt"), "w") as dst:
            dst.write(src.read())

    print("Done. Rebuild the app in Android Studio.")


if __name__ == "__main__":
    main()
