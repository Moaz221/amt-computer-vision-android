"""Build the ImageNet models bundled in the app (no training, no weights from you).

Downloads the Keras ImageNet weights, converts each to a float16 TFLite file in
app/src/main/assets/models/ and writes the app's models_config.json. EfficientNet-Lite
files are fetched as-is from Google's MediaPipe model storage.

    pip install tensorflow-cpu pillow
    python tools/make_pretrained_models.py
"""
import json, os, sys, urllib.request, zipfile, io
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "app", "src", "main", "assets")
MODELS = os.path.join(ASSETS, "models")
MP = "https://storage.googleapis.com/mediapipe-models/image_classifier/%s/float32/1/%s.tflite"

KERAS = [  # name, file, builder, size, preprocess
    ("MobileNetV2", "mobilenetv2.tflite", tf.keras.applications.MobileNetV2, 224, "tf"),
    ("InceptionV3", "inceptionv3.tflite", tf.keras.applications.InceptionV3, 299, "tf"),
    ("ResNet50", "resnet50.tflite", tf.keras.applications.ResNet50, 224, "caffe"),
]
LITE = [("EfficientNet-Lite0", "efficientnet_lite0", 224), ("EfficientNet-Lite2", "efficientnet_lite2", 260)]

def main():
    os.makedirs(MODELS, exist_ok=True)
    entries = []
    for name, fn, ctor, size, pre in KERAS:
        path = os.path.join(MODELS, fn)
        if not os.path.exists(path):
            print("converting", name, flush=True)
            m = ctor(weights="imagenet")
            c = tf.lite.TFLiteConverter.from_keras_model(m)
            c.optimizations = [tf.lite.Optimize.DEFAULT]
            c.target_spec.supported_types = [tf.float16]
            open(path, "wb").write(c.convert())
            tf.keras.backend.clear_session()
        entries.append(dict(name=name, file=fn, input_size=[size, size], preprocess=pre, output="softmax"))
    labels = None
    for name, stem, size in LITE:
        fn = stem + ".tflite"
        path = os.path.join(MODELS, fn)
        if not os.path.exists(path):
            print("downloading", name, flush=True)
            urllib.request.urlretrieve(MP % (stem, stem), path)
        if labels is None:
            with zipfile.ZipFile(path) as z:
                labels = z.read(z.namelist()[0]).decode().split("\n")
        entries.append(dict(name=name, file=fn, input_size=[size, size], preprocess="effnet", output="softmax"))
    labels = [l.strip() for l in labels if l.strip()]
    assert len(labels) == 1000, len(labels)
    open(os.path.join(ASSETS, "classes.txt"), "w").write("\n".join(labels) + "\n")
    json.dump(dict(classes_file="classes.txt", models=entries), open(os.path.join(ASSETS, "models_config.json"), "w"), indent=2)
    print("done")

if __name__ == "__main__":
    main()
