"""The one definition of how an image becomes a model input tensor.

Everything that touches preprocessing reads it from here:
`training/train_models.py` (what the model is trained on),
`tools/convert_to_tflite.py` (what gets written into models_config.json),
`tools/verify_parity.py` (the Keras vs TFLite check), and the Android app's
`app/src/main/java/com/amt/vision/ml/Preprocessing.kt`, which implements the
same four modes. If you change a mode here, change it there too - that is
exactly the drift this file exists to prevent.

PREPROCESS_MODES - input is an 8-bit RGB image already resized to the model's
input size, laid out HWC. Output is float32 in the same layout.

    raw      RGB unchanged, 0..255.
             Use when the .h5 graph normalises internally, e.g. it starts
             with a Rescaling(1/255) layer.
    rescale  RGB / 255                    -> 0..1
    tf       RGB / 127.5 - 1              -> -1..1
             (Keras preprocess_input for InceptionV3 / ResNet)
    caffe    RGB -> BGR, minus [103.939, 116.779, 123.68]
             (Keras preprocess_input for VGG / AlexNet / ResNet50, Caffe channel order)
    effnet   (RGB - 127) / 128            -> about -1..1
             (EfficientNet-Lite from MediaPipe)

OUTPUT_MODES - what the last layer of the model produces.

    softmax  last layer is a softmax: values are probabilities summing to 1,
             so the app must NOT softmax them again.
    logits   raw scores: the app applies softmax itself.
"""

import numpy as np

CAFFE_MEAN_BGR = (103.939, 116.779, 123.68)

PREPROCESS_MODES = {
    "raw": "RGB 0..255, no change (the model normalises internally)",
    "rescale": "RGB / 255 -> 0..1",
    "tf": "RGB / 127.5 - 1 -> -1..1",
    "caffe": "RGB -> BGR, minus [103.939, 116.779, 123.68]",
    "effnet": "(RGB - 127) / 128",
}

OUTPUT_MODES = {
    "softmax": "last layer is a softmax; values are probabilities summing to 1",
    "logits": "raw scores; the app applies softmax",
}

DEFAULT_PREPROCESS = "rescale"
DEFAULT_OUTPUT = "softmax"

_ALIASES = {
    "raw": "raw",
    "none": "raw",
    "identity": "raw",
    "rescale": "rescale",
    "div255": "rescale",
    "0..1": "rescale",
    "tf": "tf",
    "-1..1": "tf",
    "tf_1_1": "tf",
    "caffe": "caffe",
    "bgr": "caffe",
    "effnet": "effnet",
    "efficientnet": "effnet",
}

# The four models the app ships with. Keep in sync with
# app/src/main/assets/models_config.json.
DEFAULT_MODELS = [
    {"key": "vgg19", "name": "VGG-19", "h5": "vgg19.h5",
     "input_size": [224, 224], "preprocess": "caffe"},
    {"key": "inceptionv3", "name": "InceptionV3", "h5": "inceptionv3.h5",
     "input_size": [299, 299], "preprocess": "tf"},
    {"key": "alexnet", "name": "AlexNet", "h5": "alexnet.h5",
     "input_size": [227, 227], "preprocess": "caffe"},
    {"key": "custom_cnn", "name": "Custom CNN", "h5": "custom_cnn.h5",
     "input_size": [100, 100], "preprocess": "rescale"},
]

DEFAULT_MODELS_BY_KEY = {m["key"]: m for m in DEFAULT_MODELS}


def normalize_mode(mode, warn=None):
    """Map a config string to one of PREPROCESS_MODES, or to the default."""
    if mode is None or str(mode).strip() == "":
        if warn:
            warn(f"missing 'preprocess', assuming '{DEFAULT_PREPROCESS}'")
        return DEFAULT_PREPROCESS
    key = str(mode).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(
        f"unknown preprocess mode {mode!r}; supported: {', '.join(PREPROCESS_MODES)}"
    )


def normalize_output(output, warn=None):
    """Map a config string to 'softmax' or 'logits'."""
    if output is None or str(output).strip() == "":
        if warn:
            warn(f"missing 'output', assuming '{DEFAULT_OUTPUT}'")
        return DEFAULT_OUTPUT
    key = str(output).strip().lower()
    if key in OUTPUT_MODES:
        return key
    raise ValueError(
        f"unknown output mode {output!r}; supported: {', '.join(OUTPUT_MODES)}"
    )


def apply_preprocess(x, mode):
    """NumPy reference implementation. `x` is RGB, 0..255, shape (..., 3)."""
    mode = normalize_mode(mode)
    x = np.asarray(x, dtype=np.float32)
    if mode == "raw":
        return x
    if mode == "rescale":
        return x / 255.0
    if mode == "tf":
        return x / 127.5 - 1.0
    if mode == "effnet":
        return (x - 127.0) / 128.0
    # caffe: RGB -> BGR, then per-channel mean subtraction
    bgr = x[..., ::-1]
    return bgr - np.asarray(CAFFE_MEAN_BGR, dtype=np.float32)


def apply_preprocess_tf(x, mode):
    """TensorFlow implementation with the same semantics as apply_preprocess."""
    import tensorflow as tf

    mode = normalize_mode(mode)
    x = tf.cast(x, tf.float32)
    if mode == "raw":
        return x
    if mode == "rescale":
        return x / 255.0
    if mode == "tf":
        return x / 127.5 - 1.0
    if mode == "effnet":
        return (x - 127.0) / 128.0
    bgr = tf.stack([x[..., 2], x[..., 1], x[..., 0]], axis=-1)
    return bgr - tf.constant(CAFFE_MEAN_BGR, dtype=tf.float32)


def detect_baked_normalization(model):
    """Return (mode, detail) when the Keras graph normalises its own input.

    A model trained by an older version of train_models.py has a
    Rescaling(1/255) layer inside the graph. Feeding it app-side 0..1 values
    rescales twice, so the app must send raw 0..255 instead. Returns
    (None, None) when the graph takes plain RGB.
    """
    for layer in list(model.layers)[:3]:
        cls = type(layer).__name__
        if cls == "Rescaling":
            scale = float(np.asarray(layer.scale).ravel()[0])
            offset = float(np.asarray(layer.offset).ravel()[0])
            if abs(scale - 1.0 / 255.0) < 1e-9 and abs(offset) < 1e-9:
                return "raw", f"{layer.name}: Rescaling(1/255) is inside the graph"
            return None, f"{layer.name}: Rescaling({scale:g}, {offset:g}) - check it by hand"
        if cls == "Normalization":
            return "raw", f"{layer.name}: Normalization layer is inside the graph"
    return None, None


def detect_output_mode(model):
    """Return 'softmax' or 'logits' by looking at the model's last layer."""
    last = model.layers[-1]
    activation = getattr(last, "activation", None)
    name = getattr(activation, "__name__", "")
    if name == "softmax" or type(last).__name__ == "Softmax":
        return "softmax"
    return "logits"


def load_image_rgb(path, input_size):
    """Decode an image, resize to (w, h) bilinear, return uint8 RGB HWC."""
    import tensorflow as tf

    w, h = int(input_size[0]), int(input_size[1])
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (h, w), method="bilinear")
    return tf.clip_by_value(tf.round(image * 255.0), 0, 255).numpy().astype(np.uint8)
