"""Check that one converted model gives the same answer in Keras and in TFLite.

Usage:
    python tools/verify_parity.py --image /path/to/photo.jpg
    python tools/verify_parity.py --image photo.jpg --model "Custom CNN"
    python tools/verify_parity.py --image photo.jpg --model alexnet --top-k 3
    python tools/verify_parity.py --image photo.jpg --tflite-only

It loads the image, resizes it the way the app does, applies the preprocessing
mode from models_config.json, then runs the same tensor through the .h5 in
Keras and through the .tflite in the TensorFlow Lite interpreter. The top class
and the score for each must match; anything else means the app would show a
different answer than your notebook.

The Android app feeds the interpreter that identical tensor: Preprocessing.kt
implements the same four modes, and PreprocessingTest checks its numbers
against this file's. So a pass here plus a green PreprocessingTest means the
phone's score is the one printed below.

Exit code 0 on a match, 1 on a mismatch, 2 on a setup problem.
"""

import argparse
import json
import os
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from common.preprocess import (  # noqa: E402
    apply_preprocess,
    load_image_rgb,
    normalize_mode,
    normalize_output,
)

ASSETS = os.path.join(REPO_ROOT, "app", "src", "main", "assets")


def resolve_config(path):
    """The app's own config first: this script checks what the app will do."""
    if path:
        return os.path.abspath(path)
    assets_config = os.path.join(ASSETS, "models_config.json")
    return assets_config if os.path.isfile(assets_config) else os.path.join(REPO_ROOT, "models_config.json")


def load_labels(path, config, config_path):
    # The class list belongs with the config that names it: an app config is
    # paired with the app's classes.txt, a training manifest with the root one.
    candidates = [path] if path else [
        os.path.join(os.path.dirname(config_path), config.get("classes_file", "classes.txt")),
        os.path.join(REPO_ROOT, "classes.txt"),
        os.path.join(ASSETS, "classes.txt"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()], candidate
    return [], None


def pick_entry(entries, wanted):
    if not wanted:
        return entries[0]
    key = wanted.strip().lower()
    for entry in entries:
        stem = os.path.splitext(os.path.basename(entry["file"]))[0].lower()
        if entry.get("name", "").lower() == key or stem == key:
            return entry
    names = ", ".join(f"{e.get('name')} ({os.path.splitext(e['file'])[0]})" for e in entries)
    raise SystemExit(f"no model matches {wanted!r}; config has: {names}")


def as_probabilities(values, output):
    """Logits become probabilities the way the app's Postprocessing.softmax does."""
    if output != "logits":
        return values
    shifted = values - values.max()
    exps = np.exp(shifted)
    return exps / exps.sum()


def top_k(values, labels, k):
    order = np.argsort(values)[::-1][:k]
    return [(labels[i] if i < len(labels) else f"class {i}", float(values[i])) for i in order]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="test image (jpg/png)")
    parser.add_argument("--model", default=None, help="model name or key (default: first in config)")
    parser.add_argument("--config", default=None, help="models_config.json (default: app assets)")
    parser.add_argument("--h5", default=None, help="Keras weights (default: models/<stem>.h5)")
    parser.add_argument("--tflite", default=None, help="converted model (default: app assets/models/<stem>.tflite)")
    parser.add_argument("--labels", default=None, help="class list (default: classes.txt)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--tflite-only", action="store_true",
                        help="skip Keras and just report what the .tflite predicts")
    parser.add_argument("--dump-tensor", default=None,
                        help="write the input tensor as text, for debugging the app side")
    args = parser.parse_args(argv)

    config_path = resolve_config(args.config)
    if not os.path.isfile(config_path):
        raise SystemExit(f"no config at {config_path}; pass --config")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    entry = pick_entry(config["models"], args.model)
    stem = os.path.splitext(os.path.basename(entry["file"]))[0]
    mode = normalize_mode(entry.get("preprocess"))
    output = normalize_output(entry.get("output"))
    width, height = (int(v) for v in entry["input_size"])

    labels, labels_path = load_labels(args.labels, config, config_path)

    tflite_path = os.path.abspath(args.tflite) if args.tflite else os.path.join(
        ASSETS, "models", stem + ".tflite")
    h5_path = os.path.abspath(args.h5) if args.h5 else os.path.join(
        REPO_ROOT, "models", stem + ".h5")
    if not os.path.isfile(tflite_path):
        raise SystemExit(
            f"no .tflite at {tflite_path}\n"
            "Convert first: python tools/convert_to_tflite.py")

    pixels = load_image_rgb(args.image, (width, height))
    tensor = apply_preprocess(pixels.astype(np.float32), mode)[None, ...]  # NHWC

    print(f"model:     {entry.get('name')} ({stem})")
    print(f"image:     {args.image} -> {width}x{height}")
    print(f"preprocess {mode}  output {output}  labels {len(labels)}"
          f" ({labels_path or 'none'})")
    print(f"input:     shape {tensor.shape}, min {tensor.min():.3f}, "
          f"max {tensor.max():.3f}, mean {tensor.mean():.3f}")

    if args.dump_tensor:
        with open(args.dump_tensor, "w", encoding="utf-8") as f:
            f.write(f"{width} {height} {mode}\n")
            for y in range(height):
                for x in range(width):
                    f.write(" ".join(f"{v:.7g}" for v in tensor[0, y, x]) + "\n")
        print(f"tensor:    written to {args.dump_tensor}")

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], tensor.astype(np.float32))
    interpreter.invoke()
    lite_out = interpreter.get_tensor(out["index"])[0]

    print()
    print("TFLite (what the app shows):")
    for label, score in top_k(as_probabilities(lite_out, output), labels, args.top_k):
        print(f"  {label:<12} {score:.4f}")

    if args.tflite_only:
        return 0

    if not os.path.isfile(h5_path):
        print()
        print(f"no Keras weights at {h5_path}, so there is nothing to compare against.")
        print("Pass --h5, or keep the .h5 next to the repo in models/.")
        return 0

    keras_model = tf.keras.models.load_model(h5_path, compile=False)
    keras_out = np.asarray(keras_model.predict(tensor, verbose=0))[0]

    keras_probs = as_probabilities(keras_out, output)

    print()
    print("Keras (.h5 source):")
    for label, score in top_k(keras_probs, labels, args.top_k):
        print(f"  {label:<12} {score:.4f}")

    diff = float(np.max(np.abs(keras_out.astype(np.float64) - lite_out.astype(np.float64))))
    keras_top = int(np.argmax(keras_out))
    lite_top = int(np.argmax(lite_out))
    same = keras_top == lite_top and diff < 1e-3

    print()
    print(f"top class: keras {labels[keras_top] if keras_top < len(labels) else keras_top}"
          f" / tflite {labels[lite_top] if lite_top < len(labels) else lite_top}")
    print(f"max |keras - tflite| = {diff:.2e}")

    if len(labels) and len(labels) != len(keras_out):
        print(f"WARNING: {len(labels)} labels but the model outputs {len(keras_out)} values")
        return 1

    if not same:
        print("FAIL: the converted model does not match its Keras source.")
        return 1

    print("PASS: same top class and the same score in both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
