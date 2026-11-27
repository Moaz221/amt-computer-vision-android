"""Run every bundled model on photos, the same way the app does, and print the top-3.

    python tools/predict_demo.py photo1.jpg photo2.jpg
    python tools/predict_demo.py photo.jpg --keras     # also compare with the Keras model

It reads app/src/main/assets/models_config.json, resizes each photo to the model's
input size, applies the model's preprocessing mode (common/preprocess.py, mirrored in
Preprocessing.kt) and runs the .tflite file. The ensemble line averages all models.
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.preprocess import apply_preprocess

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "src", "main", "assets")

def interp(path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    i = Interpreter(model_path=path); i.allocate_tensors(); return i

def run(it, x):
    it.set_tensor(it.get_input_details()[0]["index"], x[None].astype(np.float32))
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])[0]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("images", nargs="+"); ap.add_argument("--keras", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(ASSETS, "models_config.json")))
    classes = [l.strip() for l in open(os.path.join(ASSETS, cfg["classes_file"])) if l.strip()]
    for img_path in a.images:
        img = Image.open(img_path).convert("RGB")
        print(f"\n== {os.path.basename(img_path)} {img.size}")
        acc = []
        for m in cfg["models"]:
            w, h = m["input_size"]
            x = apply_preprocess(np.asarray(img.resize((w, h), Image.BILINEAR)), m["preprocess"])
            p = run(interp(os.path.join(ASSETS, "models", m["file"])), x)
            if m["output"] == "logits": p = np.exp(p - p.max()); p /= p.sum()
            acc.append(p)
            top = np.argsort(-p)[:3]
            print(f"{m['name']:<20}" + " | ".join(f"{classes[i]} {p[i]*100:.1f}%" for i in top))
        e = np.mean(acc, axis=0); top = np.argsort(-e)[:3]
        print(f"{'ENSEMBLE':<20}" + " | ".join(f"{classes[i]} {e[i]*100:.1f}%" for i in top))

if __name__ == "__main__":
    main()
