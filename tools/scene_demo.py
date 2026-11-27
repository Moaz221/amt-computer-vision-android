"""Photo category + object for photos, the way the app computes it.

    python tools/scene_demo.py photo1.jpg photo2.jpg
"""
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import predict_demo as pd
from common.preprocess import apply_preprocess

def main():
    cfg = json.load(open(os.path.join(pd.ASSETS, "models_config.json")))
    classes = [l.strip() for l in open(os.path.join(pd.ASSETS, cfg["classes_file"])) if l.strip()]
    h = json.load(open(os.path.join(pd.ASSETS, "scene_head.json")))
    W, b = np.array(h["weights"]), np.array(h["bias"])
    its = [(m, pd.interp(os.path.join(pd.ASSETS, "models", m["file"]))) for m in cfg["models"]]
    for path in sys.argv[1:]:
        img = Image.open(path).convert("RGB"); acc = []
        for m, it in its:
            w, hh = m["input_size"]
            acc.append(pd.run(it, apply_preprocess(np.asarray(img.resize((w, hh), Image.BILINEAR)), m["preprocess"])))
        e = np.mean(acc, axis=0)
        z = W @ np.sqrt(e) + b; s = np.exp(z - z.max()); s /= s.sum()
        o = np.argsort(-s)[:3]; t = np.argsort(-e)[:3]
        print(os.path.basename(path), "| category:", ", ".join(f"{h['classes'][i]} {s[i]*100:.0f}%" for i in o),
              "| objects:", ", ".join(f"{classes[i]} {e[i]*100:.0f}%" for i in t))

if __name__ == "__main__":
    main()
