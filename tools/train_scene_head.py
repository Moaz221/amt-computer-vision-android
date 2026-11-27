"""Train the 'photo category' head: a tiny softmax-regression layer that maps the
1000 ImageNet probabilities of the bundled models to the six Intel scene classes
(buildings, forest, glacier, mountain, sea, street) of the course project.

The bundled ImageNet models stay frozen. For every training photo we run all five
models and use each model's probability vector as one training sample, so the same
head works for any single model and for the ensemble.

    python tools/train_scene_head.py --data-dir /path/to/intel   # has seg_train/ and seg_test/
    python tools/train_scene_head.py --features /tmp/feat        # reuse cached features

Writes app/src/main/assets/scene_head.json and prints test accuracy.
"""
import argparse, glob, json, os, sys
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

def feat(p):               # keep in sync with SceneHead.transform in Kotlin
    return np.sqrt(np.clip(p, 0, None))

def extract(data_dir, cache):
    sys.path.insert(0, os.path.join(ROOT, "tools")); sys.path.insert(0, ROOT)
    import predict_demo as pd
    from common.preprocess import apply_preprocess
    cfg = json.load(open(os.path.join(pd.ASSETS, "models_config.json")))
    its = [(m, pd.interp(os.path.join(pd.ASSETS, "models", m["file"]))) for m in cfg["models"]]
    os.makedirs(cache, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(data_dir, "seg_*", "*", "*.jpg"))):
        out = os.path.join(cache, f.replace(os.sep, "_") + ".npy")
        if os.path.exists(out): continue
        img = Image.open(f).convert("RGB"); P = []
        for m, it in its:
            w, h = m["input_size"]
            x = apply_preprocess(np.asarray(img.resize((w, h), Image.BILINEAR)), m["preprocess"])
            P.append(pd.run(it, x))
        np.save(out, np.stack(P).astype(np.float32))

def load(cache):
    X, y, split = [], [], []
    for f in sorted(glob.glob(os.path.join(cache, "*.npy"))):
        name = os.path.basename(f)
        cls = next(c for c in CLASSES if f"_{c}_" in name)
        X.append(np.load(f)); y.append(CLASSES.index(cls)); split.append("seg_test" in name)
    return np.stack(X), np.array(y), np.array(split)   # X: (N, 5 models, 1000)

def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def fit(F, y, epochs=400, lr=0.05, l2=1e-4):
    n, d = F.shape; W = np.zeros((d, 6)); b = np.zeros(6)
    mW = np.zeros_like(W); vW = np.zeros_like(W); mb = np.zeros_like(b); vb = np.zeros_like(b)
    Y = np.eye(6)[y]
    for t in range(1, epochs + 1):
        P = softmax(F @ W + b); G = (P - Y) / n
        gW = F.T @ G + l2 * W; gb = G.sum(0)
        for p, g, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
            m *= 0.9; m += 0.1 * g; v *= 0.999; v += 0.001 * g * g
            p -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W, b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir"); ap.add_argument("--features", default="/tmp/scene_features")
    a = ap.parse_args()
    if a.data_dir: extract(a.data_dir, a.features)
    X, y, is_test = load(a.features)
    tr, te = ~is_test, is_test
    F = feat(X[tr]).reshape(-1, 1000); Y = np.repeat(y[tr], X.shape[1])
    W, b = fit(F, Y)
    def acc(Xs, ys): return float((softmax(feat(Xs) @ W + b).argmax(1) == ys).mean())
    names = ["MobileNetV2", "InceptionV3", "ResNet50", "EfficientNet-Lite0", "EfficientNet-Lite2"]
    print(f"train photos {tr.sum()}  test photos {te.sum()}")
    for i, n in enumerate(names): print(f"  test accuracy {n:<20} {acc(X[te][:, i], y[te])*100:.1f}%")
    ens = X[te].mean(1)
    print(f"  test accuracy {'ENSEMBLE':<20} {acc(ens, y[te])*100:.1f}%")
    out = {"classes": CLASSES, "transform": "sqrt", "bias": [round(float(v), 5) for v in b],
           "weights": [[round(float(v), 4) for v in W[:, k]] for k in range(6)]}
    json.dump(out, open(os.path.join(ROOT, "app/src/main/assets/scene_head.json"), "w"), separators=(",", ":"))
    print("wrote scene_head.json")

if __name__ == "__main__":
    main()
