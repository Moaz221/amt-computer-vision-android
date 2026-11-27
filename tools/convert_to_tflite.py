"""Convert trained Keras .h5 weights to TensorFlow Lite for the Android app.

Usage (on your PC, with TensorFlow installed):

    # everything found in models/, using the repo defaults
    python tools/convert_to_tflite.py

    # explicit inputs
    python tools/convert_to_tflite.py \
        --config my_models.json --labels my_classes.txt \
        --weights-dir /path/to/h5 --out-dir app/src/main/assets/models

    # one model only, forcing the app-side preprocessing mode
    python tools/convert_to_tflite.py --model "Custom CNN" --set-preprocess rescale

    # validate an existing setup without converting anything
    python tools/convert_to_tflite.py --check-only

Inputs it needs:

    --config       model list. Default: ./models_config.json if it exists,
                   otherwise app/src/main/assets/models_config.json.
    --labels       class names, one per line. Default: ./classes.txt if it
                   exists, otherwise the config's "classes_file", otherwise
                   app/src/main/assets/classes.txt.
    --weights-dir  where the .h5 files live. Default: ./models.

Outputs:

    <out-dir>/<name>.tflite        the converted models
    <config-out>                   models_config.json for the app
    <labels-out>                   classes.txt for the app

Each config entry is
    {"name", "file", "input_size": [w, h], "preprocess", "output"}
where "preprocess" is one of raw/rescale/tf/caffe (see common/preprocess.py)
and "output" is "softmax" or "logits". Both are detected from the Keras model
and corrected here, because a model whose graph already normalises its own
input (an older train_models.py put Rescaling(1/255) inside the graph) would
otherwise be preprocessed twice by the app.

Every conversion is checked immediately: the same input tensor is run through
Keras and through the new .tflite, and the top class plus the largest
probability difference are printed. Use tools/verify_parity.py to repeat that
check with a real image.

With more than one model, each one is converted in its own process: a full
VGG-19 or InceptionV3 conversion holds a lot of memory, and one process per
model keeps a modest laptop from running out. Pass --in-process to do them all
in this one instead.
"""

import argparse
import contextlib
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from common.preprocess import (  # noqa: E402
    DEFAULT_PREPROCESS,
    OUTPUT_MODES,
    PREPROCESS_MODES,
    apply_preprocess,
    detect_baked_normalization,
    detect_output_mode,
    normalize_mode,
    normalize_output,
)

ASSETS = os.path.join(REPO_ROOT, "app", "src", "main", "assets")
DEFAULT_CONFIG = os.path.join(ASSETS, "models_config.json")
DEFAULT_OUT_DIR = os.path.join(ASSETS, "models")
DEFAULT_LABELS_OUT = os.path.join(ASSETS, "classes.txt")


def say(msg=""):
    print(msg, flush=True)


def short(path):
    """Repo-relative when it is inside the repo, absolute otherwise."""
    rel = os.path.relpath(path, REPO_ROOT)
    return path if rel.startswith("..") else rel


@contextlib.contextmanager
def capture_stderr():
    """Hold back TensorFlow's C++ conversion chatter; show it only on failure."""
    with tempfile.TemporaryFile(mode="w+") as log:
        saved = os.dup(2)
        try:
            os.dup2(log.fileno(), 2)
            yield log
        finally:
            os.dup2(saved, 2)
            os.close(saved)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="model config JSON (default: ./models_config.json, else "
                        "app/src/main/assets/models_config.json)")
    p.add_argument("--labels", default=None,
                   help="class names, one per line (default: ./classes.txt, else the "
                        "config's classes_file, else app/src/main/assets/classes.txt)")
    p.add_argument("--weights-dir", default=os.path.join(REPO_ROOT, "models"),
                   help="folder holding the .h5 files (default: ./models)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="where to write .tflite files (default: app assets/models)")
    p.add_argument("--config-out", default=None,
                   help="where to write the app config (default: the app assets "
                        "config, unless --config already points there)")
    p.add_argument("--labels-out", default=DEFAULT_LABELS_OUT,
                   help="where to write the app class list (default: assets/classes.txt)")
    p.add_argument("--model", action="append", default=None,
                   help="only convert these models (name or key, repeatable)")
    p.add_argument("--set-preprocess", default=None, choices=list(PREPROCESS_MODES),
                   help="force the app-side preprocessing mode for the selected models")
    p.add_argument("--set-output", default=None, choices=list(OUTPUT_MODES),
                   help="force softmax/logits for the selected models")
    p.add_argument("--check-only", action="store_true",
                   help="validate the config and existing .tflite files, convert nothing")
    p.add_argument("--force", action="store_true",
                   help="rewrite the app config even when nothing was converted")
    p.add_argument("--in-process", action="store_true",
                   help="convert every model in this process instead of one process each")
    p.add_argument("--record-json", default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def resolve_config_path(explicit):
    if explicit:
        return os.path.abspath(explicit)
    root_manifest = os.path.join(REPO_ROOT, "models_config.json")
    if os.path.isfile(root_manifest):
        return root_manifest
    return DEFAULT_CONFIG


def resolve_labels_path(explicit, config, config_path):
    candidates = []
    if explicit:
        candidates.append(os.path.abspath(explicit))
    else:
        candidates += [
            os.path.join(REPO_ROOT, "classes.txt"),
            os.path.join(os.path.dirname(config_path), config.get("classes_file", "classes.txt")),
            DEFAULT_LABELS_OUT,
        ]
    for path in candidates:
        if os.path.isfile(path):
            return path, candidates
    return None, candidates


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def stem(entry):
    return os.path.splitext(os.path.basename(entry["file"]))[0]


def find_weights(entry, weights_dir):
    """Locate the .h5 for a config entry, or None when there is none."""
    names = []
    if entry.get("weights"):
        names.append(entry["weights"])
    names += [f"{stem(entry)}.h5", entry["file"]]
    for name in names:
        for base in (weights_dir, REPO_ROOT, os.path.dirname(os.path.abspath(name))):
            path = name if os.path.isabs(name) else os.path.join(base, name)
            if os.path.isfile(path) and path.lower().endswith(".h5"):
                return path
    return None


def select_entries(entries, wanted):
    if not wanted:
        return list(entries)
    keys = {m.strip().lower() for m in wanted}
    chosen = [e for e in entries
              if e.get("name", "").lower() in keys or stem(e).lower() in keys]
    if not chosen:
        names = ", ".join(f"{e.get('name', stem(e))} ({stem(e)})" for e in entries)
        raise SystemExit(f"no config entry matches {wanted}; available: {names}")
    return chosen


def base_record(entry, mode, output):
    size = entry.get("input_size") or [224, 224]
    return {
        "name": entry.get("name") or stem(entry),
        "file": stem(entry) + ".tflite",
        "input_size": [int(size[0]), int(size[1])],
        "preprocess": mode,
        "output": output,
        "converted": False,
        "problems": [],
    }


def check_parity(keras_model, tflite_path, mode, labels, output):
    """Run one input tensor through Keras and the .tflite and compare."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    _, height, width, channels = inp["shape"]
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(1, height, width, channels), dtype=np.uint8)
    tensor = apply_preprocess(pixels.astype(np.float32), mode)

    if inp["dtype"] != np.float32:
        raise SystemExit(
            f"{os.path.basename(tflite_path)}: expected a float32 input, got "
            f"{np.dtype(inp['dtype']).name}")

    keras_out = np.asarray(keras_model.predict(tensor, verbose=0))[0]
    interpreter.set_tensor(inp["index"], tensor.astype(np.float32))
    interpreter.invoke()
    lite_out = interpreter.get_tensor(out["index"])[0]

    if keras_out.shape != lite_out.shape:
        raise SystemExit(f"shape mismatch: Keras {keras_out.shape} vs TFLite {lite_out.shape}")

    diff = float(np.max(np.abs(keras_out.astype(np.float64) - lite_out.astype(np.float64))))
    same_class = int(np.argmax(keras_out)) == int(np.argmax(lite_out))

    def label_of(index):
        return labels[index] if index < len(labels) else f"class {index}"

    say(f"        parity: keras={label_of(int(np.argmax(keras_out)))} "
        f"({float(np.max(keras_out)):.4f})  "
        f"tflite={label_of(int(np.argmax(lite_out)))} "
        f"({float(np.max(lite_out)):.4f})  max|diff|={diff:.2e}  [{output}]")
    if labels and len(labels) != len(keras_out):
        say(f"        WARNING: {len(labels)} labels but the model outputs "
            f"{len(keras_out)} values")
    if not same_class or diff > 1e-3:
        raise SystemExit(f"{os.path.basename(tflite_path)} does not match its Keras source")
    return diff


def convert_entry(entry, args, tf, labels):
    """Convert one entry (or just inspect it in --check-only). Returns a record."""
    name = entry.get("name") or stem(entry)
    tflite_path = os.path.join(args.out_dir, stem(entry) + ".tflite")
    tflite_name = os.path.basename(tflite_path)

    notes = []

    def note(msg):
        notes.append(msg)
        say(f"        note: {name}: {msg}")

    try:
        mode = normalize_mode(args.set_preprocess or entry.get("preprocess"),
                              lambda m: note(m))
        output = normalize_output(args.set_output or entry.get("output"),
                                  lambda m: note(m))
    except ValueError as exc:
        say(f"ERROR   {name}: {exc}")
        record = base_record(entry, entry.get("preprocess") or DEFAULT_PREPROCESS,
                             entry.get("output") or "softmax")
        record["problems"].append(f"{name}: {exc}")
        return record

    record = base_record(entry, mode, output)
    weights = find_weights(entry, args.weights_dir)

    if args.check_only:
        if not os.path.isfile(tflite_path):
            say(f"MISSING {name}: {tflite_path} not found")
            record["problems"].append(f"{name}: no .tflite at {tflite_path}")
            return record
        say(f"OK      {name}: {tflite_name} (preprocess={mode}, output={output})")
        if weights:
            try:
                keras_model = tf.keras.models.load_model(weights, compile=False)
                check_parity(keras_model, tflite_path, mode, labels, output)
                del keras_model
                gc.collect()
            except SystemExit as exc:
                record["problems"].append(f"{name}: {exc}")
                say(f"        FAILED: {exc}")
        record["converted"] = os.path.isfile(tflite_path)
        return record

    if weights is None:
        say(f"SKIP    {name}: no {stem(entry)}.h5 in {short(args.weights_dir)}")
        return record

    say(f"CONVERT {name}: {short(weights)} -> {tflite_name}")
    try:
        keras_model = tf.keras.models.load_model(weights, compile=False)
    except Exception as exc:  # noqa: BLE001 - report the loader's own message
        say(f"ERROR   {name}: cannot load {weights}: {exc}")
        record["problems"].append(f"{name}: cannot load {weights}: {exc}")
        return record

    # The graph is the truth: if it normalises its own input, the app must not
    # normalise again.
    baked_mode, detail = detect_baked_normalization(keras_model)
    if baked_mode and not args.set_preprocess:
        if mode != baked_mode:
            note(f"{detail}; switching app preprocess {mode} -> {baked_mode} "
                 f"(otherwise it is applied twice)")
            mode = baked_mode
        else:
            note(detail)

    detected_output = detect_output_mode(keras_model)
    if not args.set_output and detected_output != output:
        note(f"last layer is {detected_output}, config said {output}; "
             f"using {detected_output}")
        output = detected_output

    record["preprocess"] = mode
    record["output"] = output

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    try:
        with capture_stderr() as log:
            tflite_bytes = converter.convert()
    except Exception as exc:  # noqa: BLE001
        log.seek(0)
        detail = log.read().strip()
        if detail:
            say(detail[-2000:])
        say(f"ERROR   {name}: conversion failed: {exc}")
        record["problems"].append(f"{name}: conversion failed: {exc}")
        return record

    os.makedirs(args.out_dir, exist_ok=True)
    with open(tflite_path, "wb") as out:
        out.write(tflite_bytes)
    del tflite_bytes

    try:
        check_parity(keras_model, tflite_path, mode, labels, output)
    except SystemExit as exc:
        say(f"        FAILED: {exc}")
        record["problems"].append(f"{name}: {exc}")
        return record

    record["converted"] = True
    say(f"        wrote {os.path.getsize(tflite_path) / 1024:.0f} KB "
        f"(preprocess={mode}, output={output})")

    del keras_model
    tf.keras.backend.clear_session()
    gc.collect()
    return record


def child_argv(entry, args, record_path):
    """Command line for one model converted in its own process."""
    argv = [sys.executable, os.path.abspath(__file__),
            "--config", args.config_abs,
            "--weights-dir", args.weights_dir,
            "--out-dir", args.out_dir,
            "--model", stem(entry),
            "--in-process",
            "--record-json", record_path]
    if args.labels_abs:
        argv += ["--labels", args.labels_abs]
    if args.set_preprocess:
        argv += ["--set-preprocess", args.set_preprocess]
    if args.set_output:
        argv += ["--set-output", args.set_output]
    return argv


def run_children(entries, args):
    """One process per model: a big conversion should not sink the others."""
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        for entry in entries:
            record_path = os.path.join(tmp, f"{stem(entry)}.json")
            code = subprocess.call(child_argv(entry, args, record_path))
            if os.path.isfile(record_path):
                with open(record_path, "r", encoding="utf-8") as f:
                    records.append(json.load(f))
            else:
                record = base_record(entry, entry.get("preprocess") or DEFAULT_PREPROCESS,
                                     entry.get("output") or "softmax")
                reason = ("the conversion process ran out of memory" if code < 0
                          else f"the conversion process exited with code {code}")
                say(f"ERROR   {record['name']}: {reason}")
                record["problems"].append(f"{record['name']}: {reason}")
                records.append(record)
    return records


def main(argv=None):
    args = parse_args(argv)
    child_mode = args.record_json is not None

    config_path = resolve_config_path(args.config)
    if not os.path.isfile(config_path):
        raise SystemExit(
            f"no model config at {config_path}.\n"
            "Train first (python training/train_models.py --data-dir ...), which writes "
            "models_config.json, or pass --config with your own file.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    entries = config.get("models") or []
    if not entries:
        raise SystemExit(f"{config_path} lists no models under \"models\"")

    args.config_abs = config_path
    args.weights_dir = os.path.abspath(args.weights_dir)
    args.out_dir = os.path.abspath(args.out_dir)

    config_out = os.path.abspath(args.config_out) if args.config_out else (
        config_path if os.path.dirname(config_path) == ASSETS else DEFAULT_CONFIG)

    labels_path, label_candidates = resolve_labels_path(args.labels, config, config_path)
    args.labels_abs = labels_path
    labels = load_labels(labels_path) if labels_path else []
    if not labels and not child_mode:
        say("WARNING: no class list found. Looked at:")
        for path in label_candidates:
            say(f"         {path}")
        say("         Pass --labels. The app will not name classes without one.")

    selected = select_entries(entries, args.model)

    if not child_mode:
        say(f"config:  {short(config_path)}")
        say(f"labels:  {labels_path and short(labels_path) or '(none)'} ({len(labels)} classes)")
        say(f"weights: {short(args.weights_dir)}")
        say(f"out:     {short(args.out_dir)}")
        say()

    one_process = child_mode or args.in_process or len(selected) == 1
    if one_process:
        try:
            import tensorflow as tf
        except ImportError:
            raise SystemExit(
                "TensorFlow is not installed. Run: pip install tensorflow\n"
                "(TensorFlow Lite cannot be produced without it.)")
        records = [convert_entry(e, args, tf, labels) for e in selected]
    else:
        # No TensorFlow here: the children load it, and a parent holding a few
        # hundred MB of it would compete with them for memory.
        records = run_children(selected, args)

    if child_mode:
        with open(args.record_json, "w", encoding="utf-8") as f:
            json.dump(records[0], f)
        return 0 if not records[0]["problems"] else 1

    if args.check_only:
        problems = [p for r in records for p in r["problems"]]
        say()
        for problem in problems:
            say(f"PROBLEM {problem}")
        if problems:
            return 1
        say("All models present and matching their Keras sources.")
        return 0

    converted = sum(1 for r in records if r["converted"])
    found_any = any(find_weights(e, args.weights_dir) for e in selected)
    problems = [p for r in records for p in r["problems"]]

    if converted == 0 and not args.force:
        say()
        if not found_any:
            say("Nothing converted: no .h5 weights were found.")
            say(f"Put your trained weights in {short(args.weights_dir)}/ "
                "using the names in the config, or point --weights-dir at them.")
            say("Training from scratch: python training/train_models.py --data-dir <seg_train>")
            say("The app config was left unchanged.")
            return 0
        for problem in problems:
            say(f"PROBLEM {problem}")
        say()
        say("Nothing converted - see the problems above. The app config was left unchanged.")
        return 1

    # Merge the per-model results back into one config, in the original order.
    by_name = {r["name"]: r for r in records}
    out_entries = []
    for entry in entries:
        name = entry.get("name") or stem(entry)
        record = by_name.get(name)
        if record is None:
            size = entry.get("input_size") or [224, 224]
            record = base_record(entry, entry.get("preprocess") or DEFAULT_PREPROCESS,
                                 entry.get("output") or "softmax")
        out_entries.append({k: record[k] for k in
                            ("name", "file", "input_size", "preprocess", "output")})

    os.makedirs(os.path.dirname(config_out) or ".", exist_ok=True)
    out_config = {"classes_file": config.get("classes_file", "classes.txt"),
                  "models": out_entries}
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(out_config, f, indent=2)
        f.write("\n")
    say()
    say(f"wrote {short(config_out)}")

    if labels_path:
        labels_out = os.path.abspath(args.labels_out)
        os.makedirs(os.path.dirname(labels_out) or ".", exist_ok=True)
        if labels_path != labels_out:
            shutil.copyfile(labels_path, labels_out)
        say(f"wrote {short(labels_out)} ({len(labels)} classes)")

    say()
    say(f"{converted} model(s) converted. Rebuild the app in Android Studio.")
    if problems:
        for problem in problems:
            say(f"PROBLEM {problem}")
        return 1
    say("Run tools/verify_parity.py --image <photo> to check one model end to end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
