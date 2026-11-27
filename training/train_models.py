"""Train the four default models on the Intel Image Classification dataset
and save the weights where the app expects them.

Usage:
    python training/train_models.py --data-dir /path/to/seg_train --epochs 10
    python training/train_models.py --data-dir /path/to/seg_train --models custom_cnn vgg19
    python training/train_models.py --data-dir seg_train --output logits

The dataset folder must contain one subfolder per class (the standard layout
of https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

Preprocessing happens in the input pipeline, not inside the model graph, so
what a model is trained on is exactly what the Android app feeds it at
inference time. The modes are defined in common/preprocess.py and mirrored in
the app's ml/Preprocessing.kt.

Outputs:
    models/<key>.h5          the trained weights
    models_config.json       manifest read by tools/convert_to_tflite.py
    classes.txt              class names, in the generator's label order
"""

import argparse
import json
import os
import sys

import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.preprocess import (  # noqa: E402
    DEFAULT_MODELS_BY_KEY,
    apply_preprocess_tf,
    normalize_output,
)


def build_custom_cnn(size, num_classes, output):
    inputs = layers.Input(shape=(*size, 3))
    x = inputs
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    return models.Model(inputs, head(x, num_classes, output), name="custom_cnn")


def build_alexnet(size, num_classes, output):
    inputs = layers.Input(shape=(*size, 3))
    x = inputs
    x = layers.Conv2D(96, 11, strides=4, activation="relu")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.Conv2D(256, 5, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.Conv2D(384, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(384, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    return models.Model(inputs, head(x, num_classes, output), name="alexnet")


def build_transfer(base_cls, size, num_classes, name, output):
    base = base_cls(include_top=False, weights="imagenet",
                    input_shape=(*size, 3), pooling="avg")
    base.trainable = False  # feature extraction; unfreeze later if you like
    inputs = layers.Input(shape=(*size, 3))
    x = base(inputs, training=False)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    return models.Model(inputs, head(x, num_classes, output), name=name)


def head(x, num_classes, output):
    """Final layer: softmax (probabilities) or no activation (logits)."""
    activation = "softmax" if output == "softmax" else None
    return layers.Dense(num_classes, activation=activation, dtype="float32")(x)


BUILDERS = {
    "custom_cnn": lambda size, n, out: build_custom_cnn(size, n, out),
    "alexnet": lambda size, n, out: build_alexnet(size, n, out),
    "vgg19": lambda size, n, out: build_transfer(
        tf.keras.applications.VGG19, size, n, "vgg19", out),
    "inceptionv3": lambda size, n, out: build_transfer(
        tf.keras.applications.InceptionV3, size, n, "inceptionv3", out),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True,
                        help="folder with one subfolder per class (seg_train)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS_BY_KEY),
                        choices=list(DEFAULT_MODELS_BY_KEY),
                        help="which models to train (default: all four)")
    parser.add_argument("--output", default="softmax", choices=["softmax", "logits"],
                        help="final layer: softmax probabilities or raw logits")
    parser.add_argument("--out-dir", default=None,
                        help="where to write the .h5 files (default: <repo>/models)")
    args = parser.parse_args()

    output_mode = normalize_output(args.output)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = args.out_dir or os.path.join(root, "models")
    os.makedirs(models_dir, exist_ok=True)

    config = {"classes_file": "classes.txt", "models": []}
    classes_written = False

    for key in args.models:
        cfg = DEFAULT_MODELS_BY_KEY[key]
        size = tuple(cfg["input_size"])
        mode = cfg["preprocess"]

        train_ds = tf.keras.utils.image_dataset_from_directory(
            args.data_dir, image_size=size, batch_size=args.batch_size,
            validation_split=0.2, subset="training", seed=42)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            args.data_dir, image_size=size, batch_size=args.batch_size,
            validation_split=0.2, subset="validation", seed=42)

        if not classes_written:
            with open(os.path.join(root, "classes.txt"), "w") as f:
                f.write("\n".join(train_ds.class_names) + "\n")
            classes_written = True

        num_classes = len(train_ds.class_names)

        def prep(images, labels, mode=mode):
            # Same maths the app runs before the interpreter, in the pipeline.
            return apply_preprocess_tf(images, mode), labels

        autotune = tf.data.AUTOTUNE
        train_ds = train_ds.map(prep, num_parallel_calls=autotune).prefetch(autotune)
        val_ds = val_ds.map(prep, num_parallel_calls=autotune).prefetch(autotune)

        model = BUILDERS[key](size, num_classes, output_mode)
        loss = ("sparse_categorical_crossentropy" if output_mode == "softmax"
                else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
        print(f"\n=== Training {cfg['name']} ({size[0]}x{size[1]}, {num_classes} classes, "
              f"preprocess={mode}, output={output_mode}) ===")
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

        out_path = os.path.join(models_dir, cfg["h5"])
        model.save(out_path)
        print(f"saved -> {out_path}")

        config["models"].append({
            "name": cfg["name"],
            "file": cfg["h5"],
            "input_size": list(size),
            "preprocess": mode,
            "output": output_mode,
        })

    with open(os.path.join(root, "models_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("\nmodels_config.json and classes.txt written.")
    print("Next: python tools/convert_to_tflite.py")


if __name__ == "__main__":
    main()
