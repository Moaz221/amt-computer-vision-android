"""Train the four default models on the Intel Image Classification dataset
and save the weights where the app expects them.

Usage:
    python training/train_models.py --data-dir /path/to/seg_train --epochs 10
    python training/train_models.py --data-dir /path/to/seg_train --models custom_cnn vgg19

The dataset folder must contain one subfolder per class (the standard layout
of https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
Class names are written to classes.txt in the order the generator uses, so
the app and the models always agree on the label order.
"""

import argparse
import json
import os

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_CONFIG = {
    "custom_cnn": {"size": (100, 100), "preprocess": "rescale", "file": "custom_cnn.h5"},
    "alexnet": {"size": (227, 227), "preprocess": "caffe", "file": "alexnet.h5"},
    "vgg19": {"size": (224, 224), "preprocess": "caffe", "file": "vgg19.h5"},
    "inceptionv3": {"size": (299, 299), "preprocess": "tf", "file": "inceptionv3.h5"},
}


def build_custom_cnn(size, num_classes):
    inputs = layers.Input(shape=(*size, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return models.Model(inputs, outputs, name="custom_cnn")


def build_alexnet(size, num_classes):
    inputs = layers.Input(shape=(*size, 3))
    # caffe-style preprocessing (BGR + mean subtraction) is applied at
    # inference time inside predictor.py, so the model sees raw [0, 255] here
    # only if you feed it that way; for training we rescale to [0, 1] instead.
    x = layers.Rescaling(1.0 / 255)(inputs)
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
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return models.Model(inputs, outputs, name="alexnet")


def build_transfer(base_cls, size, num_classes, name):
    base = base_cls(include_top=False, weights="imagenet",
                    input_shape=(*size, 3), pooling="avg")
    base.trainable = False  # feature extraction; unfreeze later if you like
    inputs = layers.Input(shape=(*size, 3))
    x = base(inputs, training=False)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return models.Model(inputs, outputs, name=name)


BUILDERS = {
    "custom_cnn": lambda size, n: build_custom_cnn(size, n),
    "alexnet": lambda size, n: build_alexnet(size, n),
    "vgg19": lambda size, n: build_transfer(
        tf.keras.applications.VGG19, size, n, "vgg19"),
    "inceptionv3": lambda size, n: build_transfer(
        tf.keras.applications.InceptionV3, size, n, "inceptionv3"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True,
                        help="folder with one subfolder per class (seg_train)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--models", nargs="+", default=list(IMG_CONFIG),
                        choices=list(IMG_CONFIG))
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root, "models")
    os.makedirs(models_dir, exist_ok=True)

    config = {"classes_file": "classes.txt", "models": []}
    classes_written = False

    for name in args.models:
        cfg = IMG_CONFIG[name]
        size = cfg["size"]
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
        autotune = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(autotune)
        val_ds = val_ds.prefetch(autotune)

        model = BUILDERS[name](size, num_classes)
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        print(f"\n=== Training {name} ({size[0]}x{size[1]}, {num_classes} classes) ===")
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

        out_path = os.path.join(models_dir, cfg["file"])
        model.save(out_path)
        print(f"saved -> {out_path}")

        config["models"].append({
            "name": {"custom_cnn": "Custom CNN", "alexnet": "AlexNet",
                     "vgg19": "VGG-19", "inceptionv3": "InceptionV3"}[name],
            "file": cfg["file"],
            "input_size": list(size),
            "preprocess": cfg["preprocess"],
        })

    with open(os.path.join(root, "models_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print("\nmodels_config.json and classes.txt updated - the app is ready.")


if __name__ == "__main__":
    main()
