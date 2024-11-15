import argparse
import json
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def preprocess_image(image_path: str, target_size=(256, 256)) -> Any:
    """
    Convert image path to array and format it.
    Parameters
    ----------
    image_path : Path of the image to convert and process.
    target_size : Size used to reformat the image.

    Returns
    -------
    Image.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def create_dataset(image_paths: list[str]) -> tf.data.Dataset:
    """
    Build dataset into a readable format for tensorflow.
    Parameters
    ----------
    image_paths : list of path.

    Returns
    -------
    Dataset as a tf.data.Dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(1).prefetch(tf.data.AUTOTUNE)
    return dataset


def options_parser() -> argparse.ArgumentParser:
    """
    Use to handle program parameters and options.
    Returns
    -------
    The parser object.
    """

    parser = argparse.ArgumentParser(
        prog="Distribution",
        description="This program should be used to see the distribution of plant types and states.",
        epilog="Please read the subject before proceeding to understand the input file format.",
    )
    parser.add_argument("model_path", type=str, nargs=1)
    parser.add_argument("image_path", type=str, nargs=1)

    return parser


if __name__ == "__main__":
    try:
        args = options_parser().parse_args()

        model = tf.keras.models.load_model(args.model_path[0])

        model.summary()

        image = preprocess_image(args.image_path[0], target_size=(256, 256))

        print(type(image))

        prediction = model.predict(np.expand_dims(image, axis=0))

        img_prediction = prediction[0]

        with open("models/labels.json", "r") as f:
            labels = json.load(f)

        # get the index of the highest value of img_prediction
        idx = np.argmax(img_prediction)
        # get the nth elements of the dictionary labels
        label = list(labels.keys())[idx]

        print(idx)

        plt.imshow(image)
        plt.title(f"Predicted: {label} at {img_prediction[idx] * 100}")
        plt.show()

    except Exception as e:
        print(">>> Oups something went wrong.", file=sys.stderr)
        print(e)
