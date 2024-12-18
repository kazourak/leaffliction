import argparse
import json
import os
import sys
from typing import Any

from Transformation import get_mask
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tools.init_tf_env import init_tf_env

RED = "\033[0;31m"
GREEN = "\033[0;32m"

NO_COLOR = "\033[0m"


def _preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocesses the image for prediction.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    target_size : tuple, optional
        The target size to resize the image, by default (256, 256)

    Returns
    -------
    numpy.ndarray
        Preprocessed image ready for prediction.
    """
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    return image


def options_parser() -> argparse.ArgumentParser:
    """
    Use to handle program parameters and options.

    Returns
    -------
    The parser object.
    """
    parser = argparse.ArgumentParser(
        prog="Distribution",
        description="This program predicts the labels for all images in a given directory.",
        epilog="Ensure the model path and directory are correctly specified.",
    )
    parser.add_argument("model_path", type=str, nargs=1, help="Path to the trained model.")
    parser.add_argument(
        "images_path", type=str, nargs=1, help="Path to the directory containing images."
    )
    parser.add_argument("-p", "--plot", action="store_true", help="show the training history")
    return parser


def _predict(model: Any, labels: dict, files: list[str], plot: bool = False):
    """
    Predict a label for each image from the files path list.

    Parameters
    ----------
    model : Model trained used to predict labels.
    labels : All possible labels.
    files : List of images path.
    plot : bool, default False, plot images to predict.

    Returns
    -------

    """
    for file_path in files:

        if os.path.isfile(file_path) and file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                image = _preprocess_image(file_path)

                prediction = model.predict(np.expand_dims(image, axis=0))

                img_prediction = prediction[0]

                idx = np.argmax(img_prediction)
                label = labels[idx]

                basename = str(os.path.basename(file_path))
                print(f"Image: {os.path.basename(file_path)}")
                print(
                    f"Predicted: {GREEN if label in basename else RED}{label}{NO_COLOR}"
                    + f" with confidence {img_prediction[idx] * 100:.2f}%\n"
                )

                if plot:
                    fig, ax = plt.subplots(ncols=2)
                    fig.canvas.manager.set_window_title(
                        basename + f" Predicted: {label} ({img_prediction[idx] * 100:.2f}%)"
                    )
                    ax[0].imshow(cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB))
                    ax[1].imshow(cv2.cvtColor(get_mask(file_path), cv2.COLOR_BGR2RGB))
                    plt.show()

            except Exception as img_error:
                print(f"Error processing image {os.path.basename(file_path)}: {img_error}")


if __name__ == "__main__":
    try:
        init_tf_env()

        args = options_parser().parse_args()

        model_path = args.model_path[0]
        images_path = args.images_path[0]

        model = tf.keras.models.load_model(model_path)

        with open("models/labels.json", "r") as f:
            labels = json.load(f)

        if not os.path.isdir(images_path) and not os.path.isfile(images_path):
            raise FileNotFoundError(f"The directory or file {images_path} does not exist.")

        files_path = (
            [os.path.join(images_path, filename) for filename in os.listdir(images_path)]
            if os.path.isdir(images_path)
            else [images_path]
        )

        _predict(model, labels, files_path, plot=args.plot)

    except Exception as e:
        print(">>> Oops, something went wrong.", file=sys.stderr)
        print(e)
