import argparse
import json
import sys

import matplotlib.pyplot as plt
import tensorflow as tf

from .tools.build_dataset import build_dataset
from .tools.init_tf_env import init_tf_env


def _load_model(input_size: int, output_size: int) -> tf.keras.Model:
    """
    Build and compile the CNN model.
    Parameters
    ----------
    input_size : Length of input image side.
    output_size : Number of possible labels.

    Returns
    -------
    The compiled CNN model.
    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(input_size, input_size, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(output_size, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


def _train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
    epoch: int = 20,
) -> tuple:
    """
    Train the model with the given dataset.
    Parameters
    ----------
    model :
    train_dataset :
    test_dataset :

    Returns
    -------
    A tuple with the model and the training history.
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    hist = model.fit(
        train_dataset,
        epochs=epoch,
        validation_data=test_dataset,
        callbacks=[early_stopping],
    )
    return model, hist


def _plot_training(hist) -> None:
    """
    Plot the accuracy curve.
    Parameters
    ----------
    hist : History of the learning process.

    Returns
    -------
    None
    """
    plt.plot(hist.history["accuracy"], label="accuracy")
    plt.plot(hist.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")
    plt.show()


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
    parser.add_argument("directory_path", type=str, nargs=1)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models/",
        help="Path to the directory where to save the trained model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="number of epochs to train the model.",
    )
    parser.add_argument(
        "--batch_size",
        type=float,
        default=32,
        help="Used to specify how many images to process in a batch.",
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.2,
        help="Used to specify the ratio of data used for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
        help="Used to reproduce a dataset split.",
    )

    parser.add_argument("-p", "--plot", action="store_true", help="show the training history")
    parser.add_argument("-s", "--save_model", action="store_true", help="save the trained model")
    return parser


if __name__ == "__main__":
    try:
        init_tf_env()

        args = options_parser().parse_args()

        train_dataset, evaluate_dataset = build_dataset(
            data_path=args.directory_path[0],
            batch_size=args.batch_size,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )

        labels = train_dataset.class_names
        print("Class Names:", labels)

        model = _load_model(256, len(labels))

        model, history = _train_model(model, train_dataset, evaluate_dataset, epoch=args.epochs)

        if args.plot:
            _plot_training(history)

        if args.save_model:
            model.save(args.save_dir + "/model.keras")
            model.save_weights(args.save_dir + "/model.weights.h5")
            with open(args.save_dir + "/labels.json", "w", encoding="utf-8") as file:
                json.dump(labels, file)

    except Exception as e:
        print(">>> Oups something went wrong.", file=sys.stderr)
        print(e, file=sys.stderr)
