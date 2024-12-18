import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import tensorflow as tf
from tools.build_dataset import build_dataset
from tools.init_tf_env import init_tf_env


def _load_model(input_size: int, output_size: int) -> tf.keras.Model:
    """
    Build and compile the CNN model from tensorflow tutorial.
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
            # Extract features layers.
            tf.keras.layers.Input(shape=(input_size, input_size, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            # Hidden layer (learn).
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation="relu"),
            # Use soft max to extract the highest result (the predicted label).
            tf.keras.layers.Dense(output_size, activation="softmax"),
        ]
    )
    model.compile(
        # Function used to optimise weights during the training session.
        optimizer="adam",
        # Function used to evaluate the result.
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
    # Small callback function used to stop the training process when the validation loss is lower
    # than the previous one three times in a row. If this function stops the training it restores
    # the weights (from the third back).
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    # Train the model.
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


def _save(
    model: tf.keras.models.Sequential,
    train_dataset: tf.data.Dataset,
    evaluate_dataset: tf.data.Dataset,
    save_dir: str,
    save_model: bool,
    save_dataset: bool,
):
    """
    Save the trained model and the datasets if the user wants it.
    Parameters
    ----------
    model : trained model.
    train_dataset : Dataset used to train the model.
    evaluate_dataset : Dataset used to evaluate the model.
    save_dir : Path to save.
    save_model : If the user wants to save the model.
    save_dataset : If the user wants to save the datasets.

    Returns
    -------

    """
    if save_model:
        model_dir = os.path.join(save_dir, "model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save(model_dir + "/model.keras")
        model.save_weights(model_dir + "/model.weights.h5")
        with open(model_dir + "/labels.json", "w", encoding="utf-8") as file:
            json.dump(labels, file)

    if save_dataset:
        dataset_dir = os.path.join(save_dir, "dataset")
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        train_dataset.save(dataset_dir + "/train_dataset.tfrecords")
        evaluate_dataset.save(dataset_dir + "/evaluate_dataset.tfrecords")


def options_parser() -> argparse.ArgumentParser:
    """
    Use to handle program parameters and options.
    Returns
    -------
    The parser object.
    """

    parser = argparse.ArgumentParser(
        prog="Distribution",
        description="This program should be used to see the distribution of plant types and "
        + "states.",
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
    parser.add_argument("--save_model", action="store_true", help="save the trained model")
    parser.add_argument(
        "--save_dataset",
        action="store_true",
        help="save the dataset used to train evaluate the model.",
    )
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

        _save(
            model,
            train_dataset,
            evaluate_dataset,
            args.save_dir,
            args.save_model,
            args.save_dataset,
        )

    except Exception as e:
        print(">>> Oups something went wrong.", file=sys.stderr)
        print(e, file=sys.stderr)
        raise ValueError("Error") from e
