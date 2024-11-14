import argparse
import os
import random as rnd
import sys

import matplotlib.pyplot as plt
import tensorflow as tf


def preprocess_image(image_path: str, label: int, target_size=(256, 256)) -> tuple:
    """
    Convert image path to array and format it.
    Parameters
    ----------
    image_path : Path of the image to convert and process.
    label : Associated label to the given image.
    target_size : Size used to reformat the image.

    Returns
    -------
    Image and label as a tuple.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def create_dataset(
    image_paths: list[str], labels: list[int], batch_ratio=1, shuffle=True, buffer_size=1000
) -> tf.data.Dataset:
    """
    Build dataset into a readable format for tensorflow.
    Parameters
    ----------
    image_paths : list of path.
    labels : list of associated labels.
    batch_ratio : ratio used to reduce the dataset size.
    shuffle : Shuffle the dataset or not.
    buffer_size : Maximum size of the buffer.

    Returns
    -------
    Dataset as a tf.data.Dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(int(batch_ratio ** (-1))).prefetch(tf.data.AUTOTUNE)
    return dataset


def get_images_path(dir_path: str) -> list[str]:
    """
    Get, from the given path, all the images in the given directory.
    Parameters
    ----------
    dir_path : Path of the directory.

    Returns
    -------
    All images in the given directory (as a list of path).
    """
    images = []

    if not os.path.isdir(dir_path):
        raise Exception(">>> Error: Invalid path")

    for subdir in os.listdir(dir_path):
        subdir_path = os.path.join(dir_path, subdir)
        for filename in os.listdir(subdir_path):
            image_path = os.path.join(subdir_path, filename)
            images.append(image_path)

    return images


def get_labels(images_path: list[str]) -> dict[str, int]:
    """
    Get all unique labels from given path. We consider as label all directories that contain images.
    Parameters
    ----------
    images_path : list of paths.

    Returns
    -------
    A dictionary with unique str labels as keys and integer labels as values.
    """
    labels = list(set([p.split("/")[-2] for p in images_path]))
    labels_dict = {d: i for d, i in zip(labels, range(len(labels)))}
    return labels_dict


def labelise_images(images_path: list[str], labels_dict: dict[str, int]) -> dict[str, list]:
    """
    Associate all path with its corresponding label.
    Parameters
    ----------
    images_path : List of path.
    labels_dict : Dict of labels.

    Returns
    -------
    Dictionary of labeled paths.
    """
    return {img_path: [labels_dict[img_path.split("/")[-2]]] for img_path in images_path}


def sample_labelised_images(
    labeled_images: dict[str, list], ratio: float = 0.8
) -> tuple[dict[str, list], dict[str, list]]:
    """
    Split the given labeled images into two dict  one for the testing process and the other for the training process.
    Parameters
    ----------
    labeled_images : Dictionary to split.
    ratio : (float between 0 and 1) The ratio of training and testing images.

    Returns
    -------
    A tuple with the training images and the testing images.
    """
    sample_keys = rnd.sample(list(labeled_images.keys()), int(len(labeled_images) * ratio))

    first_sample = {k: labeled_images[k] for k in sample_keys}
    second_sample = {k: labeled_images[k] for k in labeled_images.keys() if k not in sample_keys}

    return first_sample, second_sample


def build_datasets(
    dir_path: str, batch_ratio: int = 1
) -> tuple[tf.data.Dataset, tf.data.Dataset, dict[str, int]]:
    """
    Prepare the dataset for training and testing process.
    Parameters
    ----------
    dir_path : Path of the directory.
    batch_ratio : Size of the batch (can be used to reduce the dataset size).

    Returns
    -------

    """
    img_list = get_images_path(dir_path)
    labels = get_labels(img_list)
    labeled_images = labelise_images(img_list, labels)

    train_sample, test_sample = sample_labelised_images(labeled_images, ratio=0.8)

    train_image_paths = list(train_sample.keys())
    train_labels = [label[0] for label in train_sample.values()]

    test_image_paths = list(test_sample.keys())
    test_labels = [label[0] for label in test_sample.values()]

    train_dataset = create_dataset(
        train_image_paths, train_labels, batch_ratio=batch_ratio, shuffle=True
    )
    test_dataset = create_dataset(
        test_image_paths, test_labels, batch_ratio=batch_ratio, shuffle=False
    )

    return train_dataset, test_dataset, labels


def load_model(input_size: int, output_size: int) -> tf.keras.Model:
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
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
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


def train_model(
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
        train_dataset, epochs=epoch, validation_data=test_dataset, callbacks=[early_stopping]
    )
    return model, hist


def plot_training(hist) -> None:
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
        "--batch_ratio",
        type=float,
        default=1,
        help="Used to reduce the dataset size.",
    )
    parser.add_argument("-p", "--plot", action="store_true", help="show the training history")
    parser.add_argument("-s", "--save_model", action="store_true", help="save the trained model")
    return parser


if __name__ == "__main__":
    try:
        tf.keras.backend.clear_session()

        args = options_parser().parse_args()

        train_dataset, test_dataset, labels = build_datasets(args.directory_path[0], batch_ratio=args.batch_ratio)

        model = load_model(256, len(labels))

        model, history = train_model(model, train_dataset, test_dataset, epoch=args.epochs)

        if args.plot:
            plot_training(history)

        test_loss, test_acc = model.evaluate(test_dataset)
        print("Test accuracy:", test_acc)

        if args.save_model:
            model.save(args.save_dir + "/model.keras")
            model.save_weights(args.save_dir + "/model.weights.h5")

    except Exception as e:
        print(">>> Oups something went wrong.", file=sys.stderr)
        print(e, file=sys.stderr)
