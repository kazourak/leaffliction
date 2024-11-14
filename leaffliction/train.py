import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


import os
import sys
import random as rnd
import numpy as np
import cv2

import argparse

from tensorflow.python.ops.numpy_ops.np_random import random


def get_images_path(dir_path: str) -> list[str]:
    images = []

    if not os.path.isdir(dir_path):
        raise Exception(">>> Error: Invalid path")

    for subdir in os.listdir(dir_path):
        subdir_path = os.path.join(dir_path, subdir)
        for filename in os.listdir(subdir_path):
            image_path = os.path.join(subdir_path, filename)
            images.append(image_path)

    return images


def get_labels(images_path:list[str])->dict[str,int]:
    labels = list(set([p.split('/')[-2] for p in images_path]))
    labels_dict = {d: i for d, i in zip(labels, range(len(labels)))}
    return labels_dict


def labelise_images(images_path: list[str], labels_dict: dict[str, int]) -> dict[str, int]:
    return {img_path: labels_dict[img_path.split('/')[-2]] for img_path in images_path}


def load_images(images_path:list[str])->list[np.ndarray]:
    return [list(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)) for img_path in images_path]


def sample_labelised_images(labelised_images: dict[str, int], ratio: float = 0.8) -> tuple[
    dict[str, int], dict[str, int]]:
    sample_keys = rnd.sample(list(labelised_images.keys()), int(len(labelised_images) * ratio))

    first_sample = {k: labelised_images[k] for k in sample_keys}
    second_sample = {k: labelised_images[k] for k in labelised_images.keys() if k not in sample_keys}

    return first_sample, second_sample


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
    return parser


if __name__ == "__main__":
    try:
        args = options_parser().parse_args()

        img_list = get_images_path(args.directory_path[0])
        labels = get_labels(img_list)
        labelised_images = labelise_images(img_list, labels)

        train_sample, test_sample = sample_labelised_images(labelised_images, ratio=0.8)

        train_images = load_images(train_sample.keys())
        train_labels = list(train_sample.values())

        test_images = load_images(test_sample.keys())
        test_labels = list(test_sample.values())

        class_names = list(labels.keys())

        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i])
            # The CIFAR labels happen to be arrays,
            # which is why you need the extra index
            plt.xlabel(class_names[train_labels[i]])
        plt.show()

        tf.keras.backend.clear_session()

        model = models.Sequential()
        model.add(layers.Rescaling(1.0 / 255))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(32, (1, 1), activation="relu"))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(len(labels), activation="softmax"))

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        model.summary()

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        history = model.fit(train_images, train_labels, epochs=10,
                            validation_data=(test_images, test_labels))

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

        plt.show()

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

        print(test_acc)

        model.save('model.keras')
        model.save_weights('model.weights.h5')

    except Exception as e:
        print(">>> Oups something went wrong.", file=sys.stderr)
        print(e, file=sys.stderr)