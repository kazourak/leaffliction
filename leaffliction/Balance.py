"""
Module: Balance

This module provides functionalities to balance datasets by copying and processing image files.
It includes functions to copy directories, process images, and manage distributions to ensure
a balanced distribution of data for machine learning tasks.
"""

import argparse
import os
import random
import sys

from Distribution import distribution
from tools.process_image import build_filename, path_to_img, process_file, save_image

CUSTOM_BAR = (
    "{l_bar}"
    "\033[92m{bar}\033[0m"
    "| {n_fmt}/{total_fmt} "
    "[{percentage:3.0f}%] "
    "[Elapsed: {elapsed}]"
)


def copy_dir(dir_path: str, source_path: str, destination_path: str):
    """
    Copy all files from dir_path to destination_path.
    Parameters
    ----------
    dir_path : Path of the directory to copy.
    source_path : Main source directory.
    destination_path : Destination directory.

    Returns
    -------

    """
    images_path = [
        os.path.join(dir_path, image_path)
        for image_path in os.listdir(dir_path)
        if image_path.lower().endswith("jpg")
    ]

    for image_path in images_path:
        try:
            _image = path_to_img(image_path)
            _new_path = build_filename(image_path, source_path, destination_path, "Original")
            save_image(_image, _new_path)
        except Exception as e:
            print(e)


def augment_directory(dir_name: str, source_path: str, destination_path: str, max_image: int):
    """
    Augment random picked image from a source directory to match the maximum number of images
    required.
    Parameters
    ----------
    dir_name : Name of the directory to augment.
    source_path : Source directory.
    destination_path : Destination directory.
    max_image : Maximum number of images to augment.

    Returns
    -------

    """
    print(dir_name)
    dir_path = os.path.join(source_path, dir_name)
    dest_dir = os.path.join(destination_path, dir_name)
    previous_images = [os.path.join(dir_path, image_path) for image_path in os.listdir(dir_path)]
    while max_image - len(os.listdir(dest_dir)) > 0:
        print(f"missing {max_image - len(os.listdir(dest_dir))}")
        try:
            _image = random.choice(previous_images)
            process_file(
                _image, source_path, destination_path, max_image - len(os.listdir(dest_dir))
            )
        except Exception as e:
            print(e)


def options_parser() -> argparse.ArgumentParser:
    """
    Use to handle program parameters and options.
    Returns
    -------
    The parser object.
    """

    parser = argparse.ArgumentParser(
        prog="balance",
        description="This program should be used to augment image from the dataset to have"
        + " for each class the same number of images.",
        epilog="Please read the subject and the README before proceeding to understand the"
        + " input file format.",
    )
    parser.add_argument("directory_path", type=str, nargs=1)
    parser.add_argument("destination_path", type=str, nargs=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
        help="Used to reproduce a dataset split.",
    )
    parser.add_argument(
        "--multiply_factor",
        type=float,
        default=1,
        help="Factor used to multiply the number of images in the dataset.",
    )
    return parser


if __name__ == "__main__":
    try:
        args = options_parser().parse_args()
        random.seed(args.seed)

        source_directory = args.directory_path[0]
        destination_directory = args.destination_path[0]

        if not os.path.isdir(destination_directory):
            os.mkdir(destination_directory)

        distribution_dict = distribution(dir_path=source_directory)
        # Get the dir with the maximum number of files
        dir_name, dir_size = sorted(
            distribution_dict.items(), key=lambda item: item[1], reverse=True
        )[0]
        print(dir_name, dir_size)

        # Step 1 duplicate images to the destination directory
        print(f"Copying images to {destination_directory}...")
        for dir in distribution_dict.keys():
            print(dir)
            copy_dir(os.path.join(source_directory, dir), source_directory, destination_directory)
        print("Done.")

        # Step 2 Augment images for each directory that has not enough files.
        print(f"Augment images to have {dir_size * args.multiply_factor} images of each class...")
        for dir in distribution_dict.keys():
            print(dir)
            augment_directory(
                dir, source_directory, destination_directory, dir_size * args.multiply_factor
            )
        print("Done.")

    except Exception as e:
        print(">>> Oups something went wrong.", file=sys.stderr)
        print(e, file=sys.stderr)
