import argparse
import os
import random
import sys

from Distribution import distribution
from tools.improve_dataset import (build_filename, path_to_img, process_file,
                                   save_image)

CUSTOM_BAR = (
    "{l_bar}"
    "\033[92m{bar}\033[0m"
    "| {n_fmt}/{total_fmt} "
    "[{percentage:3.0f}%] "
    "[Elapsed: {elapsed}]"
)


def get_random(elems: list, number: int) -> list:

    kept_elements = []

    while len(kept_elements) < number:
        _element = random.choice(elems)
        if _element not in kept_elements:
            kept_elements.append(_element)

    return kept_elements


def random_get_files(directories: list, max_number: int) -> list:
    files_selected = []

    for directory in directories:
        _files = get_random(os.listdir(directory), max_number)
        _files = list(dict.fromkeys(_files))
        print(len(_files))
        files_selected.extend(list(map(lambda x: os.path.join(directory, x), _files)))

    return files_selected


def copy_dir(dir_path: str, source_path: str, destination_path: str):
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
        prog="Balance",
        description="This program should be used to balance the dataset.",
        epilog="Please read the subject before proceeding to understand the input file format.",
    )
    parser.add_argument("directory_path", type=str, nargs=1)
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
    parser.add_argument("destination_path", type=str, nargs=1)
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
