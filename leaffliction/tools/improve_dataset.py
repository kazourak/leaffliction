import os
import sys

import cv2
import numpy as np

from leaffliction import augment
from leaffliction.tools import extract_files


def improve_dataset(path: str, dir_dest: str) -> list[str]:
    """
    Augment each image from  the given path and save them into the directory dir_dest.
    Parameters
    ----------
    path : Source path.
    dir_dest : Destination path.

    Returns
    -------
    All new files as a list of paths.
    """

    files = extract_files(path)

    new_files = []

    for file in files:
        _image = path_to_img(file)
        _augmented_files = augment(_image, 1)
        for _type, _img in _augmented_files.items():
            print('.', end='', flush=True)
            _filename = build_filename(file, path, dir_dest, _type)
            save_image(_img, _filename)
            new_files.append(_filename)

        _filename = build_filename(file, path, dir_dest, "original")
        save_image(_image, _filename)
        new_files.append(_filename)

    return new_files


def path_to_img(path: str, size: tuple = (256, 256)) -> np.ndarray:
    """
    Load and format the image from the given path.
    Parameters
    ----------
    path : Path of the image to load.
    size : Size of the image to load (resizes it if needed).

    Returns
    -------
    The loaded image as a np.ndarray.
    """
    _image = cv2.imread(path)
    _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
    _image = cv2.resize(_image, size)
    return _image


def save_image(image: np.ndarray, file_name: str) -> str:
    """
    Save the given image at the given path.
    Parameters
    ----------
    image : Image to save.
    file_name : Path.

    Returns
    -------
    The path of the new image.
    """

    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    cv2.imwrite(file_name, image)
    return file_name


def build_filename(file_src: str, dir_src: str, dir_dest: str, additional_label: str = "") -> str:
    """
    Extract info from the source to build the new path.
    Parameters
    ----------
    file_src : File path.
    dir_src :  Source path.
    dir_dest : Destination path.
    additional_label : Label to add at the end of the filename. (dir/filename_label.txt)

    Returns
    -------
    The new path as a string.
    """
    _filename = os.path.relpath(file_src, os.path.commonprefix([dir_src, file_src]))
    _filename_split = os.path.splitext(_filename)
    return os.path.normpath(
        f"{dir_dest}/{_filename_split[0]}_{additional_label}{_filename_split[1]}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python improve_dataset.py <source_path> <dir_dest>")
        exit()

    dir_path = sys.argv[1]
    dir_dest = sys.argv[2]

    if len(dir_path) == 0 or len(dir_dest) == 0:
        print("Usage: python improve_dataset.py <source_path> <dir_dest>")
        exit()

    if not os.path.isdir(dir_path):
        print("Directory does not exist.")
        exit()

    if not os.path.isdir(dir_dest):
        os.mkdir(dir_dest)

    new_files = improve_dataset(dir_path, dir_dest)
    print(new_files)
