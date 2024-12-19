import os
import random

import cv2
import numpy as np

from leaffliction.Augmentation import augment


def process_file(file: str, src_path: str, dest_path: str, nb_images: int) -> list[str]:
    """
    Augment the given image.

    Parameters
    ----------
    file : Image path.
    src_path : Source path.
    dest_path : Destination path.
    nb_images : Number of images to save.

    Returns
    -------
    Each path of the new images created and saved.
    """

    i = 1
    paths_saved = []
    _image = path_to_img(file)
    _augmented_files = augment(_image, 1)

    for _type, _img in _augmented_files.items():
        _filename = build_filename(
            file, src_path, dest_path, _type + str(int(random.random() * 1000000))
        )
        save_image(_img, _filename)
        paths_saved.append(_filename)
        if i == nb_images:
            break
        i += 1

    return paths_saved


def path_to_img(path: str, size: tuple = (256, 256)):
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

    try:
        cv2.imwrite(file_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Error saving image {file_name}")
        raise ValueError(f"Error saving image {file_name} {e}")
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
