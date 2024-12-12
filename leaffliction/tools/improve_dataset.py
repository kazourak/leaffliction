import argparse
import concurrent.futures
import os
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from leaffliction import augment
from leaffliction.tools import extract_files

CUSTOM_BAR = (
    "{l_bar}"
    "\033[92m{bar}\033[0m"
    "| {n_fmt}/{total_fmt} "
    "[{percentage:3.0f}%] "
    "[Elapsed: {elapsed}]"
)


def process_file(file: str, src_path: str, dest_path: str) -> list[str]:
    """
    Augment the given image.

    Parameters
    ----------
    file : Image path.
    src_path : Source path.
    dest_path : Destination path.

    Returns
    -------
    Each path of the new images created and saved.
    """

    paths_saved = []
    _image = path_to_img(file)
    _augmented_files = augment(_image, 1)

    for _type, _img in _augmented_files.items():
        _filename = build_filename(file, src_path, dest_path, _type)
        save_image(_img, _filename)
        paths_saved.append(_filename)

    _filename = build_filename(file, src_path, dest_path, "Original")
    save_image(_image, _filename)
    paths_saved.append(_filename)

    return paths_saved


def improve_dataset(path: str, dir_dest: str) -> List[str]:
    """
    Augment each image from the given path and save them into the directory dir_dest.

    Parameters
    ----------
    path : str
        Source path.
    dir_dest : str
        Destination path.

    Returns
    -------
    List[str]
        All new files as a list of paths.
    """
    files = extract_files(path)
    print(f"Found {len(files)} files to process.")
    new_files = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        with tqdm(total=len(files), bar_format=CUSTOM_BAR) as progress_bar:
            future_to_file = {
                executor.submit(process_file, file, path, dir_dest): file for file in files
            }
            for future in concurrent.futures.as_completed(future_to_file):
                try:
                    result = future.result()
                    new_files.extend(result)
                except Exception as e:
                    print(f"Error processing file {future_to_file[future]}: {e}")
                progress_bar.update(1)

    return new_files


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

    cv2.imwrite(file_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
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


def options_parser() -> argparse.ArgumentParser:
    """
    Use to handle program parameters and options.
    Returns
    -------
    The parser object.
    """

    parser = argparse.ArgumentParser(
        prog="Improve dataset",
        description="This program should be used to improve the existent dataset.",
        epilog="Please read the subject before proceeding to understand the input file format.",
    )
    parser.add_argument("source_path", type=str, nargs=1)
    parser.add_argument("destination_path", type=str, nargs=1)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print the list of new files created."
    )

    return parser


if __name__ == "__main__":
    args = options_parser().parse_args()

    if not os.path.isdir(args.destination_path[0]):
        os.mkdir(args.destination_path[0])

    new_files = improve_dataset(args.source_path[0], args.destination_path[0])
    if args.verbose:
        print(new_files)
