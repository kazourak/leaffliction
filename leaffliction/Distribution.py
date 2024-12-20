"""
Module: Distribution

This module provides functions to analyze and retrieve the distribution of images within a dataset
directory. It includes functionalities to get the list of subfolders and their respective image
counts, facilitating the balancing of datasets for machine learning tasks.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt


def distribution(dir_path: str) -> dict:
    """
    Get the distribution of images in each subdirectory.
    Parameters
    ----------
    dir_path : Path to the directory to analyse.

    Returns
    -------
    dict : each subdirectory name and its total number of images.
    """
    if not dir_path or not os.path.isdir(dir_path):
        raise Exception(f">>> Error {dir_path} is not a directory.")

    subfolders = _get_subfolders(dir_path)

    if not subfolders:
        raise Exception(f">>> Error {dir_path} is empty.")

    subfolders_sizes = _get_subfolders_sizes(subfolders)

    return subfolders_sizes


def _get_subfolders(dir_path: str) -> list:
    """
    Get the list of subfolders in the given directory.
    Parameters
    ----------
    dir_path : Path to the directory to analyse. In this case to get all subdirectories.

    Returns
    -------
    A list of all subfolders.
    """
    return [
        os.path.normpath(f"{dir_path}/{subdir}")
        for subdir in os.listdir(dir_path)
        if os.path.isdir(f"{dir_path}/{subdir}")
    ]


def _get_subfolders_sizes(subfolders: list) -> dict:
    """
    Get the number of files in each subfolder.
    Parameters
    ----------
    subfolders : A list of all subfolders.

    Returns
    -------
    Each subfolder with its size as a dictionary.
    """
    subfolders_sizes = {}
    for folder in subfolders:
        subfolders_sizes[folder.split("/")[-1]] = len(
            [f for f in os.listdir(folder) if os.path.isfile(f"{folder}/{f}")]
        )
    return subfolders_sizes


def _plot_plant_types(subfolders_sizes: dict) -> None:
    """
    Plot the distribution of plant diseases.
    Parameters
    ----------
    subfolders_sizes : A dictionary of subfolder and sizes.

    Returns
    -------

    """
    _folders_size = subfolders_sizes
    if len(_folders_size) == 0:
        raise Exception(">>> Error no subfolders.")

    color_map = plt.cm.tab10
    colors = [color_map(i / (len(_folders_size) - 1)) for i in range(len(_folders_size))]

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    _fig.canvas.manager.set_window_title("Distribution")

    _axes[0].pie(
        _folders_size.values(),
        labels=list(_folders_size.keys()),
        autopct="%1.0f%%",
        colors=colors,
    )
    _axes[1].bar(
        list(_folders_size.keys()),
        _folders_size.values(),
        align="center",
        color=colors,
    )
    plt.xticks(rotation=45)
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
        description="This program should be used to see the distribution of plant types and"
        + " states.",
        epilog="Please read the subject before proceeding to understand the input file format.",
    )
    parser.add_argument("directory_path", type=str, nargs=1)
    return parser


if __name__ == "__main__":
    try:
        args = options_parser().parse_args()

        distribution_dict = distribution(dir_path=args.directory_path[0])
        _plot_plant_types(distribution_dict)
    except Exception as e:
        print(">>> Oups something went wrong.", file=sys.stderr)
        print(e, file=sys.stderr)
