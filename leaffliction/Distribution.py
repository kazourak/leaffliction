import argparse
import os
import sys

import matplotlib.pyplot as plt


def distribution(dir_path: str, plant: str | None = None) -> None:
    """
    Plot for each type of plant, in the given directory, the distribution of diseases.
    Parameters
    ----------
    dir_path : Path to the directory to analyse.
    plant : Optional parameter used to plot graphs for only a specific plant.

    Returns
    -------

    """
    if not dir_path or not os.path.isdir(dir_path):
        raise Exception(f">>> Error {dir_path} is not a directory.")
    if plant is not None and len(plant) == 0:
        raise Exception(f">>> Error {plant} is empty.")

    subfolders = _get_subfolders(dir_path)
    if not subfolders:
        raise Exception(f">>> Error {dir_path} is empty.")

    subfolders_sizes = _get_subfolders_sizes(subfolders)
    plant_types = _get_plant_types(subfolders, plant)
    _plot_plant_types(subfolders_sizes, plant_types)


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


def _get_plant_types(subfolders: list, plant: str | None = None) -> list:
    """
    Get the list of each unique type of plant
    Parameters
    ----------
    subfolders : list of all subfolders.
    plant : Optional parameter used to plot graphs for only a specific plant.

    Returns
    -------
    A list of all plant types.
    """
    return (
        list(set([k.split("/")[-1].split("_")[0] for k in subfolders]))
        if plant is None
        else [plant]
    )


def _plot_plant_types(subfolders_sizes: dict, plant_types: list) -> None:
    """
    Plot for each type of plant its distribution.
    Parameters
    ----------
    subfolders_sizes : A dictionary of subfolder sizes.
    plant_types : A list of plant types.

    Returns
    -------

    """
    for plant_type in plant_types:
        _folders_size = {k: v for k, v in subfolders_sizes.items() if k.startswith(plant_type)}
        if len(_folders_size) == 0:
            raise Exception(f">>> Error {plant_type} is empty.")

        color_map = plt.cm.tab10
        colors = [color_map(i / (len(_folders_size) - 1)) for i in range(len(_folders_size))]

        _fig, _axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        _fig.canvas.manager.set_window_title(f"{plant_type}")

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
        description="This program should be used to see the distribution of plant types and states.",
        epilog="Please read the subject before proceeding to understand the input file format.",
    )
    parser.add_argument("directory_path", type=str, nargs=1)
    parser.add_argument(
        "--plant_type", type=str, default=None, help="Plot only the selected plant type."
    )
    return parser


if __name__ == "__main__":
    try:
        args = options_parser().parse_args()

        distribution(dir_path=args.directory_path[0], plant=args.plant_type)
    except Exception as e:
        print(">>> Oups something went wrong.", file=sys.stderr)
        print(e, file=sys.stderr)
