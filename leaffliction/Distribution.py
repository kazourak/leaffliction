import argparse
import os
import sys
from linecache import cache

import matplotlib.pyplot as plt


def Distribution(dir_path: str) -> None:

    if not dir_path or not os.path.isdir(dir_path):
        raise Exception(f">>> Error {dir_path} is not a directory.")

    folders = [
        os.path.normpath(f"{dir_path}/{subdir}")
        for subdir in os.listdir(dir_path)
        if os.path.isdir(f"{dir_path}/{subdir}")
    ]
    if not folders:
        raise Exception(f">>> Error {dir_path} is empty.")

    folders_size = {}
    for folder in folders:
        folders_size[folder.split("/")[-1]] = len(
            [f for f in os.listdir(folder) if os.path.isfile(f"{folder}/{f}")]
        )

    plant_types = list(set([k.split('_')[0] for k in folders_size.keys()]))

    print(plant_types)

    for plant_type in plant_types:
        _folders_size = {k: v for k, v in folders_size.items() if k.startswith(plant_type)}

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.canvas.manager.set_window_title(f"{plant_type}")

        axes[0].pie(_folders_size.values(), labels=list(_folders_size.keys()), autopct="%1.0f%%")
        axes[1].bar(list(_folders_size.keys()), _folders_size.values(), align="center")
        plt.xticks(rotation=45)
        plt.show()



def options_parser():
    """Use to handle program parameters and options.
    """
    parser = argparse.ArgumentParser(
        prog='Distribution',
        description='This program should be used to see the distribution of plant types and states.',
        epilog='Please read the subject before proceeding to understand the input file format.')
    parser.add_argument('directory_path', type=str, nargs=1)
    parser.add_argument('--plant_type', type=str, default=None, help='Plot only the selected plant type.')
    return parser


if __name__ == "__main__":
    try:
        args = options_parser().parse_args()

        Distribution(dir_path=args.directory_path[0])
    except Exception as e:
        print(">>> Oups something went wrong.", file=sys.stderr)
        print(e, file=sys.stderr)