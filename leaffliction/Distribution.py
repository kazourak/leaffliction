import glob
import os
import sys

import matplotlib.pyplot as plt


def Distribution(dir: str) -> None:

    if not dir or not os.path.isdir(dir):
        raise Exception(f">>> Error {dir} is not a directory.")

    folders = [f"{dir}/{subdir}" for subdir in os.listdir(dir) if os.path.isdir(f"{dir}/{subdir}")]
    if not folders:
        raise Exception(f">>> Error {dir} is empty.")

    print(folders)


if __name__ == "__main__":
    dir = sys.argv[1]
    Distribution(dir)
