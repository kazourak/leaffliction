import os


def extract_files(path: str) -> list[str]:
    """
    Get all files from the given path (supposed as a directory).
    Each file is extract from a subdirectory. (path/subdir/file)
    Parameters
    ----------
    path : Path of the directory.

    Returns
    -------
    A list of paths.
    """
    _subdir = get_dirs(path)

    files = []

    for dir in _subdir:
        files.extend(get_files(dir))

    return files


def get_dirs(path: str) -> list[str]:
    """
    Get from the given path (supposed as a directory) all subdirectories. (path/subdir)
    Parameters
    ----------
    path : Path of the directory.

    Returns
    -------
    A list of paths.
    """
    return [
        os.path.normpath(f"{path}/{subdir}")
        for subdir in os.listdir(path)
        if os.path.isdir(f"{path}/{subdir}")
    ]


def get_files(path: str) -> list[str]:
    """
    Get from the given path (supposed as a directory) all files. (path/file)
    Parameters
    ----------
    path : Path of the directory.

    Returns
    -------
    A list of paths.
    """
    return [
        os.path.normpath(f"{path}/{subdir}")
        for subdir in os.listdir(path)
        if os.path.isfile(f"{path}/{subdir}")
    ]
