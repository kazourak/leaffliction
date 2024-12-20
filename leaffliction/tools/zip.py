import os
import zipfile


def _zip(zip_dest: zipfile.ZipFile, source: str):
    """
    Zip a single source (dir or file) into the destination.
    Parameters
    ----------
    zip_dest : Archive destination.
    source : path of the source.

    Returns
    -------

    """
    for dir, _, files in os.walk(source):
        for file in files:
            file_path = os.path.join(dir, file)
            archive_name = os.path.relpath(file_path, source)
            zip_dest.write(file_path, os.path.join(os.path.basename(source), archive_name))


def zip_directories(dest_file: str, source_dirs: list[str]):
    """
    Zip the given list of source into the destination file.
    Parameters
    ----------
    dest_file : Destination file.
    source_dirs :

    Returns
    -------

    """
    with zipfile.ZipFile(dest_file, "w", zipfile.ZIP_DEFLATED) as zip_dest:
        for source in source_dirs:
            _zip(zip_dest, source)
