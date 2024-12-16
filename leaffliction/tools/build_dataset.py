import tensorflow as tf


def build_dataset(
    data_path: str,
    batch_size: int,
    validation_ratio: float,
    seed: int,
    size: tuple[float, float] = (256, 256),
) -> tuple:
    """
    Build from the given directory the training and evaluate dataset.

    Parameters
    ----------
    data_path : Path of the dataset> Must contain subdirectories with images. Each subdirectory corresponds to a class.
    batch_size : Size of the batches.
    validation_ratio : Ratio of data used for the validation dataset.
    seed : Magic number used to reproduce a build.
    size : Image size (height, width).

    Returns
    -------
    tuple : The training and evaluate dataset.
    """
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=validation_ratio,
        subset="training",
        seed=seed,
        image_size=size,
        batch_size=batch_size,
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=validation_ratio,
        subset="validation",
        seed=seed,
        image_size=size,
        batch_size=batch_size,
    )

    return train_dataset, test_dataset
