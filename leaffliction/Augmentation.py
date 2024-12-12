import pathlib
import sys
from typing import Any

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np


def _rotate(image: np.ndarray, percentage: float) -> np.ndarray:
    """
    Rotate the image.
    Why?: To simulate different angles of view

    Parameters
    ----------
    image : np.ndarray
        The image to rotate.
    percentage : float
        The probability with which the image will be rotated.

    Returns
    -------
    np.ndarray
        The rotated image.
    """
    rotate_transform = A.Rotate(p=percentage, limit=(-45, 45))
    return rotate_transform(image=image)["image"]


def _clahe(image: np.ndarray, percentage: float) -> np.ndarray:
    """
    Apply CLAHE to the image.
    Why?: For highlighting details in images with low light or saturated areas

    Parameters
    ----------
    image : np.ndarray
        The image to apply CLAHE to.
    percentage : float
        The probability with which CLAHE will be applied to the image.

    Returns
    -------
    np.ndarray
        The image with CLAHE applied.
    """
    clahe_transform = A.CLAHE(p=percentage)
    return clahe_transform(image=image)["image"]


def _zoom_blur(image: np.ndarray, percentage: float) -> np.ndarray:
    """
    Apply zoom blur to the image.
    Why?: To simulate motion blur in images

    Parameters
    ----------
    image : np.ndarray
        The image to apply zoom blur to.
    percentage : float
        The probability with which zoom blur will be applied to the image.

    Returns
    -------
    np.ndarray
        The image with zoom blur applied.
    """
    zoom_blur_transform = A.ZoomBlur(p=percentage, max_factor=(1, 1.1))
    return zoom_blur_transform(image=image)["image"]


def _erasing(image: np.ndarray, percentage: float) -> np.ndarray:
    """
    Randomly erase parts of the image.
    Why?: To simulate occlusions in images

    Parameters
    ----------
    image : np.ndarray
        The image to erase parts of.
    percentage : float
        The probability with which parts of the image will be erased.

    Returns
    -------
    np.ndarray
        The image with parts randomly erased.
    """
    height, width = image.shape[:2]

    erasing_transform = A.CoarseDropout(
        p=percentage,
        num_holes_range=(1, 2),
        hole_width_range=(width // 8, width // 6),
        hole_height_range=(height // 8, height // 6),
    )
    return erasing_transform(image=image)["image"]


def _blur(image: np.ndarray, percentage: float) -> np.ndarray:
    """
    Blur the image.
    Why?: To simulate out-of-focus images

    Parameters
    ----------
    image : np.ndarray
        The image to blur.
    percentage : float
        The probability with which the image will be blurred.

    Returns
    -------
    np.ndarray
        The blurred image.
    """
    blur_transform = A.GaussianBlur(p=percentage, blur_limit=(3, 9))
    return blur_transform(image=image)["image"]


def _contrast(image: np.ndarray, percentage: float) -> np.ndarray:
    """
    Adjust the contrast of the image.
    Why?: To simulate different lighting conditions

    Parameters
    ----------
    image : np.ndarray
        The image to adjust the contrast of.
    percentage : float
        The probability with which the image's contrast will be adjusted.

    Returns
    -------
    np.ndarray
        The image with adjusted contrast.
    """
    contrast_transform = A.RandomBrightnessContrast(
        p=percentage, contrast_limit=(-0.3, 0.3), brightness_limit=(-0.3, 0.3)
    )
    return contrast_transform(image=image)["image"]


def _flip(image: np.ndarray, percentage: float) -> np.ndarray:
    """
    Flip the image.
    Why?: To simulate different angles of view

    Parameters
    ----------
    image : np.ndarray
        The image to flip.
    percentage : float
        The probability with which the image will be flipped.

    Returns
    -------
    np.ndarray
        The flipped image.
    """
    flip_transform = A.HorizontalFlip(p=percentage)
    return flip_transform(image=image)["image"]


def _visualize_all(original_image: np.ndarray, augmented_images: dict[str, np.ndarray]) -> None:
    """
    Visualize the original and augmented images.

    Parameters
    ----------
    original_image : np.ndarray
        The original image.
    augmented_images : dict[str, np.ndarray]
        The augmented images.

    Returns
    -------
    None
    """
    plt.figure(figsize=(20, 10))
    plt.axis("off")

    # Plot original image
    plt.subplot(2, 4, 1)
    plt.imshow(original_image)
    plt.title("Original")
    plt.axis("off")

    # Plot augmented images
    for i, (augmentation_name, augmented_image) in enumerate(augmented_images.items(), start=2):
        plt.subplot(2, 4, i)
        plt.imshow(augmented_image)
        plt.title(augmentation_name)
        plt.axis("off")

    plt.show()


def augment(image: np.ndarray, percentage: float) -> dict[str, Any]:
    """
    Augment the image.

    Parameters
    ----------
    image : np.ndarray
        The image to augment.
    percentage : float
        The probability with which the image will be augmented.

    Returns
    -------
    list[np.ndarray]
        The augmented images.
    """
    processed = {
        "CLAHE": _clahe,
        "Zoom Blur": _zoom_blur,
        "Rotate": _rotate,
        "Blur": _blur,
        "Erased": _erasing,
        "Flip": _flip,
    }
    return {
        augmentation_name: processed[augmentation_name](image, percentage)
        for augmentation_name in processed
    }


if __name__ == "__main__":
    try:
        # Get program argument
        args = sys.argv[1:]
        if len(args) != 1:
            raise ValueError("Please, only provide the path to the image to augment.")

        # Open file with OpenCV
        image = cv2.imread(args[0])

        if image is None:
            raise ValueError("The provided image path is invalid.")

        # Get file information
        filename = pathlib.Path(args[0]).stem
        extension = pathlib.Path(args[0]).suffix
        path = pathlib.Path(args[0]).parent

        # Switch color profile BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented_images = augment(image, 1)

        # Visualize all images
        _visualize_all(image, augmented_images)

        # Save augmented images
        for func_name, augmented_image in augmented_images.items():
            if (
                cv2.imwrite(
                    f"{path}/{filename}_{func_name}{extension}",
                    cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR),
                )
                is True
            ):
                print(f"{filename}_{func_name}{extension} saved successfully.")
            else:
                raise RuntimeError(f"Error saving {filename}_{func_name}{extension}.")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
