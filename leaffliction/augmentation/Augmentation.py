import pathlib
import sys
from typing import Any

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np


def _crop(image: np.ndarray, percentage: float) -> np.ndarray:
    """
    Randomly resize and crop the image.

    Parameters
    ----------
    image : np.ndarray
        The image to randomly resize and crop.
    percentage : float
        The probability with which the image will be resized and cropped.

    Returns
    -------
    np.ndarray
        The randomly resized and cropped image.
    """
    height, width = image.shape[:2]

    resizecrop_transform = A.RandomResizedCrop(p=percentage, height=height, width=width)
    return resizecrop_transform(image=image)["image"]


def _rotate(image: np.ndarray, percentage: float) -> np.ndarray:
    """
    Rotate the image.

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
    rotate_transform = A.Rotate(p=percentage, border_mode=cv2.BORDER_CONSTANT, limit=90)
    return rotate_transform(image=image)["image"]


def _erasing(image: np.ndarray, percentage: float) -> np.ndarray:
    """
    Randomly erase parts of the image.

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
        max_holes=1,
        hole_width_range=(width // 10, width // 5),
        hole_height_range=(height // 10, height // 5),
    )
    return erasing_transform(image=image)["image"]


def _blur(image: np.ndarray, percentage: float) -> np.ndarray:
    """
    Blur the image.

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
    blur_transform = A.GaussianBlur(p=percentage, blur_limit=(3, 21))
    return blur_transform(image=image)["image"]


def _contrast(image: np.ndarray, percentage: float) -> np.ndarray:
    """
    Adjust the contrast of the image.

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
        p=percentage, contrast_limit=(-0.2, 0.3), brightness_limit=(-0.2, 0.3)
    )
    return contrast_transform(image=image)["image"]


def _flip(image: np.ndarray, percentage: float) -> np.ndarray:
    """
    Flip the image.

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


def visualize(image: np.ndarray) -> None:
    """
    Visualize the image.

    Parameters
    ----------
    image : np.ndarray
        The image to visualize.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(image)
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
        "Crop": _crop,
        "Rotate": _rotate,
        "Erasing": _erasing,
        "Blur": _blur,
        "Contrast": _contrast,
        "Flip": _flip,
    }
    return {
        augmentation_name: processed[augmentation_name](image, percentage)
        for augmentation_name in processed
    }


def visualize_all(original_image: np.ndarray, augmented_images: dict[str, np.ndarray]) -> None:
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


if __name__ == "__main__":
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
    visualize_all(image, augmented_images)

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
            print(f"Error saving {filename}_{func_name}{extension}.")
