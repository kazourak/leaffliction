"""
Module: Transformation

This module provides functions for transforming images, including applying filters,
masking, selecting regions of interest (ROIs), performing analyses, detecting landmarks,
and generating histograms. It utilizes libraries such as OpenCV, plantcv, NumPy, and tqdm
to perform various image processing tasks essential for data augmentation in machine learning
workflows.
"""

import argparse
import itertools
import multiprocessing
import os
import pathlib
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv
import tqdm

# List of transformations names. Used for plotting and saving images
TRANSFORMATIONS_NAMES = [
    "Original",
    "Gaussian Blur",
    "Masked",
    "ROI",
    "Analysis",
    "Landmarks",
    "Histogram",
]


def is_in_circle(x: int, y: int, center_x: int, center_y: int, radius: int) -> bool:
    """
    Check if a pixel lies within a circle.

    Parameters
    ----------
    x : int
        X-coordinate of the pixel.
    y : int
        Y-coordinate of the pixel.
    center_x : int
        X-coordinate of the circle center.
    center_y : int
        Y-coordinate of the circle center.
    radius : int
        Radius of the circle.

    Returns
    -------
    bool
        True if the pixel is within the circle, False otherwise.
    """
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2


def draw_pseudo_landmarks(
    image: np.ndarray, pseudo_landmarks: list[list[int]], color: tuple, radius: int
) -> np.ndarray:
    """
    Draw circles on an image at specified pseudo-landmark coordinates.

    Parameters
    ----------
    image : numpy.ndarray
        The image array to draw on.
    pseudo_landmarks : list[list[int]]
        List of pseudo-landmark coordinates as [[y1, x1], [y2, x2], ...].
    color : tuple
        RGB color value for the circles.
    radius : int
        Radius of the circles to draw.

    Returns
    -------
    numpy.ndarray
        The modified image array.
    """
    for landmark in pseudo_landmarks:
        if len(landmark) >= 1 and len(landmark[0]) >= 2:
            center_x, center_y = landmark[0]
            for x, y in itertools.product(range(image.shape[0]), range(image.shape[1])):
                if is_in_circle(x, y, center_x, center_y, radius):
                    image[y, x] = color
    return image


def create_pseudo_landmarks_image(image: np.ndarray, kept_mask: np.ndarray) -> np.ndarray:
    """
    Create a displayable image with pseudo-landmarks.

    Parameters
    ----------
    image : numpy.ndarray
        The input image.
    kept_mask : numpy.ndarray
        The mask to use for identifying pseudo-landmarks.

    Returns
    -------
    numpy.ndarray
        The modified image with pseudo-landmarks drawn.
    """
    pseudo_landmarks = image.copy()

    top_x, bottom_x, center_v_x = pcv.homology.x_axis_pseudolandmarks(
        img=pseudo_landmarks, mask=kept_mask, label="default"
    )

    pseudo_landmarks = draw_pseudo_landmarks(pseudo_landmarks, top_x, (0, 0, 255), 3)
    pseudo_landmarks = draw_pseudo_landmarks(pseudo_landmarks, bottom_x, (255, 0, 255), 3)
    pseudo_landmarks = draw_pseudo_landmarks(pseudo_landmarks, center_v_x, (255, 0, 0), 3)

    return pseudo_landmarks


def create_roi_image(image: np.ndarray, masked: np.ndarray, filled: np.ndarray) -> tuple:
    """
    Create an image with a region of interest (ROI) highlighted.

    Parameters
    ----------
    image : numpy.ndarray
        The input image to analyze.
    masked : numpy.ndarray
        The masked image.
    filled : numpy.ndarray
        The filled image.

    Returns
    -------
    tuple
        A tuple containing the ROI image and the kept mask.
    """
    roi_start_x = 0
    roi_start_y = 0
    roi_w = image.shape[1]
    roi_h = image.shape[0]
    roi_line_w = 5

    roi = pcv.roi.rectangle(img=masked, x=roi_start_x, y=roi_start_y, w=roi_w, h=roi_h)

    kept_mask = pcv.roi.filter(mask=filled, roi=roi, roi_type="partial")

    roi_image = image.copy()
    roi_image[kept_mask != 0] = (0, 255, 0)

    roi_image = cv2.rectangle(
        roi_image,
        (roi_start_x, roi_start_y),
        (roi_start_x + roi_w, roi_start_y + roi_h),
        color=(255, 0, 0),
        thickness=roi_line_w,
    )

    return roi_image, kept_mask


def plot_stat_hist(label: str, scale: float = 1.0):
    """
    Plot the histogram of image colors for a specific label.

    Parameters
    ----------
    label : str
        The label to plot.
    scale : float, optional
        The scale factor for the histogram, by default 1.0.

    Returns
    -------
    None
    """
    observation_label = label + "_frequencies"
    observation = pcv.outputs.observations["default_1"][observation_label]
    y = observation["value"]
    x = [i * scale for i in observation["label"]]

    if label == "hue":
        x = x[:128]
        y = y[:128]
    elif label in {"blue-yellow", "green-magenta"}:
        x = [val + 128 for val in x]

    plt.plot(x, y, label=label)


def create_histogram(image: np.ndarray, kept_mask: np.ndarray) -> plt.Figure:
    """
    Create a histogram of image colors.

    Parameters
    ----------
    image : numpy.ndarray
        The image to analyze.
    kept_mask : numpy.ndarray
        The mask used for the analysis.

    Returns
    -------
    plt.Figure
        The histogram plot as a Matplotlib figure.
    """
    scale_factors = {
        "blue": 1,
        "green": 1,
        "green-magenta": 1,
        "lightness": 2.55,
        "red": 1,
        "blue-yellow": 1,
        "hue": 1,
        "saturation": 2.55,
        "value": 2.55,
    }

    # Generate labels and analyze color spaces
    labels, _ = pcv.create_labels(mask=kept_mask)
    pcv.analyze.color(rgb_img=image, colorspaces="all", labeled_mask=labels, label="default")

    # Create and configure the plot
    fig, ax = plt.subplots(figsize=(16, 9))
    for label, scale in scale_factors.items():
        plot_stat_hist(label, scale)

    ax.legend()
    ax.set_title("Color Histogram")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.grid(visible=True, which="major", axis="both", linestyle="--")

    return fig


def plot_all_images(images: list):
    """
    Plot all images in a grid.

    Parameters
    ----------
    images : list
        List of images to plot.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Convert BGR to RGB
    for i in range(6):
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)

    # Plot images
    for i in range(6):
        axes[i // 3, i % 3].imshow(images[i])
        axes[i // 3, i % 3].axis("off")
        axes[i // 3, i % 3].set_title(f"Image: {TRANSFORMATIONS_NAMES[i]}")

    # Show plot
    plt.show()


def transform_image(img: np.ndarray) -> list:
    """
    Transform an image using PlantCV techniques.

    Parameters
    ----------
    img : numpy.ndarray
        The image to transform.

    Returns
    -------
    list
        List of transformed images.
    """
    # Step 1: Convert to grayscale with LAB space with the blue channel
    img_lab = pcv.rgb2gray_lab(img, channel="b")

    # Step 2: Convert to binary ( Used to help differentiate plant and background )
    img_binary = pcv.threshold.otsu(gray_img=img_lab, object_type="light")

    # Step 3: Fill holes ( Used to reduce image noise )
    img_filled = pcv.fill_holes(img_binary)

    # Step 4: Apply median blur ( Used to reduce image noise )
    img_blur = pcv.median_blur(img_filled, ksize=5)

    # Mask image
    img_masked = pcv.apply_mask(img=img, mask=img_blur, mask_color="white")

    # Gaussian blur image
    gaussian_image = pcv.gaussian_blur(img_filled, ksize=(3, 3))

    # Create ROI image
    roi_image, roi_mask = create_roi_image(img, img_masked, img_filled)

    # Analyze the image
    analysis_image = pcv.analyze.size(img=img, labeled_mask=roi_mask)

    # Pseudo-landmarks image
    landmarks_image = create_pseudo_landmarks_image(img, roi_mask)

    # Histogram image
    histogram_image = create_histogram(img, roi_mask)

    transformations_list = [
        gaussian_image,
        img_masked,
        roi_image,
        analysis_image,
        landmarks_image,
        histogram_image,
    ]

    return transformations_list


def process_image(args: tuple):
    """
    Process a single image with transformations and save the results.

    Parameters
    ----------
    args : tuple
        Tuple containing the file path and destination directory.

    Returns
    -------
    None
    """
    file, destination = args
    img = cv2.imread(str(file))

    if img is None:
        print(f"Skipping {file} as it is not an image file.")
        return

    # Transform the image
    transformations = transform_image(img)

    # Save the images
    for i in range(len(transformations) - 1):
        cv2.imwrite(
            f"{destination}/{file.stem}_{TRANSFORMATIONS_NAMES[i + 1]}{file.suffix}",
            transformations[i],
        )

    # Save the histogram
    transformations[-1].savefig(
        f"{destination}/{file.stem}_{TRANSFORMATIONS_NAMES[-1]}{file.suffix}"
    )

    # Close the histogram plot
    plt.close(transformations[-1])


def transform_all(source: str, destination: str):
    """
    Transform all images in the source directory and save them in the destination directory
    using multiprocessing.

    Parameters
    ----------
    source : str
        Source directory path.
    destination : str
        Destination directory path.

    Returns
    -------
    None
    """
    # Get all files in the source directory
    files = [f for f in pathlib.Path(source).rglob("*") if f.is_file()]
    files_count = len(files)

    if files_count == 0:
        print("No files found in the source directory.")
        return

    # Prepare arguments for parallel processing
    tasks = [(file, destination) for file in files]

    # Use multiprocessing to process images in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_image, tasks), total=files_count))

    print("All transformations completed.")


def transform_one(image_path: str):
    """
    Transform a single image and plot the results.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    None
    """
    # Open the image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("The provided image path is invalid.")

    # Transform the image
    transformations = transform_image(img)

    # Add original image
    transformations.insert(0, img)

    # Plot
    plot_all_images(transformations)


def get_mask(image_path: str) -> np.ndarray:
    """
    Return the masked image.
    Parameters
    ----------
    image_path : Leaf to processed.

    Returns
    -------
    The processed image as a np.ndarray.
    """
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("The provided image path is invalid.")

    img_lab = pcv.rgb2gray_lab(img, channel="b")
    img_binary = pcv.threshold.otsu(gray_img=img_lab, object_type="light")
    img_filled = pcv.fill_holes(img_binary)
    img_blur = pcv.median_blur(img_filled, ksize=5)

    return pcv.apply_mask(img=img, mask=img_blur, mask_color="white")


def options_parser() -> argparse.ArgumentParser:
    """
    Create command-line options for the script.

    Returns
    -------
    argparse.ArgumentParser
        Parser object containing the command-line options.
    """
    parser = argparse.ArgumentParser(
        prog="Transformation",
        description="This program should be used to transform the image.",
        epilog="Please read the subject before proceeding to understand the input file format.",
    )
    parser.add_argument(
        "source_path",
        type=str,
        nargs=1,
        help="Image or directory path to  transform. If you give a directory please define the "
        + "destination option.",
    )
    parser.add_argument(
        "-dst",
        "--destination",
        type=str,
        nargs=1,
        help="Destination directory path (must be used only with a directory source).",
    )
    return parser


if __name__ == "__main__":
    try:
        args = options_parser().parse_args()

        if os.path.isfile(args.source_path[0]) and args.destination is None:
            transform_one(args.source_path[0])
        elif os.path.isdir(args.source_path[0]) and args.destination is not None:
            transform_all(args.source_path[0], args.destination)
        else:
            raise ValueError(
                "Bad usage: \n"
                + "1 - Give a valid image path without the destination argument\n"
                + "2 - Give a valid directory path with the destination option."
            )

    except Exception as e:
        print(">>> Oups something went wrong.", file=sys.stderr)
        print(e, file=sys.stderr)
