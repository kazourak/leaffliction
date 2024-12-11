import argparse
import multiprocessing
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


def is_in_circle(x: int, y: int, center_x: int, center_y: int, radius: int):
    """
    Check if pixel (x, y) is within the circle defined by center_x, center_y, and radius.

    Args:
        x (int): X-coordinate of the pixel
        y (int): Y-coordinate of the pixel
        center_x (int): X-coordinate of the circle center
        center_y (int): Y-coordinate of the circle center
        radius (int): Radius of the circle

    Returns:
        bool: True if pixel is within the circle, False otherwise

    """
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2


def draw_pseudo_landmarks(image: np.ndarray, pseudo_landmarks: list[list[int]], color: tuple, radius: int):
    """
    Draw circles on the image at the given pseudo-landmark coordinates.

    Args:
        image (numpy.ndarray): The image array to draw on
        pseudo_landmarks (list[list[int]]): List of pseudo-landmark coordinates in the form [[y1, x1], [y2, x2], ...]
        color (tuple): RGB color value to use for the circles
        radius (int): Radius of the circles to draw

    Returns:
        numpy.ndarray: The modified image array

    """

    for landmark in pseudo_landmarks:
        if len(landmark) >= 1 and len(landmark[0]) >= 2:
            center_x, center_y = landmark[0]
            for x in range(image.shape[0]):
                for y in range(image.shape[1]):
                    if is_in_circle(x, y, center_x, center_y, radius):
                        image[y, x] = color
    return image


def create_pseudo_landmarks_image(image: np.ndarray, kept_mask: np.ndarray) -> np.ndarray:
    """
    Create a displayable image with the pseudo-landmarks.

    Args:
        image (numpy.ndarray): The input image
        kept_mask (numpy.ndarray): The mask to use for finding pseudo-landmarks

    Returns:
        numpy.ndarray: The modified image with pseudo-landmarks drawn

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
    Create an image with the region of interest (ROI) highlighted.

    Args:
        image (numpy.ndarray): Image to analyze.
        masked (numpy.ndarray): Masked image.
        filled (numpy.ndarray): Filled image.

    Returns:
        tuple: The ROI image and the kept mask.

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
    Plot the histogram of the image colors.

    Args:
        label (str): The label to plot.
        scale (float): The scale factor.

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
    Create a histogram of the image colors.

    Args:
        image (np.ndarray): The image to analyze.
        kept_mask (np.ndarray): The mask to use for the analysis.

    Returns:
        plt.Figure: The histogram plot.

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

    Args:
        images (list): The list of images to plot.

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

    # Get histogram
    histogram = images[-1]

    # Show plot
    plt.show()
    histogram.show()


def transform_image(img: np.ndarray) -> list:
    """
    Transform the image using PlantCV.

    Args:
        img (numpy.ndarray): The image to transform.

    Returns:
        list (list): The list of transformed images.

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
    Helper function to process a single image with transformations and save the results.

    Args:
        args (tuple): Contains the file path, source, destination, and transformation names.

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
            f"{destination}/{file.stem}_{TRANSFORMATIONS_NAMES[i+1]}{file.suffix}",
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
    Transform all images in the source directory and save them in the destination directory using multiprocessing.

    Parameters:
    source (str): The source directory path
    destination (str): The destination directory path
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

    Args:
        image_path:

    Returns:
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


def options_parser() -> argparse.ArgumentParser:
    """
    Create the command line options.

    Returns:
        argparse.ArgumentParser: The command line options.
    """

    parser = argparse.ArgumentParser(
        prog="Transformation",
        description="This program should be used to transform the image.",
        epilog="Please read the subject before proceeding to understand the input file format.",
    )
    parser.add_argument("image_path", type=str, nargs="?", help="Image file path")
    parser.add_argument("-src", "--source", type=str, nargs=1, help="Source directory path")
    parser.add_argument(
        "-dst", "--destination", type=str, nargs=1, help="Destination directory path"
    )
    return parser


if __name__ == "__main__":
    try:
        args = options_parser().parse_args()

        if args.source is not None and args.destination is not None:
            transform_all(args.source[0], args.destination[0])
        elif args.image_path is not None:
            transform_one(args.image_path)
        else:
            raise ValueError(
                "Please, provide the image path or the source and destination directories."
            )

    except Exception as e:
        print(">>> Oups something went wrong.", file=sys.stderr)
        print(e, file=sys.stderr)
