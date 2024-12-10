import pathlib
import sys

from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import onnxruntime as ort
import rembg
import tqdm
import cv2
import argparse
import numpy as np
import torch

def is_in_circle(x, y, center_x, center_y, radius):
    """
    Check if pixel (x, y) is within the circle defined by center_x, center_y, and radius.

    Parameters:
    x (int): X-coordinate of the pixel
    y (int): Y-coordinate of the pixel
    center_x (int): X-coordinate of the circle center
    center_y (int): Y-coordinate of the circle center
    radius (int): Radius of the circle

    Returns:
    bool: True if pixel is within the circle, False otherwise
    """
    return (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2


def draw_pseudo_landmarks(image, pseudo_landmarks, color, radius):
    """
    Draw circles on the image at the given pseudo-landmark coordinates.

    Parameters:
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


def create_pseudo_landmarks_image(image, kept_mask):
    """
    Create a displayable image with the pseudo-landmarks.

    Parameters:
    image (numpy.ndarray): The input image
    kept_mask (numpy.ndarray): The mask to use for finding pseudoland-marks

    Returns:
    numpy.ndarray: The modified image with pseudoland-marks drawn
    """
    pseudoland_marks = image.copy()

    top_x, bottom_x, center_v_x = pcv.homology.x_axis_pseudolandmarks(img=pseudoland_marks, mask=kept_mask,
                                                                      label='default')

    pseudoland_marks = draw_pseudo_landmarks(pseudoland_marks, top_x, (0, 0, 255), 3)
    pseudoland_marks = draw_pseudo_landmarks(pseudoland_marks, bottom_x, (255, 0, 255), 3)
    pseudoland_marks = draw_pseudo_landmarks(pseudoland_marks, center_v_x, (255, 0, 0), 3)

    return pseudoland_marks


def create_roi_image(image, masked, filled):
    """
    Create an image with the ROI rectangle and the mask
    """
    roi_start_x = 0
    roi_start_y = 0
    roi_w = image.shape[1]
    roi_h = image.shape[0]
    roi_line_w = 5

    roi = pcv.roi.rectangle(
        img=masked,
        x=roi_start_x,
        y=roi_start_y,
        w=roi_w,
        h=roi_h
    )

    kept_mask = pcv.roi.filter(mask=filled, roi=roi, roi_type='partial')

    roi_image = image.copy()
    roi_image[kept_mask != 0] = (0, 255, 0)

    roi_image = cv2.rectangle(
        roi_image,
        (roi_start_x, roi_start_y),
        (roi_start_x + roi_w, roi_start_y + roi_h),
        color=(255, 0, 0),
        thickness=roi_line_w
    )

    return roi_image, kept_mask


def plot_all_images(images: list, transformations_names: list):
    """
    Plot all images in the list

    Parameters:
    images (list): List of images to plot
    transformations_names (list): List of names for each image
    """

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Convert BGR to RGB
    for i, img in enumerate(images):
        images[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot images
    for i, ax in enumerate(axes.flat):
        axes[i // 3, i % 3].imshow(images[i])
        axes[i // 3, i % 3].axis('off')
        axes[i // 3, i % 3].set_title(f'Image: {transformations_names[i]}')

    plt.show()


def transform_image(img: np.ndarray):
    # Prepare the GPU session for rembg
    session_options = ort.SessionOptions()
    session_options.enable_mem_pattern = False
    session_options.enable_cpu_mem_arena = False

    # Detect available providers
    available_providers = ort.get_available_providers()
    print("Available providers:", available_providers)

    # Choose the best available provider
    preferred_providers = ['CoreMLExecutionProvider', 'CUDAExecutionProvider', 'TensorrtExecutionProvider']
    selected_provider = None

    for provider in preferred_providers:
        if provider in available_providers:
            selected_provider = provider
            break

    if selected_provider is None:
        selected_provider = 'CPUExecutionProvider'

    print(f"Using provider: {selected_provider}")

    # Remove background using the selected provider
    img_no_bg = rembg.remove(img, session=rembg.new_session("isnet-general-use"), providers=[selected_provider])

    # Rest of the function remains the same...
    # Step 2: Convert to grayscale with LAB space
    img_lab = pcv.rgb2gray_lab(img_no_bg, channel='l')

    # Step 3: Convert to binary ( Used to help differentiate plant and background )
    img_binary = pcv.threshold.binary(gray_img=img_lab, threshold=35, object_type='light')

    # Step 4: Apply median blur ( Used to reduce image noise )
    img_blur = pcv.median_blur(img_binary, ksize=5)

    # Step 5: Fill holes ( Used to reduce image noise )
    img_filled = pcv.fill(img_blur, size=200)

    # Mask image
    img_masked = pcv.apply_mask(img=img, mask=img_filled, mask_color='black')

    # Gaussian blur image
    gaussian_image = pcv.gaussian_blur(img_binary, ksize=(3, 3))

    # Create ROI image
    roi_image, roi_mask = create_roi_image(img, img_masked, img_filled)

    # Analyze the image
    analysis_image = pcv.analyze.size(img=img, labeled_mask=roi_mask)

    # Pseudo-landmarks image
    landmarks_image = create_pseudo_landmarks_image(img, roi_mask)

    # List of images
    transformations_list = [gaussian_image,
                            img_masked,
                            roi_image,
                            analysis_image,
                            landmarks_image]

    return transformations_list


def options_parser() -> argparse.ArgumentParser:
    """
    Use to handle program parameters and options.
    Returns
    -------
    The parser object.
    """

    parser = argparse.ArgumentParser(
        prog="Transformation",
        description="This program should be used to transform the image.",
        epilog="Please read the subject before proceeding to understand the input file format.",
    )
    # $> ./Transformation.[extension] ./Apple/apple_healthy/image (1).JPG
    parser.add_argument("image_path", type=str, nargs='?', help="Image file path")
    parser.add_argument("-src", "--source", type=str, nargs=1, help="Source directory path")
    parser.add_argument("-dst", "--destination", type=str, nargs=1, help="Destination directory path")
    return parser


def transform_all(source: str, destination: str):
    """
    Transform all images in the source directory and save them in the destination directory.

    Parameters:
    source (str): The source directory path
    destination (str): The destination directory path
    """

    # Get all files in the source directory
    files = [f for f in pathlib.Path(source).rglob("*") if f.is_file()]

    files_count = len(files)

    for file in tqdm.tqdm(files, total=files_count):
        # Open the image
        img = cv2.imread(str(file))

        # Transform the image
        transformations = transform_image(img)

        # Save the images
        for i, transformation in enumerate(transformations):
            cv2.imwrite(f"{destination}/{file.stem}_{i}{file.suffix}", transformation)


def transform_one(image_path: str):
    """
    Transform one image and display the results.

    Parameters:
    image_path (str): The image file path
    """

    # Open the image
    img = cv2.imread(image_path)

    # Transform the image
    transformations = transform_image(img)

    # Add original image
    transformations.insert(0, img)

    # Plot
    plot_all_images(transformations, ["Original", "Gaussian Blur", "Masked", "ROI", "Analysis", "Landmarks"])


if __name__ == "__main__":
    try:
        args = options_parser().parse_args()

        if args.source is not None and args.destination is not None:
            transform_all(args.source[0], args.destination[0])
        elif args.image_path is not None:
            transform_one(args.image_path)
        else:
            raise ValueError("Please, provide the image path or the source and destination directories.")

    except Exception as e:
        print(">>> Oups something went wrong.", file=sys.stderr)
        print(e, file=sys.stderr)