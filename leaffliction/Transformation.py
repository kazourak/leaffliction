from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import rembg
import cv2


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


def transform_image(file: str, show: bool = False):
    # Open the image
    img, path, img_filename = pcv.readimage(file, mode='rgb')

    # Prepare the image
    # Step 1: Remove the background
    img_no_bg = rembg.remove(img)

    # Step 2: Convert to grayscale with LAB space
    img_lab = pcv.rgb2gray_lab(img_no_bg, channel='l')

    # Step 3: Convert to binary ( Used to help differentiate plant and background )
    img_binary = pcv.threshold.binary(gray_img=img_lab, threshold=35, object_type='light')

    # TODO: check best way
    # Step 5: Apply gaussian blur ( Used to reduce image noise )
    img_blur = pcv.median_blur(img_binary, ksize=5)

    # Step 4: Fill holes ( Used to reduce image noise )
    img_filled = pcv.fill(img_blur, size=200)

    # # Step 4: Fill holes ( Used to reduce image noise )
    # img_filled = pcv.fill(img_binary, size=200)
    #
    # # Step 5: Apply gaussian blur ( Used to reduce image noise )
    # img_blur = pcv.gaussian_blur(img_binary, ksize=(3, 3))
    # # Masked image
    # img_masked = pcv.apply_mask(img=img, mask=img_blur, mask_color='black')

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
    transformations_list = [img,
                            gaussian_image,
                            img_masked,
                            roi_image,
                            analysis_image,
                            landmarks_image]

    # Transformations names
    transformations_names = ["Original",
                             "Gaussian Blur",
                             "Mask",
                             "ROI",
                             "Analysis",
                             "Landmarks"]
    # Plot all images
    if show:
        plot_all_images(transformations_list, transformations_names)

    # cv2.waitKey(0)

    # Get ROI Mask


transform_image("../data/external/images/Apple_Black_rot/image (3).JPG", show=True)
