import cv2

from leaffliction.Augmentation import augment


def test_augmentation() -> None:
    """
    Test the augmentation function.

    Returns
    -------

    """
    image = cv2.imread('tests/augmentation/test_image.png')

    if image is None:
        assert False

    augmented_images = augment(image, 0.5)

    assert augmented_images is not None

    # check if number of augmented images is correct
    assert len(augmented_images) == 6

    # check if all images are not None
    for augmented_image in augmented_images.values():
        assert augmented_image is not None
