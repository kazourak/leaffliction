from plantcv import plantcv as pcv
import rembg
import cv2

# img, path, filename = pcv.readimage(filename="../data/external/images/leaf.png")
#
# img_no_background = rembg.remove(img)
#
# tmp2 = pcv.rgb2gray_lab(rgb_img=img_no_background, channel='l')
#
# l_tresh = pcv.threshold.binary(gray_img=tmp2, threshold=35, object_type='light')
#
# gaussian_img = pcv.gaussian_blur(l_tresh, ksize=(5, 5), sigma_x=0)
#
# cv2.imshow("Image", l_tresh)
# cv2.waitKey(0)


def transform_image(file: str):

    # Open the image
    img, path, img_filename = pcv.readimage(str, mode='rgb')




