import cv2

import util

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def save_rgb(path,image):
    util.ensure_folders(path)
    return cv2.imwrite(path,transform_rgb_bgr(image))
