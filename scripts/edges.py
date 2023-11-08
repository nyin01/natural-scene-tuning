# https://github.com/zhang-tianran/Sketch-to-Image-Generation/blob/main/preprocess.py
# cs1470 deep learning spring 22

import numpy as np
import os
import glob
import cv2


def get_images_paths(directory_name, image_type='png'):
    """
    get the file name/path of all the files within a folder.
        e.g. glob.glob("/home/adam/*/*.txt").
    Use glob.escape to escape strings that are not meant to be patterns
        glob.glob(glob.escape(directory_name) + "/*.txt")
    :param directory_name: (str) the root directory name that contains all the images we want
    :param image: (str) either "jpg" or "png"
    :return: a list of queried files and directories
    """
    # concatnate strings
    end = "/*." + image_type

    return glob.glob(glob.escape(directory_name) + end)


def image_to_sketch(img, kernel_size=21):
    """
    Inputs:
    - img: RGB image, ndarray of shape []
    - kernel_size: 7 by default, used in DoG processing
    - greyscale: False by default, convert to greyscale image if True, RGB otherwise
    Returns:
    - RGB or greyscale sketch, ndarray of shape [] or []
    """

    # img = adjust_contrast(img)

    # convert to greyscale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # invert
    inv = cv2.bitwise_not(grey)
    # blur
    blur = cv2.GaussianBlur(
        inv, (kernel_size, kernel_size), sigmaX=0, sigmaY=0)
    # invert
    inv_blur = cv2.bitwise_not(blur)
    # convert to sketch
    sketch = cv2.divide(grey, inv_blur, scale=256.0)

    # sketch = adjust_contrast(sketch)

    out = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    return out


def adjust_contrast(img):

    # weight = np.ones(img.shape) * 1.2
    # out = np.uint8(cv2.multiply(np.float32(img), weight))

    out = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

    return out


def pad_resize(img, img_size):
    # pad or resize img to square of side length (img_size)
    # h, w, c = img.shape
    # # white padding
    # color = (255,255,255)

    down_points = (img_size, img_size)
    out = cv2.resize(img, down_points, interpolation=cv2.INTER_LINEAR)

    return out


def store_inputs(from_dir, to_dir, img_size, i):
    # store processed images (after concat)
    # size: desired original img size, output will be twice as wide (concat)
    files = get_images_paths(from_dir)
    for f in files:
        ext = str(i)+".png"
        img = cv2.imread(f)
        # img = pad_resize(img, img_size)
        sketch = image_to_sketch(img)

        # img = pad_resize(img, img_size)
        # sketch = pad_resize(sketch, img_size)

        # out = cv2.hconcat([sketch, img])
        # cv2.imwrite(os.path.join(to_dir, ext), out)

        cv2.imwrite(os.path.join(to_dir, ext), sketch)

        i += 1

    return i


def generate_data(from_dir, to_dir, img_size):
    # generate sketches
    os.makedirs(to_dir, exist_ok=True)

    folders = glob.glob(f'{from_dir}/*/')
    i = 0
    for f in folders:
        print(f)
        i = store_inputs(f, to_dir, img_size, i)


def get_data(input_dir):
    input_paths = get_images_paths(input_dir)
    inputs = []

    for f in input_paths:
        inputs.append(cv2.imread(f))

    inputs = np.array(inputs, dtype='float32')

    return inputs


def main():
    from_dir = "images/original"
    to_dir = "images/edges"
    img_size = 64
    generate_data(from_dir, to_dir, img_size)


if __name__ == '__main__':
    main()
