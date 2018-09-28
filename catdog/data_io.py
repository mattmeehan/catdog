
import os
import numpy as np
from skimage import io
from skimage.transform import resize


TRAIN_DIR = '../train/'
TEST_DIR = '../test1/'


def convert_image(img_path, size=50):
    """Load an image, resize, and convert to grayscale.

    Parameters
    ----------
    img_path : str
        Path to the image
    size : int
        Length of each side of resized image in pixels

    Returns
    -------
    img : np.ndarray
        Array containing image pixel values in grayscale
    """

    img = io.imread(img_path, as_gray=True)
    img = resize(img, [size,size])
    assert img.shape == (size,size), f'Image shape is {img.shape}, expected ({size},{size})'

    return img


