
from __future__ import division, print_function
import os
import glob
import numpy as np
from skimage import io
from skimage.transform import resize


# TRAIN_DIR and TEST_DIR contain raw jpg images from kaggle
TRAIN_DIR = './train/'
TEST_DIR = './test1/'
# DATA_DIR contains 4000 resized, grayscale images saved as npy files
DATA_DIR = './train_data/'


def load_image(img_path, size=50):
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


def load_training_images(n_images=2000, size=50):
    """Load training images and create one-hot encoded labels for each image.

    Parameters
    ----------
    n_images : int
        Number of images from each category to load; total images = 2*n_images
    size : int
        Length of each side of resized image in pixels
    
    Returns
    -------
    data : np.ndarray, shape=(2*n_images, size, size)
        Array of grayscale training images
    labels : np.ndarray, shaep=(2*n_images, 2)
        Array of one-hot encoded labels
    """
    
    data = []
    labels = []
  
    assert os.path.exists(TRAIN_DIR), f'Training image directory {TRAIN_DIR} does not exist.'

    if n_images > 12500:
        print('Requested more than the 12500 available training images of each type. Loading all images.')
        n_images = 12500

    for img_path in glob.glob(os.path.join(TRAIN_DIR, 'cat*.jpg'))[:n_images]:
        data.append(load_image(img_path, size=size))
        labels.append(np.array([1,0]))

    for img_path in glob.glob(os.path.join(TRAIN_DIR, 'dog*.jpg'))[:n_images]:
        data.append(load_image(img_path, size=size))
        labels.append(np.array([0,1]))
    
    data = np.array(data)
    labels = np.array(labels)
    shape_data = (2*n_images, size, size)
    shape_labels = (2*n_images, 2)
    assert data.shape == shape_data, f'Expected shape of data: {shape_data}, got: {data.shape}'
    assert labels.shape == shape_labels, f'Expected shape of labels: {shape_labels}, got: {labels.shape}'
    
    return data, labels


def load_resized_images(image_dir=DATA_DIR):
    """Load 4000 50x50 grayscale training images and associated labels.

    Parameters
    ----------
    image_dir : str
        Directory containing resized images and labels

    Returns
    -------
    data : np.ndarray, shape=(4000, size, size)
        Array of 50x50 grayscale training images (2000 of each type)
    labels : np.ndarray, shaep=(4000, 2)
        Array of one-hot encoded labels, e.g. [1,0] == cat
    """

    data_path = os.path.join(image_dir, 'training_data.npy')
    labels_path = os.path.join(image_dir, 'training_labels.npy')
    assert os.path.exists(data_path), f'{data_path} does not exist'
    assert os.path.exists(labels_path), f'{labels_path} does not exist'

    data = np.load(data_path)
    labels = np.load(labels_path)

    return data, labels
