import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import axis

from tensorflow.keras.preprocessing import image_dataset_from_directory

def import_images(dir_path, image_size, categorical_labels=False, to_numpy=False):
    if categorical_labels:
        (images, labels) = image_dataset_from_directory(
            dir_path,
            labels="inferred",
            image_size=image_size,
            batch_size=None,
            label_mode='categorical'
        )
        if to_numpy:
            return np.array(list(images.as_numpy_iterator())), labels
        else:
            return images, labels

    images = image_dataset_from_directory(
        dir_path,
        labels=None,
        image_size=image_size,
        batch_size=None,
        label_mode='categorical'
    )
    if to_numpy:
        return np.array(list(images.as_numpy_iterator()))
    else:
        return images

def rotate_all_90(images):
    return np.rot90(images, axes=(1,2))

def normalize_images(images):
    return images.astype('float32') / 255.0

def mirror_images(images, axis=0):
    if axis == 0:
        return images[:, ::-1, :, :].copy()
    elif axis == 1:
        return images[:, :,::-1,:].copy()
    else:
        raise Exception("Only valid axis are 0 or 1 (x mirror or y mirror)")
