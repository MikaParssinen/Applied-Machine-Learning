import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import axis
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def import_images(dir_path, image_size, categorical_labels=False, to_numpy=False, batch_size=None, dataset=False):

    if dataset:
        data = image_dataset_from_directory(
            dir_path,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='categorical'

        )
        return data
    else:
        if categorical_labels:
            (images, labels) = image_dataset_from_directory(
                dir_path,
                labels="inferred",
                image_size=image_size,
                batch_size=batch_size,
                label_mode='categorical'
            )
            if to_numpy:
                return np.array(list(images.as_numpy_iterator())), labels
        else:
            images = image_dataset_from_directory(
                dir_path,
                labels=None,
                image_size=image_size,
                batch_size=batch_size,
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


def normalize_dataset(train_data, test_data):
    """Normalize the data to have a value between 0 and 1"""
    def normalize(image, label):
        image = image / 255.0
        return image, label

    train_data_norm = train_data.map(normalize)
    test_data_norm = test_data.map(normalize)

    return test_data_norm, train_data_norm

def mirror_images(images, axis=0):
    if axis == 0:
        return images[:, ::-1, :, :].copy()
    elif axis == 1:
        return images[:, :,::-1,:].copy()
    else:
        raise Exception("Only valid axis are 0 or 1 (x mirror or y mirror)")


def add_gaussian_noise(image, label):
    """Adds Gaussian noise to the image."""
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=2.0)
    noisy_image = image + noise
    return noisy_image, label


def add_salt_and_pepper_noise(image, label, salt_prob=0.04, pepper_prob=0.04):
    """Adds salt and pepper noise to the image."""
    random_vals = tf.random.uniform(shape=tf.shape(image), minval=0.0, maxval=1.0)
    salt = random_vals < salt_prob
    image = tf.where(salt, 255.0, image)

    pepper = random_vals > (1.0 - pepper_prob)
    image = tf.where(pepper, 0.0, image)
    return image, label