import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import axis
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def get_dataset(dir_path, image_size):
    images = []
    labels = []
    data = image_dataset_from_directory(
        dir_path,
        labels="inferred",
        image_size=image_size,
        batch_size=None,
        label_mode='categorical'
    )
    for image, label in data.take(-1):
        images.append(image)
        labels.append(label)

    X = np.array(images)
    y = np.array(labels)

    return X, y

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
                return images, labels
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


def add_gaussian_noise(images, mean=0.0, stddev=2.0):
    """Adds Gaussian noise to images."""
    noise = np.random.normal(mean, stddev, images.shape)
    noisy_images = np.clip(images + noise, 0, 255)  # Ensure pixel values stay in range
    return noisy_images.astype(np.uint8)

def add_salt_and_pepper_noise(images, salt_prob=0.04, pepper_prob=0.04):
    """Adds salt and pepper noise to a batch."""
    noisy_images = images.copy()

    # Generate random mask for salt & pepper
    salt_mask = np.random.rand(*images.shape) < salt_prob
    pepper_mask = np.random.rand(*images.shape) > (1 - pepper_prob)

    noisy_images[salt_mask] = 255  # Set salt pixels to white
    noisy_images[pepper_mask] = 0  # Set pepper pixels to black

    return noisy_images.astype(np.uint8)

def convert_to_numpy(test_data):
    """Saves labels and images to numpy arrays."""
    all_labels = []
    all_images = []

    for images, labels in test_data:
        all_labels.append(labels.numpy())
        all_images.append(images.numpy())

    all_labels_np = np.concatenate(all_labels, axis=0)
    all_images_np = np.concatenate(all_images, axis=0)

    return all_labels_np, all_images_np

def normal_anomaly_split(X_train, y_train):
    pass