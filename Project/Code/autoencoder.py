import preprocessing
import numpy as np
import cv2

from skimage.metrics import structural_similarity
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

def create_image_dataset(dir_path, image_size):
    """
    - Loads images from the directory dir_path
    - Creates a new set of images that are the original images mirrored in both axis
    - Rotates all images 90, 180 and 270 degrees and adds to the set of images
    - The returning set of images is 12 times as big, where an image have been
      manipulated to create 11 new versions of each image
    """

    # Import and normalize original images
    images = preprocessing.import_images(dir_path, image_size, to_numpy=True)
    images = preprocessing.normalize_images(images)

    # Create new images by mirroring the images in both axis
    x_mirrored = preprocessing.mirror_images(images, 0)
    y_mirrored = preprocessing.mirror_images(images, 1)
    images = np.concatenate((images, x_mirrored, y_mirrored))

    # Create new images by rotating the images 90, 180 and 270 degrees
    rotated_90 = preprocessing.rotate_all_90(images)
    rotated_180 = preprocessing.rotate_all_90(rotated_90)
    rotated_270 = preprocessing.rotate_all_90(rotated_180)
    images = np.concatenate((images, rotated_90, rotated_180, rotated_270))

    return images


def build_and_compile(image_size):
    encoder_input = Input(shape=(image_size[0], image_size[1], 3))

    # Encoding layers
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(encoder_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)

    encoded_shape = x.shape[1:]
    print(encoded_shape)

    # Flatten layers
    x = Flatten()(x)
    flat_length = x.shape[1]
    print(flat_length)
    x = Dense(1024, activation="relu")(x)
    x = Dense(flat_length, activation="relu")(x)
    x = Reshape(encoded_shape)(x)

    # Decoding layers
    x = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    # Create and compile autoencoder
    optimizer = Adam(learning_rate=0.0005)
    autoencoder = Model(encoder_input, x)
    autoencoder.compile(optimizer=optimizer, loss="mse")

    return autoencoder

def fit_model(model, images, epochs=25, batch_size=16, callbacks=[]):
    history = model.fit(
        x=images,
        y=images,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.2,
        callbacks=callbacks,
    )
    return history

def get_masks(before, after, win_size=9, gaussian_weights=False, threshold=0):
    scores = []
    masks = []
    for (image1, image2) in zip(before, after):
        (score, diff) = structural_similarity(image1, image2, win_size=win_size, data_range=1.0, full=True,
                                              channel_axis=2, gaussian_weights=gaussian_weights)
        diff = np.clip(diff, a_min=0.0, a_max=1.0)
        scores.append(score)
        _, mask = cv2.threshold(diff, threshold, 1.0, cv2.THRESH_BINARY)
        masks.append(mask)
    return np.array(scores), np.array(masks)

def calculate_threshold(normal_masks, anomaly_masks):
    min_normal = np.min(normal_masks)
    max_anomaly = np.max(anomaly_masks)
    prediction_threshold = (min_normal + max_anomaly) / 2

    print(min_normal)
    print(max_anomaly)
    print(prediction_threshold)

def predict_anomalies(before, after, threshold):
    y = []

    for (image1, image2) in zip(before, after):
        score = structural_similarity(image1, image2, win_size=9, data_range=1.0,
                                              channel_axis=2, gaussian_weights=False)
        if score > threshold:
            y.append(False)
        else:
            y.append(True)

    return np.array(y)