import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sns

from sklearn.metrics import f1_score, classification_report, confusion_matrix

def display_before_after(before, after, n=4):
    """Displays n random images from each array."""
    indices = np.random.randint(len(before), size=n)

    images1 = before[indices, :]
    images2 = after[indices, :]

    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()



def plot_img(train_data, class_names):
    """Plot 8 images from each class."""
    plt.figure(figsize=(8, 5))
    for images, labels in train_data.take(1):  # Take one batch from the dataset
        for i in range(8):
            ax = plt.subplot(2, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[tf.argmax(labels[i]).numpy()])
            plt.axis("off")
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.show()


def plot_metrics(history, model_type, accuracy=True):
    """Displays metrics plots of history."""
    # Plot traning and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} Training and Validation Loss')
    plt.legend()

    if accuracy:
        # Plot traning and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_type} Training and Validation Accuracy')
        plt.legend()
        plt.show()



def plot_heatmap(y_true, y_pred):
    """Plots heatmap"""
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)

    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix Heatmap")
    plt.show()


def plot_images_w_pred_n_true(anomaly_images, anomaly_labels, predictions):
    class_names = {
        0: "Broken Large",
        1: "Broken Small",
        2: "Contaminated",
        3: "Good"
    }
    rows = 2
    cols = 5
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(anomaly_labels, axis=1)

    total_images = len(anomaly_images)

    for batch_start in range(0, total_images, rows * cols):
        batch_end = min(batch_start + rows * cols, total_images)
        num_subplots = batch_end - batch_start

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))  # Create grid

        for i in range(num_subplots):
            row = i // cols
            col = i % cols
            img_index = batch_start + i  # Actual index in anomaly_images

            pred_label = class_names.get(y_pred[img_index], "Unknown")
            true_label = class_names.get(y_true[img_index], "Unknown")

            axes[row, col].imshow(anomaly_images[img_index])
            axes[row, col].axis('off')
            axes[row, col].set_title(f"Pred: {pred_label}\nTrue: {true_label}")

        # Hide any empty subplots (only needed if the last batch has < rows * cols images)
        for j in range(num_subplots, rows * cols):
            fig.delaxes(axes.flatten()[j])

        plt.tight_layout()
        plt.show()


