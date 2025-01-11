import numpy as np
from matplotlib import pyplot as plt

def display_before_after(before, after, n=6):
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