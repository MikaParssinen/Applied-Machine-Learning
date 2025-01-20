import numpy as np

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def psnr(img1, img2, max_value=1.0):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    error = mse(img1, img2)
    if error == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(error)))

def get_avarage_psnr(before, after):
    count = len(after)
    psnr_list = []

    for img1, img2 in zip(before, after):
        temp_psnr = psnr(img1, img2)
        psnr_list.append(temp_psnr)

    return sum(psnr_list) / count