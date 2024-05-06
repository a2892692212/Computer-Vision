import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from matplotlib import pyplot as plt


def multiband_blending(img1, img2, mask, levels=4):
    # Ensure that the dimension of the images and mask are divisible by 2**(levels-1).
    if img1.shape[0] % 2**(levels - 1) != 0:
        img1 = img1[:img1.shape[0] - img1.shape[0] % 2 ** (levels - 1), :, :]
        img2 = img2[:img2.shape[0] - img2.shape[0] % 2**(levels-1), :, :]
        mask = mask[:mask.shape[0] - mask.shape[0] % 2**(levels-1), :, :]

    if img1.shape[1] % 2**(levels - 1) != 0:
        img1 = img1[:, 0:img1.shape[1] - img1.shape[1] % 2 ** (levels - 1), :]
        img2 = img2[:, 0:img2.shape[1] - img2.shape[1] % 2**(levels-1), :]
        mask = mask[:, 0:mask.shape[1] - mask.shape[1] % 2**(levels-1), :]

    # Generate Gaussian pyramids for both images.
    gaussian_pyr1 = create_gaussian_pyramid(img1, levels)
    gaussian_pyr2 = create_gaussian_pyramid(img2, levels)

    # Generate Laplacian pyramids by subtracting each level of Gaussian pyramid from its expanded version.
    lap1 = []
    lap2 = []

    for i in range(levels-1):
        lap1.append(cv2.subtract(gaussian_pyr1[i], cv2.pyrUp(gaussian_pyr1[i+1])))
        lap2.append(cv2.subtract(gaussian_pyr2[i], cv2.pyrUp(gaussian_pyr2[i+1])))

    lap1.append(gaussian_pyr1[-1])  # Append the last level of the Gaussian pyramid.
    lap2.append(gaussian_pyr2[-1])  # Append the last level of the Gaussian pyramid.

    # Blend the Laplacian pyramids using the Gaussian pyramid of the mask.
    LS = []
    mask_pyramid = create_gaussian_pyramid(mask, levels)
    for i in range(len(lap1)):
        blended = mask_pyramid[i] * lap1[i] + (1 - mask_pyramid[i]) * lap2[i]
        LS.append(blended)

    # Reconstruct the blended image from the blended pyramid.
    LS = LS[::-1]
    img_blend = LS[0]
    for i in range(1, levels):
        img_blend = cv2.add(cv2.pyrUp(img_blend), LS[i])

    return img_blend


def simple_blending(img1, img2, mask):
    """
    Perform simple blending of two images using a mask.
    """
    # Perform the blending operation
    blended = mask * img1 + (1 - mask) * img2
    return blended


def create_gaussian_pyramid(img, levels):
    # Initialize the Gaussian pyramid list with the original image.
    gaussian_pyramid = [img.copy()]

    # Create the pyramid by progressively blurring and downsampling the image.
    for i in range(levels - 1):
        blur = cv2.GaussianBlur(gaussian_pyramid[i], (3, 3), 3, borderType=cv2.BORDER_CONSTANT)
        gaussian_pyramid.append(cv2.pyrDown(blur))

    return gaussian_pyramid
