import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils.blend import multiband_blending, simple_blending


def generate_panorama(video_path, option):
    """
    Generates a panorama from a video by stitching frames using either multiband or simple blending based on the provided option.

    Parameters:
        video_path (str): Path to the video file.
        option (str): Blending option, 'Option 1' for multiband blending or others for simple blending.

    Returns:
        numpy.ndarray: The panorama image.
    """
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    panorama = prev_frame
    idx = 0
    step = 30

    while True:
        idx += 1
        ret, frame = cap.read()

        if not ret:
            break

        if idx % step == 0:
            # Match histograms for color correction
            frame = match_histograms(frame, panorama)

            # Prepare frame for stitching
            mask = np.ones_like(panorama[:, :-200, :], dtype='uint8')
            panorama = cv2.copyMakeBorder(panorama, 0, 0, 0, frame.shape[1], cv2.BORDER_CONSTANT)
            mask = cv2.copyMakeBorder(mask, 0, 0, 0, frame.shape[1] + 200, cv2.BORDER_CONSTANT)

            # Compute transformation to align new frame to panorama
            M = match_transform(frame, panorama)
            img_warped = cv2.warpPerspective(frame, M, (panorama.shape[1], panorama.shape[0]))

            # Apply blending based on selected option
            if option == 'Option 1':
                panorama = multiband_blending(panorama, img_warped, mask)
            else:
                panorama = simple_blending(panorama, img_warped, mask)

            # Crop the black borders if any
            for col in range(panorama.shape[1] - 1, -1, -1):
                if (panorama[:, col] == [0, 0, 0]).sum() < 100:
                    last_black_col = col
                    break
            panorama = panorama[:, :last_black_col + 1]

    cap.release()

    return panorama


def match_transform(src, dst):
    """
    Computes a homography matrix that aligns keypoints between two images using SIFT and FLANN-based matching.

    Parameters:
        src (numpy.ndarray): Source image in BGR color space.
        dst (numpy.ndarray): Destination image in BGR color space.

    Returns:
        numpy.ndarray: The homography matrix that warps the source image to align with the destination.
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Set FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    # Create the FLANN matcher object
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform matching using KNN algorithm
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # Extract location of good matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Find homography matrix using RANSAC method
    HM, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return HM


def match_histograms(source, template):
    """
    Match the histogram of the source image to that of the template image.
    """
    # Convert BGR to YCrCb color space (separating luminance from chrominance)
    src = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)
    tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2YCrCb)

    # Compute and apply histogram matching
    for i in range(3):  # Process each channel separately
        # Compute the histograms
        src_hist, _ = np.histogram(src[:, :, i], 256, [0, 256])
        tmpl_hist, _ = np.histogram(tmpl[:, :, i], 256, [0, 256])

        # Compute the cumulative distribution function (CDF) for both images
        cdf_src = np.cumsum(src_hist) / np.sum(src_hist)
        cdf_tmpl = np.cumsum(tmpl_hist) / np.sum(tmpl_hist)

        # Build a mapping function from the CDFs using interpolation
        interp_map = np.interp(cdf_src, cdf_tmpl, np.arange(256))
        src[:, :, i] = interp_map[src[:, :, i]].astype('uint8')

    # Convert the modified YCrCb image back to BGR
    result = cv2.cvtColor(src, cv2.COLOR_YCrCb2BGR)
    return result
