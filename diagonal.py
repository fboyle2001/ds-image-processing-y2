import cv2
import numpy as np
import os
import math



"""
Creates the Gaussian Mask to slide over the image
Required Parameters:
n - Used to create a (2 * n + 1, 2 * n + 1) sliding mask, sigma - s.d. used in the Gaussian
"""
def generate_gaussian_mask(n, sigma):
    # The Gaussian function as a lambda
    gaussian = lambda input, sigma: (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(input ** 2) / (2 * sigma ** 2))
    # Will store the masks rows and columns
    mask = []
    # The total of all of the gaussian(r, sigma) to normalise
    total = 0

    # Create a (2 * n + 1) * (2 * n + 1) mask
    for x in range(0, 2 * n + 1):
        row = []

        for y in range(0, 2 * n + 1):
            # Distance from centre
            r = math.sqrt((x - n) ** 2 + (y - n) ** 2)
            # Value of the gaussian at a point
            value = gaussian(r, sigma)
            total += value
            row.append(value)

        mask.append(row)

    # Normalise the result and convert it to a numpy array
    normalised = np.array([[element / total for element in row] for row in mask])
    return normalised

"""
Implementation of the Gaussian Filter
Required Parameters:
img - Source image, n - Used to create a (2 * n + 1, 2 * n + 1) sliding mask, sigma - s.d. used in the Gaussian
Optional Parameters:
width - apply to fixed vertical slice, height - apply to fixed horizontal slice,
separate - only returns the affected slice if this is True
"""
def gaussian_filter(img, n, sigma, width = None, height = None, separate = False):
    result = None

    # Deals with the optional parameters
    if separate:
        result = np.zeros(img.shape, np.uint8)
    else:
        result = np.copy(img)

    mask = generate_gaussian_mask(n, sigma)

    if width == None:
        width = (0, img.shape[0])

    if height == None:
        height = (0, img.shape[1])

    # The channels are stored in the shape
    channels = img.shape[2] if len(img.shape) == 3 else 1

    # Apply separately to each channel
    for channel in range(channels):
        # Pad either side so we can slide easily
        arr = None

        if channels != 1:
            arr = np.pad(img[:, :, channel], ((n, n), (n, n)), "edge")
        else:
            arr = np.pad(img[:, :], ((n, n), (n, n)), "edge")
        # Loop over each original value but with the necessary offset
        for x in range(width[0] + n, width[1] + n):
            for y in range(height[0], height[1]):
                # Select the neighbourhood of the pixel
                slide = arr[x - n : x + n + 1, y : y + 2 * n + 1]
                # Multiple the mask and slide and then sum the matrix elements
                # This is the Gaussian blur at the pixel's b/g/r channel
                if channels != 1:
                    result[x - n, y, channel] = np.sum(mask * slide).astype(np.uint8)
                else:
                    result[x - n, y] = np.sum(mask * slide).astype(np.uint8)

    return result

def vertical_motion_blur(img, size):
    kernel = np.zeros((size, size))
    kernel[:, int((size-1)/2)] = np.ones(size)
    kernel /= size
    vertical_mb = cv2.filter2D(img, -1, kernel)
    print(kernel)
    return vertical_mb

def horizontal_motion_blur(img, size):
    kernel = np.zeros((size, size))
    kernel[int((size - 1)/2), :] = np.ones(size)
    kernel /= size
    horizonal_mb = cv2.filter2D(img, -1, kernel)
    print(kernel)
    return horizonal_mb

def diagonal_stroke(img, x, y, gradient, thickness, segment_length):
    new_img = np.copy(img)
    func = lambda i: int(gradient * i)

    q = func(segment_length)
    m = thickness // 2
    y = y - thickness // 2
    for i in range(segment_length):
        #print(x + i, y + func(i))
        #print(x - i, y - func(i))
        #220 - ((1 - c / thickness) * (c / thickness) * 255)
        for c in range(thickness):
            new_img[y + func(i) + c, x + i] = 30
            new_img[y - func(i) + c, x - i] = 30

    return new_img

mode = "monochrome"

if mode == "monochrome":
    img = cv2.imread("face1.jpg", cv2.IMREAD_GRAYSCALE)
    # Laplacian to sharpen the edges
    kernel = np.array([[0, 1, 0], [1,-4,1], [0, 1, 0]])
    smoothed = gaussian_filter(img, 3, 1)
    laplacian = cv2.filter2D(smoothed, -1, kernel)
    sharpened_edges = cv2.absdiff(img, laplacian)
    sharpened_edges = horizontal_motion_blur(sharpened_edges, 3)
    #sharpened_edges = vertical_motion_blur(sharpened_edges, 1)
    equalised = cv2.equalizeHist(sharpened_edges)
    smoothed_equalised = cv2.bilateralFilter(equalised, 9, 5, 5)
    #smoothed_equalised = horizontal_motion_blur(smoothed_equalised, 2)
    alpha = 0.6
    blended = cv2.addWeighted(sharpened_edges, alpha, smoothed_equalised, 1 - alpha, 0)
    concat = np.concatenate((img, smoothed, sharpened_edges, equalised, blended), axis=1)
    cv2.imshow("OLD Displayed Image", concat)
    cv2.waitKey(0)
else:
    img_r = cv2.imread("face1.jpg", cv2.IMREAD_COLOR)
    channels = [0, 1]
    cp = np.copy(img_r)
    img = cv2.imread("face1.jpg", cv2.IMREAD_GRAYSCALE)
    cp[:, :, 0] = img
    cp[:, :, 1] = img
    cp[:, :, 2] = img

    for channel in channels:
        part = cp[:, :, channel]
        kernel = np.array([[0, 1, 0], [1,-4,1], [0, 1, 0]])
        smoothed = gaussian_filter(part, 3, 1)
        laplacian = cv2.filter2D(smoothed, -1, kernel)
        sharpened_edges = cv2.absdiff(part, laplacian)
        sharpened_edges = horizontal_motion_blur(sharpened_edges, 3)
        sharpened_edges = vertical_motion_blur(sharpened_edges, 1)
        equalised = cv2.equalizeHist(sharpened_edges)
        smoothed_equalised = cv2.bilateralFilter(equalised, 9, 5, 5)
        #smoothed_equalised = horizontal_motion_blur(smoothed_equalised, 2)
        alpha = 0.6
        blended = cv2.addWeighted(sharpened_edges, alpha, smoothed_equalised, 1 - alpha, 0)
        cp[:, :, channel] = blended

    concat = np.concatenate((img_r, cp), axis=1)
    cv2.imshow("Displayed Image", concat)
    cv2.waitKey(0)
