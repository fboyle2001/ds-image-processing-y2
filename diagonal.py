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

def horizontal_motion_blur(img, size):
    kernel = np.zeros((size, size))
    kernel[int((size - 1)/2), :] = np.ones(size)
    kernel /= size
    horizonal_mb = cv2.filter2D(img, -1, kernel)
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

img = cv2.imread("face1.jpg", cv2.IMREAD_GRAYSCALE)
equ = cv2.equalizeHist(img)
kernel = np.array([[0, 1, 0], [1,-4,1], [0, 1, 0]])
im = cv2.filter2D(img, -1, kernel)
q = img - np.uint8(0.2*im)

#im = horizontal_motion_blur(im, 4)

ivr = cv2.bitwise_not(img)

j = 10
diagonal = np.ones(img.shape, np.uint8) * 255
# for x in range(40, 360, j):
#     #int(math.sqrt(x ** 2)) works up to 200
#     if x > 200:
#         z = int(int(math.sqrt((400 - x) ** 2)) * 0.8)
#         print(x, z)
#         diagonal = diagonal_stroke(diagonal, x, x, -1, 2 * j, z)
#     else:
#         z = int(int(math.sqrt(x ** 2))  * 0.9)
#         diagonal = diagonal_stroke(diagonal, x, x, -1, 2 * j, z)


diagonal = diagonal_stroke(diagonal, 50, 200, 1, 20, 30)

equ = cv2.bilateralFilter(equ, 5, 0.5, 0.5)#gaussian_filter(equ, 2, 0.5)
mb = horizontal_motion_blur(equ, 2)

alpha = 0.7
beta = 1 - alpha
gamma = 0
dist = cv2.addWeighted(img, alpha, mb, beta, gamma)
dist = horizontal_motion_blur(dist, 2)
dist = cv2.addWeighted(dist, alpha, im, beta, 20)
#dist = cv2.addWeighted(dist, alpha, im, beta, 0)

concat = np.concatenate((q, im, equ, mb, dist), axis=1)
cv2.imshow("Displayed Image", concat)
cv2.waitKey(0)
