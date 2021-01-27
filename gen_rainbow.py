import cv2
import numpy as np
import os
import math

def gen_rainbow(width_range = None, hsv_max_hue_deg = 300, max_sat = 255, max_value = 255):
    new_img = np.zeros(img.shape, np.uint8)
    width = new_img.shape[1]

    if width_range == None:
        width_range = [0, width]

    max_width = abs(width_range[1] - width_range[0])
    width_offset = min(*width_range)

    for x in range(*width_range):
        for y in range(100, 300):
            hsv_deg = int((((x - width_offset) / max_width * hsv_max_hue_deg) / 360) * 179)
            new_img[y, x] = [hsv_deg, max_sat, max_value]

    return cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)

def blend_images(img, mask, amount):
    blended = np.copy(img)
    height, width, _ = img.shape

    for x in range(width):
        for y in range(height):
            blended[x, y] = blended[x, y] * amount + mask[x, y] * (1 - amount)

    for x in range(width):
        for y in range(height):
            pass

    return blended.astype(np.uint8)

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
    channels = img.shape[2]

    # Apply separately to each channel
    for channel in range(channels):
        # Pad either side so we can slide easily
        arr = np.pad(img[:, :, channel], ((n, n), (n, n)), "edge")
        # Loop over each original value but with the necessary offset
        for x in range(width[0] + n, width[1] + n):
            for y in range(height[0], height[1]):
                # Select the neighbourhood of the pixel
                slide = arr[x - n : x + n + 1, y : y + 2 * n + 1]
                # Multiple the mask and slide and then sum the matrix elements
                # This is the Gaussian blur at the pixel's b/g/r channel
                result[x - n, y, channel] = np.sum(mask * slide).astype(np.uint8)

    return result

from pprint import pprint
def gaussian_pixel_blur(img, pixel, n, sigma):
    height, width, channels = img.shape
    x, y = pixel
    gaussian = lambda input, sigma: (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(input ** 2) / (2 * sigma ** 2))

    grid = []
    total = 0
    #print((max(x - n, 0),min(x + n + 1, width - 1)), (max(y - n, 0),min(y + n + 1, height - 1)))
    neighbourhood = img[max(x - n, 0):min(x + n + 1, width), max(y - n, 0):min(y + n + 1, height)]
    for px in range(max(x - n, 0), min(x + n + 1, width)):
        row = []

        for py in range(max(y - n, 0), min(y + n + 1, height)):
            r = math.sqrt((px - x) ** 2 + (py - y) ** 2)
            value = gaussian(r, sigma)
            total += value
            row.append(value)

        grid.append(row)


    normalised = np.array([[element / total for element in row] for row in grid])
    #print(normalised)
    #print(neighbourhood)

    new_pixel = [0 for channel in range(channels)]

    for i, row in enumerate(neighbourhood):
        for j, pixel in enumerate(row):
            for channel in range(channels):
                new_pixel[channel] += pixel[channel] * normalised[i][j]

    #print(img[x, y], np.array(new_pixel).astype(np.uint8))
    img[x, y] = np.array(new_pixel).astype(np.uint8)
    return img

def blur_in_rectangle(new, width_range, n, sigma):
    new_img = np.copy(new)

    for x in range(*width_range):
        for y in range(new.shape[0]):
            new_img = gaussian_pixel_blur(new_img, (y, x), n, sigma)

    return new_img

def blur_boundary(new, cols):
    new_img = np.copy(new)

    for col in cols:
        for y in range(new.shape[0]):
            new_img = gaussian_pixel_blur(new_img, (y, col[0]), col[1], col[2])

    return new_img

img = cv2.imread("face1.jpg", cv2.IMREAD_COLOR)
gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

n, sigma = 3, 1

gauss = gaussian_filter(img, n, sigma)
cv2g = cv2.GaussianBlur(img, (2 * n + 1, 2 * n + 1), sigma)

concat = np.concatenate((img, cv2g, gauss), axis=1)
cv2.imshow("Displayed Image", concat)

cv2.waitKey(0)
import sys
sys.exit(0)

#gaussian_filter(np.array([[x for x in range(y, y + 9)] for y in range(0, 81, 9)]), 2, 0.5)


img = cv2.imread("face1.jpg", cv2.IMREAD_COLOR)
width_range = [200, 300]
template = gen_rainbow(width_range)
template = template.astype(np.uint8)

# size=5
# kernel_motion_blur = np.zeros((size, size))
# kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
# kernel_motion_blur = kernel_motion_blur / size
# template = cv2.filter2D(template, -1, kernel_motion_blur)
#

alpha = 0.9
copy = blur_in_rectangle(template, width_range, 9, 4)
new = blend_images(img, copy, alpha)
new_2 = blur_in_rectangle(new, width_range, 2, 0.5)
new_3 = blur_boundary(new_2, [(220, 9, 2), (260, 9, 2)])
#edged = cv2.GaussianBlur(gr, (5, 5), 0)
#edged = cv2.Canny(edged, 100, 300)
#edged = cv2.threshold(edged, 1, 255, cv2.THRESH_BINARY_INV)[0]
#print(edged)

concat = np.concatenate((img, new, new_2, new_3), axis=1)
cv2.namedWindow("Displayed Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Displayed Image", 1600, 400)
cv2.imshow("Displayed Image", concat)
cv2.waitKey(0)
