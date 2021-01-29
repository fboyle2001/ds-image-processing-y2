import cv2
import numpy as np
import os
import math

def gen_rainbow(width_range = None, height_range = None, hsv_max_hue_deg = 300, max_sat = 255, max_value = 255, background = None):
    new_img = None

    if background == None:
        new_img = np.zeros(img.shape, np.uint8)
    else:
        new_img = np.full(img.shape, background, np.uint8)

    height = new_img.shape[0]
    width = new_img.shape[1]

    if width_range == None:
        width_range = [0, width]

    if height_range == None:
        height_range = [0, height]

    max_width = abs(width_range[1] - width_range[0])
    width_offset = min(*width_range)

    for x in range(*width_range):
        for y in range(*height_range):
            hsv_deg = int((((x - width_offset) / max_width * hsv_max_hue_deg) / 360) * 179)
            new_img[y, x] = [hsv_deg, max_sat, max_value]

    return cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)

def blend_images(img, mask, amount, width_range = None, height_range = None):
    blended = np.copy(img)
    height, width, _ = img.shape

    if width_range == None:
        width_range = [0, width]

    if height_range == None:
        height_range = [0, height]

    for x in range(*height_range):
        for y in range(*width_range):
            blended[x, y] = blended[x, y] * amount + mask[x, y] * (1 - amount) + 2

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

def darken_image(img, coefficient):
    darkened = np.copy(img)
    darkened = cv2.cvtColor(darkened, cv2.COLOR_BGR2HSV)

    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            darkened[x, y, 2] = np.uint8(darkened[x, y, 2] * coefficient)

    return cv2.cvtColor(darkened, cv2.COLOR_HSV2BGR)

img = cv2.imread("face1.jpg", cv2.IMREAD_COLOR)
d = darken_image(img, 0.5)
width_range = [225, 245]
height_range = [120, 320]
template = gen_rainbow(width_range, height_range, background = [int((23 / 360) * 179), int(34 / 100 * 255), int(87 / 100 * 255)]).astype(np.uint8)
blurred_template = gaussian_filter(template, 9, 4, height=width_range, separate=True)
alpha = 0.8
blended = blend_images(d, blurred_template, alpha, width_range, height_range)
blended_blurred = gaussian_filter(blended, 2, 0.5, height=width_range, width=height_range, separate=False)

concat = np.concatenate((img, d, blurred_template, blended), axis=1)
cv2.imshow("Displayed Image", concat)
cv2.waitKey(0)
