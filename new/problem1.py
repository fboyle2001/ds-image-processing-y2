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

def generate_light_mask(img, brighten, left, right, upper, bottom):
    mask = np.zeros((img.shape[0], img.shape[1]))

    return mask

def white_slice_mask(img, left, right, upper, bottom):
    mask = np.copy(img)
    img_height, img_width, img_channels = mask.shape

    for x in range(img_width):
        for y in range(img_height):
            #print(x, y, left(x), right(x))
            if upper(x) <= y <= bottom(x) and left(y) <= x <= right(y):
                factor = (200 - abs(y - 200)) / 200
                mask[y, x] = np.array([255 * factor, 255 * factor, 255 * factor], dtype=np.uint8)
                continue

            mask[y, x] = [0, 0, 0]

    return mask

def merge_images(img, mask):
    copy = np.copy(img)
    img_height, img_width, img_channels = mask.shape

    for x in range(img_width):
        for y in range(img_height):
            if np.sum(mask[y, x]) != 0:
                copy[y, x] = mask[y, x]

    return copy

def darken_image(img, coefficient):
    darkened = np.copy(img)
    darkened = cv2.cvtColor(darkened, cv2.COLOR_BGR2HSV)

    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            darkened[x, y, 2] = np.uint8(darkened[x, y, 2] * coefficient)

    return cv2.cvtColor(darkened, cv2.COLOR_HSV2BGR)

def different_white(img, left, right, upper, bottom):
    mask = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2HSV)
    img_height, img_width, img_channels = mask.shape
    hsv_max_hue_deg = 300
    max_sat = 255
    max_value = 255

    half = img_height // 2

    for x in range(img_width):
        for y in range(img_height):
            #print(x, y, left(x), right(x))
            if upper(x) <= y <= bottom(x) and left(y) <= x <= right(y):
                #factor = max(0, -15 * (y / 100 - 2) ** 4 + 200) / half
                #factor = (half - abs(y - half)) / half
                factor = ((-1 / half) * (y - half) ** 2 + half) / half

                mask[y, x, 2] = np.uint8(mask[y, x, 2] * factor)
                continue

            mask[y, x] = [0, 0, 0]

    return cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

def rainbow_slice_mask(img, left, right, upper, bottom):
    mask = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2HSV)
    img_height, img_width, img_channels = mask.shape
    hsv_max_hue_deg = 300
    max_sat = 255
    max_value = 128

    width_start = img_width
    width_end = 0
    half = img_height // 2

    for x in range(img_width):
        for y in range(img_height):
            if left(y) <= x <= right(y) and upper(x) <= y <= bottom(x):
                if x < width_start:
                    width_start = x

                if x > width_end:
                    width_end = x

    max_width = width_end - width_start

    for x in range(img_width):
        for y in range(img_height):
            #print(x, y, left(x), right(x))
            if upper(x) <= y <= bottom(x) and left(y) <= x <= right(y):
                hsv_deg = int((((x - width_start) / max_width * hsv_max_hue_deg) / 280) * 179)
                factor = ((-1 / half) * (y - half) ** 2 + half) / half
                mask[y, x] = [hsv_deg, max_sat, np.uint8(mask[y, x, 2] * factor)]
                continue

            mask[y, x] = [0, 0, 0]

    return cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

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

upper = lambda x: 0
bottom = lambda x: 400
#left = lambda y: 255
#right = lambda y : 275
left = lambda y: -0.1 * y + 255
right = lambda y: -0.1 * y + 275

img = cv2.imread("face1.jpg", cv2.IMREAD_COLOR)
mode = "rainbow"

if mode == "simple":
    dark = darken_image(img, 0.4)
    sliced = different_white(img, left, right, upper, bottom)
    sliced = gaussian_filter(sliced, 4, 2.2)
    merged = cv2.addWeighted(dark, 1, sliced, 0.7, 0)
    concat = np.concatenate((img, dark,sliced, merged), axis=1)
    cv2.imshow("Displayed Image", concat)
    cv2.waitKey(0)
else:
    dark = darken_image(img, 0.5)
    sliced = rainbow_slice_mask(img, left, right, upper, bottom)
    sliced = gaussian_filter(sliced, 4, 2.2)
    merged = cv2.addWeighted(dark, 1, sliced, 0.5, 0)
    concat = np.concatenate((img, dark,sliced, merged), axis=1)
    cv2.imshow("Displayed Image", concat)
    cv2.waitKey(0)
