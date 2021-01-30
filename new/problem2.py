import cv2
import numpy as np
import os
import math
from matplotlib import pyplot as plt

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

"""
Applies the Laplacian to the image to isolate the edges
"""
def laplacian(img):
    copy = np.copy(img)
    # The mask from the lectures
    mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # Convolve
    copy = cv2.filter2D(copy, -1, mask)
    return copy

"""
Sharpen the edges of the image
"""
def sharpen_edges(img, edges):
    return cv2.absdiff(img, edges)

"""
Histogram equalisation of a greyscale image
"""
def histogram_equalisation(img):
    img_height, img_width = img.shape
    # Initialise the bins
    bins = [0 for i in range(256)]

    # Fill the bins
    for x in range(img_width):
        for y in range(img_height):
            bins[img[y, x]] += 1

    # Calculate the cumulative sum of the histogram
    cumulative = [sum(bins[:i]) for i in range(256)]

    # New image to store the equalised image
    equalised = np.zeros(img.shape, np.uint8)
    # We need to divide by maximum and then scale to 0-255
    scale_factor = 255 / max(cumulative)

    for x in range(img_width):
        for y in range(img_height):
            # Scale the image based on the bins
            equalised[y, x] = np.uint8(cumulative[img[y, x]] * scale_factor)

    return equalised

"""
Applies a motion blur effect to the image
"""
def motion_blur(img, n, direction):
    copy = np.copy(img)
    size = 2 * n + 1
    # Create an empty (2 * n + 1) x (2 * n + 1) matrix
    matrix = np.zeros((size, size))

    if direction == "vertical":
        # For vertical we want to fill the centre column, this is just n
        for x in range(size):
            matrix[x, n] = 1
    else:
        # Default is horizontal
        # For vertical we want to fill the centre row, this is just n
        for y in range(size):
            matrix[n, y] = 1

    # Now normalise the matrix
    matrix /= size

    # Convolve the image with the matrix
    copy = cv2.filter2D(copy, -1, matrix)
    return copy

"""
Creates the Gaussian Mask to slide over the image
Required Parameters:
n - Used to create a (2 * n + 1, 2 * n + 1) sliding mask, sigma - s.d. used in the Gaussian
"""
def generate_spatial_gaussian_mask(n, sigma):
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
    #normalised = np.array([[element / total for element in row] for row in mask])
    return np.array(mask)

def generate_intensity_gaussian_mask(slide, n, sigma):
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
            r = abs(int(slide[x, y]) - int(slide[n, n]))
            #print(r, slide[x, y], slide[n, n], x, y, n)
            # Value of the gaussian at a point
            value = gaussian(r, sigma)
            total += value
            row.append(value)

        mask.append(row)

    # Normalise the result and convert it to a numpy array
    #normalised = np.array([[element / total for element in row] for row in mask])
    return np.array(mask)

# In my experimentation, this was 3x faster than my initial implementation
def bilateral_filter_greyscale(img, n, sigma_colour, sigma_space):
    result = np.zeros(img.shape, np.uint8)
    space_mask = generate_spatial_gaussian_mask(n, sigma_space)

    width = (0, img.shape[0])
    height = (0, img.shape[1])

    # Pad either side so we can slide easily
    arr = np.pad(img[:, :], ((n, n), (n, n)), "edge")
    # Loop over each original value but with the necessary offset
    for x in range(width[0] + n, width[1] + n):
        for y in range(height[0], height[1]):
            # Select the neighbourhood of the pixel
            slide = arr[x - n : x + n + 1, y : y + 2 * n + 1]
            colour_mask = generate_intensity_gaussian_mask(slide, n, sigma_colour)
            dual_mask = space_mask * colour_mask
            # print(dual_mask, np.sum(dual_mask))
            # Multiple the mask and slide and then sum the normalised matrix elements
            # This is the Gaussian blur at the pixel
            result[x - n, y] = np.uint8(np.sum(dual_mask * slide) / np.sum(dual_mask))

    return result

"""
Replicates cv2.addWeighted
alpha + beta does not have to sum to 1
"""
def add_images(img, alpha, secondary, beta, gamma = 0):
    # Create a blank image of the same shape
    added = np.zeros(img.shape, np.uint8)
    img_height, img_width = img.shape

    # Loop each pixel and add them according to the weights
    for x in range(img_width):
        for y in range(img_height):
            added[y, x] = np.uint8(img[y, x] * alpha + secondary[y, x] * beta + gamma)

    return added

img = cv2.imread("face1.jpg", cv2.IMREAD_GRAYSCALE)
mode = "monocharome"

if mode == "monochrome":
    smoothed = gaussian_filter(img, 3, 1)
    sharpened_edges = sharpen_edges(img, laplacian(smoothed))
    sharpened_edges = motion_blur(sharpened_edges, 1, "horizontal")
    sharpened_edges = motion_blur(sharpened_edges, 1, "vertical")
    equalised = histogram_equalisation(sharpened_edges)
    equalised = motion_blur(equalised, 0, "horizontal")
    noise = bilateral_filter_greyscale(equalised, 2, 0.5, 0.5)
    output = add_images(sharpened_edges, 0.6, noise, 0.4)

    concat = np.concatenate((img, output), axis=1)
    cv2.imshow("NEW Displayed Image", concat)
    cv2.waitKey(0)
else:
    # Make all the channels have the same
    coloured_image = cv2.imread("face1.jpg", cv2.IMREAD_COLOR)
    multichannel_grey = np.zeros((*img.shape, 3), np.uint8)
    allowed_channels = [1, 2]
    multichannel_grey[:, :, 0] = img
    multichannel_grey[:, :, 1] = img
    multichannel_grey[:, :, 2] = img

    for channel in allowed_channels:
        separated = multichannel_grey[:, :, channel]
        smoothed = gaussian_filter(separated, 3, 1)
        sharpened_edges = sharpen_edges(separated, laplacian(smoothed))
        sharpened_edges = motion_blur(sharpened_edges, 1, "horizontal")
        sharpened_edges = motion_blur(sharpened_edges, 1, "vertical")
        equalised = histogram_equalisation(sharpened_edges)
        equalised = motion_blur(equalised, 0, "horizontal")
        noise = bilateral_filter_greyscale(equalised, 2, 0.5, 0.5)
        output_channel = add_images(sharpened_edges, 0.6, noise, 0.4)
        multichannel_grey[:, :, channel] = output_channel

    concat = np.concatenate((coloured_image, multichannel_grey), axis=1)
    cv2.imshow("NEW Displayed Image", concat)
    cv2.waitKey(0)

#plt.show()
