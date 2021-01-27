import cv2
import numpy as np
import os
import math

def swirl_image(img, swirl_radius, swirl_angle, swirl_direction=-1, bilinear=True):
    # Create an empty image of the same dimensions
    new_img = np.zeros(img.shape, np.uint8)

    # Calculate the center of the image
    img_height, img_width, _ = img.shape
    centre_x, centre_y = img_height // 2, img_width // 2

    # Loop over each pixel
    for x in range(0, img_width):
        for y in range(0, img_height):
            # Center the pixel's coordinates
            centred_x = -(x - centre_x)
            centred_y = -(y - centre_y)

            # Convert to polar coordinates
            # atan2 handles quadrants
            r = math.sqrt(centred_x ** 2 + centred_y ** 2)
            theta = math.atan2(centred_y, centred_x)

            # Idea is that the further from the origin, the less we want to rotate the pixels
            # this will give the swirl effect

            # Only swirl the pixel if its less than the swirl radius
            # r >= 0 as we are square rooting
            if r <= swirl_radius:
                # Use a linear scale for the swirl, pixels near the center have the most swirl
                # swirl_direction can be used to switch between clockwise and anti-clockwise
                theta += swirl_angle * (1 - r / swirl_radius) * swirl_direction

                # Instead of rounding we should use interpolation
                # Rounding is just nearest neighbour?
                # Convert back to cartesian
                new_x = -(round(math.cos(theta) * r) - centre_x)
                new_y = -(round(math.sin(theta) * r) - centre_y)

                if bilinear:
                    # Bilinear interpolation
                    neighbourhood = img[max(new_x-1,0):min(new_x+1, img_width - 1), max(new_y-1, 0):min(new_y+1, img_width+1)]
                    bilinear_avg = np.average(neighbourhood, axis=(1, 0)).astype(dtype=np.uint8)

                    # Map the pixels to the new image
                    new_img[x, y] = bilinear_avg
                else:
                    new_img[x, y] = img[new_x, new_y]
            else:
                new_img[x, y] = img[x, y]

    return new_img

def inverse_swirl(img, swirl_radius, swirl_angle, swirl_direction=-1):
    # Create an empty image of the same dimensions
    new_img = np.zeros(img.shape, np.uint8)

    # Calculate the center of the image
    img_height, img_width, _ = img.shape
    centre_x, centre_y = img_height // 2, img_width // 2

    # Loop over each pixel
    for x in range(0, img_width):
        for y in range(0, img_height):
            # Center the pixel's coordinates
            centred_x = -(x - centre_x)
            centred_y = -(y - centre_y)

            # Convert to polar coordinates
            # atan2 handles quadrants
            r = math.sqrt(centred_x ** 2 + centred_y ** 2)
            theta = math.atan2(centred_y, centred_x)

            # Idea is that the further from the origin, the less we want to rotate the pixels
            # this will give the swirl effect

            # Only swirl the pixel if its less than the swirl radius
            # r >= 0 as we are square rooting
            if r <= swirl_radius:
                # Use a linear scale for the swirl, pixels near the center have the most swirl
                # swirl_direction can be used to switch between clockwise and anti-clockwise
                theta += swirl_angle * (1 - r / swirl_radius) * -swirl_direction

            # Convert back to cartesian
            new_x = -(round(math.cos(theta) * r) - centre_x)
            new_y = -(round(math.sin(theta) * r) - centre_y)

            # Map the pixels to the new image
            new_img[x, y] = img[new_x, new_y]

    #print(len(expect - s))

    return new_img

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

img = cv2.imread("me.png", cv2.IMREAD_COLOR)
img = gaussian_filter(img, 2, 0.5)
swirled = swirl_image(img, 170, math.pi / 2, bilinear=True)
reversed = inverse_swirl(swirled, 170, math.pi / 2)
diff = img - reversed

concat = np.concatenate((img, swirled, reversed, diff), axis=1)
cv2.namedWindow("Displayed Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Displayed Image", 1600, 400)
cv2.imshow("Displayed Image", concat)
cv2.waitKey(0)
