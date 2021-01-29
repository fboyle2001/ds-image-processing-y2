import cv2
import numpy as np
import os
import math

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

def bilateral_filter(img, n, sigma_colour, sigma_space):
    result = np.zeros(img.shape, np.uint8)
    space_mask = generate_spatial_gaussian_mask(n, sigma_space)

    width = (0, img.shape[0])
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
                colour_mask = generate_intensity_gaussian_mask(slide, n, sigma_colour)
                dual_mask = space_mask * colour_mask
                dual_mask = dual_mask
                # print(dual_mask, np.sum(dual_mask))
                # Multiple the mask and slide and then sum the matrix elements
                # This is the Gaussian blur at the pixel's b/g/r channel
                result[x - n, y, channel] = np.uint8(np.sum(dual_mask * slide) / np.sum(dual_mask))

    return result

def compute_polynomial_single_channel(points, end):
    def fitter(x, points):
        s = 0

        for i, point in enumerate(points):
            product = point[1]

            for j, other in enumerate(points):
                if i == j:
                    continue
                product *= (x - other[0]) / (point[0] - other[0])

            s += product

        return s

    return np.array([fitter(x, points) for x in range(end)], dtype=np.uint8)

def generate_hsv_lut():
    # I decided to use HSV instead of BGR to for my colour curves
    # this was because it allows targeting of colours that would create the beautifying
    # effect without distorting the image's contrast too badly
    # I have not applied a curve to the hue since this would simply change the colour
    # which would have an adverse effect
    hue = [x for x in range(180)]
    # I have used Lagrange Interpolating Polynomials to calculate the colour curves
    # these are polynomials of degree n-1 (given n points) that pass through the specified points
    # essentially allowed me to 'drag' the curve to where I wanted
    # The saturation curve is higher than the value curve
    saturation = compute_polynomial_single_channel([(0, 0), (128, 150), (255, 255)], 256)
    value = compute_polynomial_single_channel([(0, 0), (128, 135), (255, 255)], 256)

    return [hue, saturation, value]

def apply_hsv_lut(img, lut):
    copy = np.copy(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    img_height, img_width, img_channels = img.shape

    for x in range(img_width):
        for y in range(img_height):
            h, s, v = copy[y, x]
            nh, ns, nv = lut[0][h], lut[1][s], lut[2][v]
            copy[y, x] = [nh, ns, nv]

    return cv2.cvtColor(copy, cv2.COLOR_HSV2BGR)

img = cv2.imread("face1.jpg", cv2.IMREAD_COLOR)
lut = generate_hsv_lut()

from matplotlib import pyplot as plt
# plt.plot([x for x in range(180)], lut[0])
# plt.plot([x for x in range(180)], [x for x in range(180)])
# plt.show()
# plt.plot([x for x in range(256)], lut[1])
# plt.plot([x for x in range(256)], [x for x in range(256)])
# plt.show()
# plt.plot([x for x in range(256)], lut[2])
# plt.plot([x for x in range(256)], [x for x in range(256)])
# plt.show()
f = bilateral_filter(img, 2, 6, 6)

out = apply_hsv_lut(f, lut)


#c = apply_colour_curve(img, [], hsv=True)
#d = img - c

concat = np.concatenate((img, f, out), axis=1)

cv2.imshow("FAST Displayed Image", concat)
cv2.waitKey(0)
