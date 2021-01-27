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

# In my experimentation, this was 3x faster than my initial implementation
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

def apply_colour_curve(img, channel_maps, hsv = True):
    new_img = np.copy(img)

    if hsv:
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)

    img_height, img_width, img_channels = img.shape

    for x in range(img_width):
        for y in range(img_height):
            for channel in range(img_channels):
                if channel == 1:
                    r = new_img[x, y][1]
                    new_img[x, y][1] *= 1.5
                    n = new_img[x, y][1]

                    if r > n:
                        new_img[x, y][1] = 255

    if hsv:
        new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)

    return new_img

## NOT MINE MUST REPLACE
def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[1],channels[1])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_HSV2BGR,img)
    return img

channel_maps = [
    lambda h: h,
    lambda s: s * 1.2 if s > s * 1.2 else 255,
    lambda v: v
]

for i in range(3):
    print(i, 1, channel_maps[i](1))
img = cv2.imread("face1.jpg", cv2.IMREAD_COLOR)
#img = hisEqulColor(img)
f = bilateral_filter(img, 2, 15, 15)


c = apply_colour_curve(f, channel_maps, hsv=True)
d = img - c

concat = np.concatenate((img, f, c, d), axis=1)

cv2.imshow("FAST Displayed Image", concat)
cv2.waitKey(0)
