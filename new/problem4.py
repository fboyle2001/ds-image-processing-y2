import cv2
import numpy as np
import os
import math
import sys

def swirl_image(img, swirl_radius, swirl_angle, swirl_direction, bilinear=True):
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

def low_pass_butterworth(img, n = 1, K = 15):
    # Fourier Transform of the image
    fft = np.fft.fft2(img)
    # Shift it so (0, 0) is in the centre
    fft = np.fft.fftshift(fft)
    img_width, img_height = img.shape
    # Calculate the coordinates of the centre
    centre_x, centre_y = img_width // 2, img_height // 2

    lpf = np.zeros((img_width, img_height), np.float64)

    for x in range(-2 * K, 2 * K):
        for y in range(-2 * K, 2 * K):
            dist = math.sqrt(x ** 2 + y ** 2)
            lpf[centre_x + x, centre_y + y] = 1 / (1 + np.power(dist / K, 2 * n))
    #np.set_printoptions(threshold=sys.maxsize)
    #print(lpf[centre_x-h:centre_x+h, centre_y-h:centre_y+h])
    #lpf[centre_x-30:centre_x+30, centre_y-30:centre_y+30] = 1
    fft *= lpf

    f_ishift = np.fft.ifftshift(fft)
    ifft = np.fft.ifft2(f_ishift)
    ifft = np.abs(ifft)
    return np.uint8(ifft)

img = cv2.imread("face1.jpg", cv2.IMREAD_GRAYSCALE)
lpf = low_pass_butterworth(img)
# swirl_direction = -1
# swirled = swirl_image(img, 170, math.pi / 2, swirl_direction, bilinear=True)
# reversed = swirl_image(swirled, 170, math.pi / 2, -swirl_direction, bilinear=True)
# diff = img - reversed

concat = np.concatenate((img, lpf), axis=1)
cv2.imshow("Displayed Image", concat)
cv2.waitKey(0)
