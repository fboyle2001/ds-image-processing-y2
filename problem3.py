import cv2
import numpy as np
import os
import math

import matplotlib.pyplot as plt

def apply_colour_curve(img, map):
    img_height, img_width, img_channels = img.shape

    for x in range(0, img_width):
        for y in range(0, img_height):
            img[x, y][0] = map[img[x, y][0]]
            img[x, y][1] = map[img[x, y][1]]
            img[x, y][2] = map[img[x, y][2]]

def apply_colour_hsv(hsvimg, sf):
    img_height, img_width, img_channels = hsvimg.shape

    for x in range(0, img_width):
        for y in range(0, img_height):
            r = hsvimg[x, y][1]
            hsvimg[x, y][1] *= sf
            n = hsvimg[x, y][1]

            if r > n:
                hsvimg[x, y][1] = 255

            # r = hsvimg[x, y][2]
            # hsvimg[x, y][2] += 10
            # n = hsvimg[x, y][2]
            #
            # if r > n:
            #     hsvimg[x, y][2] = 255

def apply_colour_curve_hsv(hsvimg, map):
    img_height, img_width, img_channels = hsvimg.shape

    for x in range(0, img_width):
        for y in range(0, img_height):
            hsvimg[x, y][1] = map[hsvimg[x, y][1]]

def bilateral_filter(img, size, sigma_colour, sigma_space):
    new_img = np.zeros(img.shape, np.uint8)
    img_height, img_width, _ = new_img.shape
    gaussian = lambda input, sigma: (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(input ** 2) / (2 * sigma ** 2))

    for x in range(0, img_width):
        for y in range(0, img_height):
            numer_b, numer_g, numer_r = 0, 0, 0
            denom_b, denom_g, denom_r = 0, 0, 0
            b, g, r = img[x, y]

            for n_x in range(max(x - size, 0), min(x + size + 1, img_width - 1)):
                for n_y in range(max(y - size, 0), min(y + size + 1, img_height - 1)):
                    nb, ng, nr = img[n_x, n_y].astype(dtype=int)

                    g_1 = gaussian((x - n_x) ** 2 + (y - n_y) ** 2, sigma_space)
                    g_2_b = gaussian(b - nb, sigma_colour)
                    g_2_g = gaussian(g - ng, sigma_colour)
                    g_2_r = gaussian(r - nr, sigma_colour)

                    numer_b += g_1 * g_2_b * nb
                    numer_g += g_1 * g_2_g * ng
                    numer_r += g_1 * g_2_r * nr

                    denom_b += g_1 * g_2_b
                    denom_g += g_1 * g_2_g
                    denom_r += g_1 * g_2_r

            # if numer_b > denom_b or numer_g > denom_g or numer_r > denom_r:
            #     print("?")

            new_img[x, y][0] = min(255, numer_b / denom_b)
            new_img[x, y][1] = min(255, numer_g / denom_g)
            new_img[x, y][2] = min(255, numer_r / denom_r)

    return new_img

#print(np.log([]))

sf = 255 / math.log(255) * 0.75
#print(sf)
map = np.arange(1, 256)
map = (np.log(map) * sf).astype(np.uint8)
map = np.concatenate(([0], map))

#plt.plot([x for x in range(0, 256)], map, 'r', label = 'Map')
#plt.show()

#print(map)



img = cv2.imread("face2.jpg", cv2.IMREAD_COLOR)
#filtered = cv2.bilateralFilter(img, 5, 40, 40)
filtered = bilateral_filter(img, 2, 15, 15)
#my_filtered = np.zeros(img.shape, np.uint8)

#concat = np.concatenate((img, filtered, my_filtered), axis=1)

#filtered = cv2.blur(img, (2, 2))
hsvf = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
#apply_colour_curve(hsvf, map)
#apply_colour_curve_hsv(hsvf, map)
apply_colour_hsv(hsvf, 1.5)
rgbf = cv2.cvtColor(hsvf, cv2.COLOR_HSV2BGR)

concat = np.concatenate((img, rgbf), axis=1)

cv2.imshow("Displayed Image", concat)
cv2.waitKey(0)
