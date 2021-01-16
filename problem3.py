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

#print(np.log([]))

sf = 255 / math.log(255) * 0.75
#print(sf)
map = np.arange(1, 256)
map = (np.log(map) * sf).astype(np.uint8)
map = np.concatenate(([0], map))

#plt.plot([x for x in range(0, 256)], map, 'r', label = 'Map')
#plt.show()

#print(map)


# import sys
# sys.exit(0)

img = cv2.imread("face2.jpg", cv2.IMREAD_COLOR)
filtered = cv2.bilateralFilter(img, 5, 40, 40)
hsvf = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
#apply_colour_curve(hsvf, map)
apply_colour_curve_hsv(hsvf, map)
#apply_colour_hsv(hsvf, 1.2)
rgbf = cv2.cvtColor(hsvf, cv2.COLOR_HSV2BGR)

concat = np.concatenate((img, rgbf), axis=1)

cv2.imshow("Displayed Image", concat)
cv2.waitKey(0)
