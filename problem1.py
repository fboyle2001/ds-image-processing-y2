import cv2
import numpy as np
import os
import math

def blend_mask(img, mask, blending_coefficient):
    img_height, img_width, img_channels = img.shape
    mask_height, mask_width, mask_channels = mask.shape

    # Need to check height == height, width == width and fix otherwise
    for x in range(0, img_width):
        for y in range(0, img_height):
            mask_pixel = mask[x, y]

            img[x, y] = img[x, y] * (1 - darkening_coefficient)

            if not mask_pixel.any():
                continue

            img[x, y] = (1 - blending_coefficient) * img[x, y] + blending_coefficient * mask_pixel
            #print(mask_pixel)

img = cv2.imread("face1.jpg", cv2.IMREAD_COLOR)

# 0.5, 0.04 alright

darkening_coefficient = 0.5
blending_coefficient = 0.04
mode = "rainbow"

if mode == "rainbow":
    rainbow_mask = cv2.imread("./mask_generating/rainbow_pos.png")
    blend_mask(img, rainbow_mask, blending_coefficient)
    cv2.imshow("Displayed Image", img)

cv2.waitKey(0)
