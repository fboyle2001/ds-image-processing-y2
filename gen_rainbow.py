import cv2
import numpy as np
import os
import math

def gen_rainbow(width_range = None, hsv_max_hue_deg = 300, max_sat = 255, max_value = 255):
    new_img = np.zeros(img.shape, np.uint8)
    width = new_img.shape[1]

    if width_range == None:
        width_range = [0, width]

    max_width = abs(width_range[1] - width_range[0])
    width_offset = min(*width_range)

    for x in range(*width_range):
        hsv_deg = int((((x - width_offset) / max_width * hsv_max_hue_deg) / 360) * 179)
        new_img[:, x] = [hsv_deg, max_sat, max_value]

    return cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)

img = cv2.imread("face1.jpg", cv2.IMREAD_COLOR)
temp_2 = gen_rainbow()
template = gen_rainbow(width_range = [210, 280])

concat = np.concatenate((img, temp_2, template), axis=1)
cv2.namedWindow("Displayed Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Displayed Image", 1200, 400)
cv2.imshow("Displayed Image", concat)
cv2.waitKey(0)
