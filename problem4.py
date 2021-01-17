import cv2
import numpy as np
import os
import math

img = cv2.imread("face2.jpg", cv2.IMREAD_COLOR)

def to_polar_coordinates(x, y):
    if x == 0 and y == 0:
        return None

    theta = math.atan2(y, x)
    r = math.sqrt(x ** 2 + y ** 2)

    return (r, theta)

def to_cartesian(r, theta):
    return (r * math.cos(theta), r * math.sin(theta))

def swirl_image(img):
    new_img = np.zeros(img.shape, np.uint8)

    img_height, img_width, _ = img.shape
    centre_x, centre_y = img_height // 2, img_width // 2
    sr = 100
    st = 1

    for x in range(0, img_width):
        for y in range(0, img_height):
            px = x - centre_x
            py = y - centre_y
            pd = math.sqrt(px ** 2 + py ** 2)
            pa = math.atan2(py, px)
            sa = 1 - (pd / sr)

            if sa > 0:
                ta = st * sa * math.pi * 2
                pa += ta
                px = round(math.cos(pa) * pd)
                py = round(math.sin(pa) * pd)

                new_img[x, y] = img[px + centre_x, py + centre_y]
            else:
                new_img[x, y] = img[x, y]


    return new_img

swirled = swirl_image(img)

# print(to_polar_coordinates(1, 1), to_cartesian(*to_polar_coordinates(1, 1)))
# print(to_polar_coordinates(-1, 1), to_cartesian(*to_polar_coordinates(-1, 1)))
# print(to_polar_coordinates(-1, -1), to_cartesian(*to_polar_coordinates(-1, -1)))
# print(to_polar_coordinates(1, -1), to_cartesian(*to_polar_coordinates(1, -1)))
# print(to_polar_coordinates(0, -1), to_cartesian(*to_polar_coordinates(0, -1)))
# print(to_polar_coordinates(0, 1), to_cartesian(*to_polar_coordinates(0, 1)))
# print(to_polar_coordinates(-1, 0), to_cartesian(*to_polar_coordinates(-1, 0)))
# print(to_polar_coordinates(1, 0), to_cartesian(*to_polar_coordinates(1, 0)))

concat = np.concatenate((img, swirled), axis=1)
cv2.namedWindow("Displayed Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Displayed Image", 800, 400)
cv2.imshow("Displayed Image", concat)
cv2.waitKey(0)
