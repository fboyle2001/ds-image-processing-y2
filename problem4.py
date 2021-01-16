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

def swirl_image(img, angle):
    new_img = np.zeros(img.shape, np.uint8)
    centre_x, centre_y = img.shape[0] // 2, img.shape[1] // 2

    img_height, img_width, _ = img.shape
    s = set()

    for x in range(0, img_width):
        for y in range(0, img_height):
            remapped = to_polar_coordinates(y - centre_y, -(x - centre_x))

            if remapped == None:
                new_img[x, y] = img[x, y]
                continue

            swirl_pc = (math.log(2) * remapped[0] / 5, 0.5 * angle * math.exp(remapped[1]))
            new_cart = to_cartesian(*swirl_pc)
            new_cart = (math.ceil(remapped[0]) if remapped[0] < 0 else math.floor(remapped[0]) , math.ceil(remapped[1]) if remapped[1] < 0 else math.floor(remapped[1]))

            org_map_new_cart = (centre_y - new_cart[1], centre_x + new_cart[0])
            s.add(org_map_new_cart)
            print((x, y), (y - centre_y, -x + centre_x), new_cart, org_map_new_cart)
            new_img[org_map_new_cart[0], org_map_new_cart[1]] = img[x, y]

    print(len(s))

    return new_img

swirled = swirl_image(img, math.pi/2)

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
