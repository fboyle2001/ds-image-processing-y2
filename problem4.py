import cv2
import numpy as np
import os
import math

img = cv2.imread("face2.jpg", cv2.IMREAD_COLOR)

s = set()

def swirl_image(img, swirl_radius, swirl_angle, swirl_direction=-1):
    # Create an empty image of the same dimensions
    new_img = np.zeros(img.shape, np.uint8)

    # Calculate the center of the image
    img_height, img_width, _ = img.shape
    centre_x, centre_y = img_height // 2, img_width // 2

    # Loop over each pixel
    for x in range(0, img_width):
        for y in range(0, img_height):
            # Center the pixel's coordinates
            centred_x = x - centre_x
            centred_y = y - centre_y

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

            # Convert back to cartesian
            new_x = round(math.cos(theta) * r) - centre_x
            new_y = round(math.sin(theta) * r) - centre_y

            if (new_x, new_y) in s:
                print("!")

            s.add((new_x, new_y))

            # Map the pixels to the new image
            new_img[x, y] = img[new_x, new_y]

    return new_img

swirled = swirl_image(img, 150, math.pi / 2)

concat = np.concatenate((img, swirled), axis=1)
cv2.namedWindow("Displayed Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Displayed Image", 800, 400)
cv2.imshow("Displayed Image", concat)
cv2.waitKey(0)
