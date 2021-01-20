import cv2
import numpy as np
import os
import math

def apply_charcoal_effect(img, blur, blending_coefficient):
    mask = make_noise_texture(img)
    #img = gaussian_filter(img, *blur)
    img = blend_mask(img, mask, blending_coefficient)
    #img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

# Apply the motion blur effect
def gaussian_filter(img, n, sigma):
    distance = lambda x, y: math.sqrt((x - (n // 2)) ** 2 + (y - (n // 2)) ** 2)
    gaussian = lambda x, sigma: (1 / (sigma * (math.sqrt(2 * math.pi)))) * math.exp((-1 * (x ** 2)) / (2 * (sigma ** 2)))
    mask = np.array([[gaussian(distance(x, y), sigma) for y in range(n)] for x in range(n)], np.float64)
    mask = mask / np.sum(mask)
    filtered = cv2.filter2D(img, -1, mask)
    return filtered

def blend_mask(img, mask, blending_coefficient):
    img_height, img_width = img.shape
    mask_height, mask_width = mask.shape

    # Need to check height == height, width == width and fix otherwise
    for x in range(0, img_width):
        for y in range(0, img_height):
            mask_pixel = mask[x, y]
            o = img[x, y]
            img[x, y] = (1 - blending_coefficient) * img[x, y] + blending_coefficient * mask_pixel
            #print(o, mask_pixel, img[x, y])
            #print(mask_pixel)

    return img

def make_noise_texture(img):
    #parameters = 190, 18
    noise = cv2.bitwise_not(img)
    #noise = np.random.normal(*parameters, img.shape).astype(np.uint8)
    # MUST REPLACE STOLEN FROM https://stackoverflow.com/questions/40305933/how-to-add-motion-blur-to-numpy-array
    size=5
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    noise = cv2.filter2D(noise, -1, kernel_motion_blur)
    v=5
    kernel_motion_blur_v = np.zeros((v, v))
    kernel_motion_blur_v[:, int((v-1)/2)] = np.ones(v)
    kernel_motion_blur_v = kernel_motion_blur_v / v
    noise = cv2.filter2D(noise, -1, kernel_motion_blur_v)
    return noise

img = cv2.imread("face1.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
c = np.copy(img)

cv2.waitKey(0)

blending_coefficient = 0.2
mode = "monochrome"
new = None
inverted = make_noise_texture(img)

cv2.waitKey(0)
if mode == "monochrome":
    new = apply_charcoal_effect(img, (5, 1), blending_coefficient)
    #s = cv2.bitwise_not(img)
    #blend_mask(img, s, blending_coefficient)
    #img = cv2.GaussianBlur(img, (3, 3), 0)

concat = np.concatenate((c, inverted, new), axis=1)
cv2.namedWindow("Displayed Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Displayed Image", 1200, 400)
cv2.imshow("Displayed Image", concat)

cv2.waitKey(0)
