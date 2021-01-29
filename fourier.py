import cv2
import numpy as np
import os
import math
from matplotlib import pyplot as plt

img = cv2.imread("face1.jpg", cv2.IMREAD_COLOR)

def low_pass(channel):
    fourier = np.fft.fft2(channel)
    fshift = np.fft.fftshift(fourier)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    rows, cols = channel.shape
    crow,ccol = rows//2 , cols//2
    halfEdge = 100

    mask = np.zeros((rows,cols),np.uint8)
    mask[crow-halfEdge:crow+halfEdge, ccol-halfEdge:ccol+halfEdge] = 1

    fshift *= mask

    # fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return np.uint8(img_back)

cp = np.copy(img)

for channel in [0, 1, 2]:
    cp[:, :, channel] = low_pass(cp[:, :, channel])

concat = np.concatenate((img, cp), axis=1)
cv2.imshow("Displayed Image", concat)
cv2.waitKey(0)
