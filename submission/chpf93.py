import cv2
import numpy as np
import math
import argparse
import os
import time

"""
Creates the Gaussian Mask to slide over the image
Required Parameters:
n - Used to create a (2 * n + 1, 2 * n + 1) sliding mask, sigma - s.d. used in the Gaussian
"""
def generate_gaussian_mask(n, sigma):
    # The Gaussian function as a lambda
    gaussian = lambda input, sigma: (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(input ** 2) / (2 * sigma ** 2))
    # Will store the masks rows and columns
    mask = []
    # The total of all of the gaussian(r, sigma) to normalise
    total = 0

    # Create a (2 * n + 1) * (2 * n + 1) mask
    for x in range(0, 2 * n + 1):
        row = []

        for y in range(0, 2 * n + 1):
            # Distance from centre
            r = math.sqrt((x - n) ** 2 + (y - n) ** 2)
            # Value of the gaussian at a point
            value = gaussian(r, sigma)
            total += value
            row.append(value)

        mask.append(row)

    # Normalise the result and convert it to a numpy array
    normalised = np.array([[element / total for element in row] for row in mask])
    return normalised

"""
Implementation of the Gaussian Filter
Required Parameters:
img - Source image, n - Used to create a (2 * n + 1, 2 * n + 1) sliding mask, sigma - s.d. used in the Gaussian
Optional Parameters:
width - apply to fixed vertical slice, height - apply to fixed horizontal slice,
separate - only returns the affected slice if this is True
"""
def gaussian_filter(img, n, sigma, width = None, height = None, separate = False):
    result = None

    # Deals with the optional parameters
    if separate:
        result = np.zeros(img.shape, np.uint8)
    else:
        result = np.copy(img)

    mask = generate_gaussian_mask(n, sigma)

    if width == None:
        width = (0, img.shape[0])

    if height == None:
        height = (0, img.shape[1])

    # The channels are stored in the shape
    channels = img.shape[2] if len(img.shape) == 3 else 1

    # Apply separately to each channel
    for channel in range(channels):
        # Pad either side so we can slide easily
        arr = None

        if channels != 1:
            arr = np.pad(img[:, :, channel], ((n, n), (n, n)), "edge")
        else:
            arr = np.pad(img[:, :], ((n, n), (n, n)), "edge")
        # Loop over each original value but with the necessary offset
        for x in range(width[0] + n, width[1] + n):
            for y in range(height[0], height[1]):
                # Select the neighbourhood of the pixel
                slide = arr[x - n : x + n + 1, y : y + 2 * n + 1]
                # Multiple the mask and slide and then sum the matrix elements
                # This is the Gaussian blur at the pixel's b/g/r channel
                if channels != 1:
                    result[x - n, y, channel] = np.sum(mask * slide).astype(np.uint8)
                else:
                    result[x - n, y] = np.sum(mask * slide).astype(np.uint8)

    return result

"""
Darkens an image by converting to HSV and multiplying the value by a scale factor
coefficient < 1 darkens, coefficient > 1 brightens
"""
def darken_image(img, coefficient):
    darkened = np.copy(img)
    darkened = cv2.cvtColor(darkened, cv2.COLOR_BGR2HSV)

    # Loop each pixel and update the value
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            darkened[x, y, 2] = np.uint8(darkened[x, y, 2] * coefficient)

    return cv2.cvtColor(darkened, cv2.COLOR_HSV2BGR)

"""
Applies the light leak effect to an image
left, right, upper and bottom are lambda functions that define the region
"""
def light_leak(img, left, right, upper, bottom):
    mask = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2HSV)
    img_height, img_width, img_channels = mask.shape

    half = img_height // 2

    # Loop over each pixel
    for x in range(img_width):
        for y in range(img_height):
            # To restrict to a specific region we use the passed lambda functions
            if upper(x) <= y <= bottom(x) and left(y) <= x <= right(y):
                # Polynomial peaks at img.height // 2
                # Gives a smooth fade to make it seem like light leaking
                factor = ((-1 / half) * (y - half) ** 2 + half) / half

                mask[y, x, 2] = np.uint8(mask[y, x, 2] * factor)
                continue

            # If it is outside the region just set it to black
            mask[y, x] = [0, 0, 0]

    return cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

"""
Applies the rainbow light leak effect to an image
left, right, upper and bottom are lambda functions that define the region
hsv_max_hue_deg, max_sat, max_value control the rainbow effect
these are the defaults and limits for the HSV computations
"""
def rainbow_light_leak(img, left, right, upper, bottom, hsv_max_hue_deg = 300, max_sat = 255, max_value = 128):
    mask = cv2.cvtColor(np.copy(img), cv2.COLOR_BGR2HSV)
    img_height, img_width, img_channels = mask.shape

    # Want to capture the full rainbow in the region
    # so we want to spread it over the width of the region
    width_start = img_width
    width_end = 0
    half = img_height // 2

    # Calculates the width of the region in advance
    for x in range(img_width):
        for y in range(img_height):
            if left(y) <= x <= right(y) and upper(x) <= y <= bottom(x):
                if x < width_start:
                    width_start = x

                if x > width_end:
                    width_end = x

    max_width = width_end - width_start

    # Loop each pixel
    for x in range(img_width):
        for y in range(img_height):
            # Only apply the effect if it is in the region
            if upper(x) <= y <= bottom(x) and left(y) <= x <= right(y):
                # Calculate the colour
                # this works by scaling over the width of the region to get a blend
                # of the colours like a rainbow
                hsv_deg = int((((x - width_start) / max_width * hsv_max_hue_deg) / 280) * 179)
                # Then apply the same polynomial as the light leak mask to get the fade effect
                factor = ((-1 / half) * (y - half) ** 2 + half) / half
                mask[y, x] = np.uint8(np.clip([hsv_deg, max_sat, mask[y, x, 2] * factor], 0, 255))
                continue

            # If it is outside the region set it to black
            mask[y, x] = [0, 0, 0]

    return cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

"""
Replicates cv2.addWeighted
alpha and beta do not have to sum to 1
"""
def add_images(img, alpha, secondary, beta, gamma = 0):
    # Create a blank image of the same shape
    added = np.zeros(img.shape, np.uint8)

    img_height, img_width = img.shape[:2]

    # Loop each pixel and add them according to the weights
    for x in range(img_width):
        for y in range(img_height):
            # Clip values in case they exceed 255
            added[y, x] = np.uint8(np.clip(img[y, x] * alpha + secondary[y, x] * beta + gamma, 0, 255))

    return added

"""
Applies the Laplacian to the image to isolate the edges
"""
def laplacian(img):
    copy = np.copy(img)
    # The mask from the lectures
    mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # Convolve
    copy = cv2.filter2D(copy, -1, mask)
    return copy

"""
Sharpen the edges of the image
"""
def sharpen_edges(img, edges):
    # Just use absdiff as it is a basic operation
    return cv2.absdiff(img, edges)

"""
Histogram equalisation of a greyscale image
"""
def histogram_equalisation(img):
    img_height, img_width = img.shape
    # Initialise the bins
    bins = [0 for i in range(256)]

    # Fill the bins
    for x in range(img_width):
        for y in range(img_height):
            bins[img[y, x]] += 1

    # Calculate the cumulative sum of the histogram
    cumulative = [sum(bins[:i]) for i in range(256)]

    # New image to store the equalised image
    equalised = np.zeros(img.shape, np.uint8)
    # We need to divide by maximum and then scale to 0-255
    scale_factor = 255 / max(cumulative)

    for x in range(img_width):
        for y in range(img_height):
            # Scale the image based on the bins
            equalised[y, x] = np.uint8(cumulative[img[y, x]] * scale_factor)

    return equalised

"""
Applies a motion blur effect to the image
"""
def motion_blur(img, n, direction):
    copy = np.copy(img)
    size = 2 * n + 1
    # Create an empty (2 * n + 1) x (2 * n + 1) matrix
    matrix = np.zeros((size, size))

    if direction == "vertical":
        # For vertical we want to fill the centre column, this is just n
        for x in range(size):
            matrix[x, n] = 1
    else:
        # Default is horizontal
        # For vertical we want to fill the centre row, this is just n
        for y in range(size):
            matrix[n, y] = 1

    # Now normalise the matrix
    matrix /= size

    # Convolve the image with the matrix
    copy = cv2.filter2D(copy, -1, matrix)
    return copy

"""
Creates the Gaussian Mask with respect to the size of the neighbourhood
Required Parameters:
n - Used to create a (2 * n + 1, 2 * n + 1) sliding mask, sigma - s.d. used in the Gaussian
"""
def generate_spatial_gaussian_mask(n, sigma):
    # The Gaussian function as a lambda
    gaussian = lambda input, sigma: (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(input ** 2) / (2 * sigma ** 2))
    # Will store the masks rows and columns
    mask = []
    # The total of all of the gaussian(r, sigma) to normalise
    total = 0

    # Create a (2 * n + 1) * (2 * n + 1) mask
    for x in range(0, 2 * n + 1):
        row = []

        for y in range(0, 2 * n + 1):
            # Distance from centre
            r = math.sqrt((x - n) ** 2 + (y - n) ** 2)
            # Value of the gaussian at a point
            value = gaussian(r, sigma)
            total += value
            row.append(value)

        mask.append(row)

    return np.array(mask)

"""
Creates the Gaussian Mask with respect to the intensity of the neighbourhood
Required Parameters:
n - Used to create a (2 * n + 1, 2 * n + 1) sliding mask, sigma - s.d. used in the Gaussian
"""
def generate_intensity_gaussian_mask(slide, n, sigma):
    # The Gaussian function as a lambda
    gaussian = lambda input, sigma: (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-(input ** 2) / (2 * sigma ** 2))
    # Will store the masks rows and columns
    mask = []
    # The total of all of the gaussian(r, sigma) to normalise
    total = 0

    # Create a (2 * n + 1) * (2 * n + 1) mask
    for x in range(0, 2 * n + 1):
        row = []

        for y in range(0, 2 * n + 1):
            # Distance from centre
            r = abs(int(slide[x, y]) - int(slide[n, n]))
            #print(r, slide[x, y], slide[n, n], x, y, n)
            # Value of the gaussian at a point
            value = gaussian(r, sigma)
            total += value
            row.append(value)

        mask.append(row)

    return np.array(mask)

"""
Applies a Bilateral Filter to a greyscale image
Required Parameters:
n - Used to create a (2 * n + 1, 2 * n + 1) sliding mask
sigma_colour, sigma_space - the s.d. to be used in the respective Gaussians
"""
def bilateral_filter_greyscale(img, n, sigma_colour, sigma_space):
    result = np.zeros(img.shape, np.uint8)
    space_mask = generate_spatial_gaussian_mask(n, sigma_space)

    width = (0, img.shape[0])
    height = (0, img.shape[1])

    # Pad either side so we can slide easily
    arr = np.pad(img[:, :], ((n, n), (n, n)), "edge")
    # Loop over each original value but with the necessary offset
    for x in range(width[0] + n, width[1] + n):
        for y in range(height[0], height[1]):
            # Select the neighbourhood of the pixel
            slide = arr[x - n : x + n + 1, y : y + 2 * n + 1]
            colour_mask = generate_intensity_gaussian_mask(slide, n, sigma_colour)
            dual_mask = space_mask * colour_mask
            # print(dual_mask, np.sum(dual_mask))
            # Multiple the mask and slide and then sum the normalised matrix elements
            # This is the Gaussian blur at the pixel
            result[x - n, y] = np.uint8(np.sum(dual_mask * slide) / np.sum(dual_mask))

    return result

"""
Applies a Bilateral Filter to a colour image
Required Parameters:
n - Used to create a (2 * n + 1, 2 * n + 1) sliding mask
sigma_colour, sigma_space - the s.d. to be used in the respective Gaussians
"""
def bilateral_filter(img, n, sigma_colour, sigma_space):
    result = np.zeros(img.shape, np.uint8)
    space_mask = generate_spatial_gaussian_mask(n, sigma_space)

    width = (0, img.shape[0])
    height = (0, img.shape[1])

    # The channels are stored in the shape
    channels = img.shape[2]

    # Apply separately to each channel
    for channel in range(channels):
        # Pad either side so we can slide easily
        arr = np.pad(img[:, :, channel], ((n, n), (n, n)), "edge")
        # Loop over each original value but with the necessary offset
        for x in range(width[0] + n, width[1] + n):
            for y in range(height[0], height[1]):
                # Select the neighbourhood of the pixel
                slide = arr[x - n : x + n + 1, y : y + 2 * n + 1]
                colour_mask = generate_intensity_gaussian_mask(slide, n, sigma_colour)
                dual_mask = space_mask * colour_mask
                dual_mask = dual_mask
                # print(dual_mask, np.sum(dual_mask))
                # Multiple the mask and slide and then sum the matrix elements
                # This is the Gaussian blur at the pixel's b/g/r channel
                result[x - n, y, channel] = np.uint8(np.sum(dual_mask * slide) / np.sum(dual_mask))

    return result

"""
Uses Lagrange Interpolating Polynomial to compute a polynomial between the specified points
the polynomial is of degree n-1 given n points, no restrictions on the x coordinates
I use this to calculate the colour curves
"""
def compute_polynomial_single_channel(points, end):
    # Fits a single x coordinate
    def fitter(x, points):
        y = 0

        # Loop over each specified point
        for i, point in enumerate(points):
            # Start with the y value of the point
            product = point[1]

            # Multiply by (x_0 - p_x) / (y_0 - p_y)
            for j, other in enumerate(points):
                if i == j:
                    continue
                product *= (x - other[0]) / (point[0] - other[0])

            # Sum up for each point
            y += product

        return y

    # Round to np.uint8
    return np.array([fitter(x, points) for x in range(end)], dtype=np.uint8)

"""
Generates my HSV Lookup Table
"""
def generate_hsv_lut():
    # I decided to use HSV instead of BGR to for my colour curves
    # this was because it allows targeting of colours that would create the beautifying
    # effect without distorting the image's contrast too badly
    # I have not applied a curve to the hue since this would simply change the colour
    # which would have an adverse effect
    hue = [x for x in range(180)]
    # I have used Lagrange Interpolating Polynomials to calculate the colour curves
    # these are polynomials of degree n-1 (given n points) that pass through the specified points
    # essentially allowed me to 'drag' the curve to where I wanted
    # The saturation curve is higher than the value curve
    saturation = compute_polynomial_single_channel([(0, 0), (66, 100), (128, 160), (198, 216), (255, 255)], 256)
    value = compute_polynomial_single_channel([(0, 0), (128, 135), (255, 255)], 256)

    return [hue, saturation, value]

"""
Applies a Lookup Table to a HSV image
"""
def apply_hsv_lut(img, lut):
    copy = np.copy(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    img_height, img_width, img_channels = img.shape

    # Loop each pixel and lookup the values
    # then set them on the copy of the image
    for x in range(img_width):
        for y in range(img_height):
            h, s, v = copy[y, x]
            nh, ns, nv = lut[0][h], lut[1][s], lut[2][v]
            copy[y, x] = [nh, ns, nv]

    # Convert back to BGR for display
    return cv2.cvtColor(copy, cv2.COLOR_HSV2BGR)

"""
Swirls an image by converting to polar coordinates
swirl_radius is how far from the centre the swirl should affect
swirl_angle is the maximum rotation of any pixel (scaled by distance from centre relative to the swirl_radius)
swirl_direction is which way to apply the angle
bilinear defaults to True and applies bilinear interpolation. False will apply nearest neighbour interpolation
"""
def swirl_image(img, swirl_radius, swirl_angle, swirl_direction, bilinear = True):
    swirl_direction *= -1
    # Create an empty image of the same dimensions
    new_img = np.zeros(img.shape, np.uint8)

    # Calculate the center of the image
    img_height, img_width, _ = img.shape
    centre_x, centre_y = img_width // 2, img_height // 2

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

                if not(0 <= new_y < img_width):
                    new_y = np.clip(new_y, 0, img_width - 1)

                if not(0 <= new_x < img_height):
                    new_x = np.clip(new_x, 0, img_height - 1)

                if bilinear:
                    # Bilinear interpolation
                    neighbourhood = img[max(new_y-1, 0):min(new_y+1, img_width+1), max(new_x-1,0):min(new_x+1, img_width - 1)]
                    bilinear_avg = np.average(neighbourhood, axis=(1, 0)).astype(dtype=np.uint8)

                    # Map the pixels to the new image
                    new_img[y, x] = bilinear_avg
                else:
                    new_img[y, x] = img[new_y, new_x]
            else:
                new_img[y, x] = img[y, x]

    return new_img

"""
Applies a Low Pass Butterworth Filter to the Fourier Transform of a channel of an image
This is the pre-filtering for part 2 of the problem 4
n is the order of the filter and K is the decay threshold
"""
def low_pass_butterworth(channel, n, K):
    # Fourier Transform of the image
    fft = np.fft.fft2(channel)
    # Shift it so (0, 0) is in the centre
    fft = np.fft.fftshift(fft)
    img_width, img_height = channel.shape
    # Calculate the coordinates of the centre
    centre_x, centre_y = img_width // 2, img_height // 2

    lpf = np.zeros((img_width, img_height), np.float64)

    for x in range(-centre_x, centre_x):
        for y in range(-centre_y, centre_y):
            dist = math.sqrt(x ** 2 + y ** 2)
            lpf[centre_x + x, centre_y + y] = 1 / (1 + np.power(dist / K, 2 * n))

    # Convolution is the same as multiplication on freq domain
    fft *= lpf

    # Reverse the centre shift
    f_ishift = np.fft.ifftshift(fft)
    # Reverse the fourier transform
    ifft = np.fft.ifft2(f_ishift)
    # Take abs as the values could be complex
    ifft = np.abs(ifft)
    ifft = np.clip(ifft, 0, 255)
    return np.uint8(ifft)

"""
Applies a Low Pass Butterworth Filter to each channel of a colour image
n is the order of the filter and K is the decay threshold
"""
def lpb_colour(img, n, K):
    b, g, r = cv2.split(img)

    bl = low_pass_butterworth(b, n, K)
    gl = low_pass_butterworth(g, n, K)
    rl = low_pass_butterworth(r, n, K)

    return cv2.merge((bl, gl, rl))

"""
Displays multiple images side-by-side
"""
def display_images(images):
    title = [image[0] for image in images]
    concat = np.concatenate([image[1] for image in images], axis = 1)

    cv2.imshow("|".join(title), concat)
    cv2.waitKey(0)

"""
Problem 1: Light Leak
mode defaults to simple but can be set to rainbow, darkening_coefficient is the scale factor for darkening (<1 to darken, >1 to brighten)
alpha_blend is how much of the darkened image to include, beta_blend is how much of the light leak to include
gaussian_n and gaussian_sigma are optional parameters that control the blur applied to the light leak to smooth the edges
Recommended parameters:
problem1(img, "simple", 0.4, 1, 0.7)
problem1(img, "rainbow", 0.5, 1, 0.5) *TODO*
"""
def problem1(img, mode, darkening_coefficient, alpha_blend, beta_blend, gaussian_n = 4, gaussian_sigma = 2.2):
    # Defines the boundaries of where to apply the light effect
    upper = lambda x: 0
    bottom = lambda x: img.shape[0]
    left_c = int(0.6375 * img.shape[1])
    right_c = int(0.6875 * img.shape[1])
    left = lambda y: -0.1 * y + left_c
    right = lambda y: -0.1 * y + right_c

    # Darken the image
    darkened = darken_image(img, darkening_coefficient)
    mask = None

    # Switch mask depending on mode
    if mode.lower() == "rainbow":
        mask = rainbow_light_leak(img, left, right, upper, bottom)
    else:
        # Default to simple mode
        mask = light_leak(img, left, right, upper, bottom)

    # Combine the light leak and the darkened image
    blurred_mask = gaussian_filter(mask, gaussian_n, gaussian_sigma)
    weighted = add_images(darkened, alpha_blend, blurred_mask, beta_blend)
    return weighted

"""
Problem 2: Pencil/Charcoal Effect
mode defaults to regular pencil but can be set to 'coloured pencil', blending_coefficient is the amount of the source to include
uses 1 - blending_coefficient for the noise
channels is used to define which channels in the BGR image to apply the effect on
defaults to B, G channels
Recommended parameters:
problem2(img, "regular", 0.6)
problem2(img, "coloured_pencil", 0.6, channels = [0, 1])
"""
def problem2(img, mode, blending_coefficient, channels = [0, 1]):
    # Convert the image to greyscale
    greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output = None

    if mode.lower() == "coloured_pencil":
        # Create the same structure as a coloured image but set all the
        # channels to be the greyscale image
        multichannel_grey = np.zeros((*greyscale.shape, 3), np.uint8)
        multichannel_grey[:, :, 0] = greyscale
        multichannel_grey[:, :, 1] = greyscale
        multichannel_grey[:, :, 2] = greyscale

        for i, channel in enumerate(channels):
            if channel < 0 or channel > 2:
                continue

            separated = multichannel_grey[:, :, channel]
            # Smooth the image so the laplacian isn't influenced by noise
            smoothed = gaussian_filter(separated, 3 + i, 1)
            # Sharpen the edges with a laplacian
            # Apply some motion blur to get a pencil like effect
            sharpened_edges = sharpen_edges(separated, laplacian(smoothed))
            sharpened_edges = motion_blur(sharpened_edges, i, "horizontal")
            sharpened_edges = motion_blur(sharpened_edges, i, "vertical")
            # Apply histogram equalisation to give a 'shaded' effect
            # Makes it look like a pencil drawing
            equalised = histogram_equalisation(sharpened_edges)
            equalised = motion_blur(equalised, 0, "horizontal")
            # Finally apply a bilateral filter to smooth out the roughest noise
            # generated by the laplacian and equalisation
            noise = bilateral_filter_greyscale(equalised, 2, 0.5, 0.5)
            # Add the sharpened image and the noise
            output_channel = add_images(sharpened_edges, blending_coefficient, noise, 1 - blending_coefficient)
            # Update the channel
            multichannel_grey[:, :, channel] = np.uint8(np.clip(output_channel, 0, 255))

        output = multichannel_grey
    else:
        # Default mode is regular pencil
        smoothed = gaussian_filter(greyscale, 3, 1)
        # Sharpen the edges with a laplacian
        # Apply some motion blur to get a pencil like effect
        sharpened_edges = sharpen_edges(greyscale, laplacian(smoothed))
        sharpened_edges = motion_blur(sharpened_edges, 1, "horizontal")
        sharpened_edges = motion_blur(sharpened_edges, 1, "vertical")
        # Apply histogram equalisation to give a 'shaded' effect
        # Makes it look like a pencil drawing
        equalised = histogram_equalisation(sharpened_edges)
        equalised = motion_blur(equalised, 0, "horizontal")
        # Finally apply a bilateral filter to smooth out the roughest noise
        # generated by the laplacian and equalisation
        noise = bilateral_filter_greyscale(equalised, 2, 0.5, 0.5)
        # Blend the image
        output = add_images(sharpened_edges, blending_coefficient, noise, 1 - blending_coefficient)

    return output

"""
Problem 3: Smoothing and Beautifying Filtering
n is used to determine the size of the neighbourhood for the Bilateral Filter (2 * n + 1 x 2 * n + 1)
sigma_colour and sigma_space are the s.d. used in the respective Gaussians in the bilateral
Recommended parameters:
problem3(img, 2, 6, 6)
"""
def problem3(img, n, sigma_colour, sigma_space):
    # Generate the lookup table for the colour mapping
    lookup_table = generate_hsv_lut()
    # Apply a Bilateral Filter
    # This is useful for preserving the edges while smoothing the skin of the person
    # This creates the 'Instagram Beautification' effect
    filtered = bilateral_filter(img, n, sigma_colour, sigma_space)
    # Apply the LUT to the image's pixels
    # Increases saturation and brightness
    output = apply_hsv_lut(filtered, lookup_table)

    return output

"""
Problem 4: Face Swirl
swirl_radius is how far from the centre the swirl should affect
swirl_angle is the maximum rotation of any pixel (scaled by distance from centre relative to the swirl_radius)
swirl_direction is which way to apply the angle
bilinear defaults to True and applies bilinear interpolation. False will apply nearest neighbour interpolation
Recommended parameters:
problem4(img, 170, math.pi / 2)
"""
def problem4(img, swirl_radius, swirl_angle, swirl_direction = -1, bilinear = True, n = 1, K = 40):
    # Part A
    normal = [("Swirled", problem4_normal(img, swirl_radius, swirl_angle, swirl_direction, bilinear))]

    # Part B
    blur = problem4_blur_demonstration(img, swirl_radius, swirl_angle, swirl_direction, bilinear, n, K)

    # Part C
    inverse = problem4_inverse_demonstration(img, swirl_radius, swirl_angle, swirl_direction, bilinear)

    return normal + blur[1:] + inverse[1:]

"""
Regular swirl
swirl_radius is how far from the centre the swirl should affect
swirl_angle is the maximum rotation of any pixel (scaled by distance from centre relative to the swirl_radius)
swirl_direction is which way to apply the angle
bilinear defaults to True and applies bilinear interpolation. False will apply nearest neighbour interpolation
Recommended parameters:
problem4(img, 170, math.pi / 2)
"""
def problem4_normal(img, swirl_radius, swirl_angle, swirl_direction, bilinear):
    swirled = swirl_image(img, swirl_radius, swirl_angle, swirl_direction, bilinear)
    return swirled

"""
Showcases the effect of prefiltering the image for problem 4
Recommended running:
display_images(problem4_blur_demonstration(img, 170, math.pi / 2))
"""
def problem4_blur_demonstration(img, swirl_radius, swirl_angle, swirl_direction = -1, bilinear = True, n = 1, K = 40):
    # Prefilter the image using Low Pass Butterworth
    prefiltered = lpb_colour(img, n, K)
    # Swirl the regular image
    regular_swirled = swirl_image(img, swirl_radius, swirl_angle, swirl_direction, bilinear)
    # Swirl the prefiltered image
    prefiltered_swirled = swirl_image(prefiltered, swirl_radius, swirl_angle, swirl_direction, bilinear)

    return [("No Filtering", regular_swirled), ("Filtered without Swirl", prefiltered), ("Low Pass Butterworth Filtering", prefiltered_swirled)]

"""
Showcases the effect of reversing the swirl
Recommended running:
display_images(problem4_inverse_demonstration(img, 170, math.pi / 2))
"""
def problem4_inverse_demonstration(img, swirl_radius, swirl_angle, swirl_direction = -1, bilinear = False):
    # Swirl the regular image
    swirled = swirl_image(img, swirl_radius, swirl_angle, swirl_direction, bilinear)
    # Reverse the swirl
    reversed = swirl_image(swirled, swirl_radius, swirl_angle, -swirl_direction, bilinear)
    # Consider their difference
    difference = reversed - img

    return [("Swirled", swirled), ("Reversed", reversed), ("Reversed - Original", difference)]

"""
Used to parse the command line arguments
"""
def main():
    parser = argparse.ArgumentParser(description = "chpf93 Image Processing Coursework")
    requiredGroup = parser.add_argument_group("required")

    # Input file
    requiredGroup.add_argument("--inputFile", "-i", dest="inputFile", action="store", required = True)
    # Set the problem
    requiredGroup.add_argument("--problem", "-p", dest="problem", action="store", required = True)
    # Output file, if not set then it displays it instead
    parser.add_argument("--outputFile", "-o", dest="outputFile", action="store")

    # Problem 1 arguments
    parser.add_argument("--mode", "-m", dest="mode", action="store")
    parser.add_argument("--darkening-coefficient", "-d", dest="darkening_coefficient", action="store")
    parser.add_argument("--alpha-blend", "-b", dest="alpha_blend", action="store")
    parser.add_argument("--beta-blend", "-bb", dest="beta_blend", action="store")
    parser.add_argument("--neighbourhood-size", "-n", dest="neighbourhood_size", action="store")
    parser.add_argument("--gaussian-sigma", "-s", dest="gaussian_sigma", action="store")

    # Problem 2 arguments
    # Reuse mode from problem 1
    # Reuse alpha_blend from problem 2
    parser.add_argument("--channels", "-c", dest="channels", action="append")

    # Problem 3 arguments
    # Reuse neighbourhood_size from problem 1
    parser.add_argument("--gaussian-sigma-spatial", "-ss", dest="gaussian_sigma_spatial", action="store")
    parser.add_argument("--gaussian-sigma-intensity", "-si", dest="gaussian_sigma_intensity", action="store")

    # Problem 4 arguments
    parser.add_argument("--part-a", dest="demonstration_mode", const=0, action="store_const")
    parser.add_argument("--part-b", dest="demonstration_mode", const=1, action="store_const")
    parser.add_argument("--part-c", dest="demonstration_mode", const=2, action="store_const")
    parser.add_argument("--swirl-radius", "-sr", dest="swirl_radius", action="store")
    parser.add_argument("--swirl-angle", "-sa", dest="swirl_angle", action="store")
    parser.add_argument("--swirl-anticlockwise", dest="swirl_direction", const=-1, action="store_const")
    parser.add_argument("--swirl-clockwise", dest="swirl_direction", const=1, action="store_const")
    parser.add_argument("--bilinear", dest="bilinear", action="store_true")
    parser.add_argument("--no-bilinear", "--nearest-neighbour", dest="bilinear", action="store_false")
    parser.add_argument("--lpf-n", dest="lpf_n", action="store")
    parser.add_argument("--lpf-K", dest="lpf_K", action="store")
    parser.set_defaults(bilinear=True, demonstration_mode=3)

    args = parser.parse_args()

    # Checks whether we are outputting to a file or displaying in a window
    display = args.outputFile == None
    outputFile = args.outputFile
    displayMode = "Screen" if display else "Saved to file"
    problem = None

    # Figure out which problem was requested
    try:
        problem = int(args.problem)
    except ValueError:
        print(f"-p accepts integers only, received {args.problem}")
        return

    inputFile = None

    # Check the input file is valid
    if os.path.isfile(args.inputFile):
        inputFile = cv2.imread(args.inputFile, cv2.IMREAD_COLOR)
    else:
        print(f"{args.inputFile} does not exist")
        return

    if inputFile.size == 0:
        print(f"{args.inputFile} is not a valid image")
        return

    output = None
    multiple_images = False
    displayTitle = "Output Image"
    timeTaken = 0

    print(f"Display Mode: {displayMode}")

    if problem == 1:
        # Get the parameters for problem 1 and their defaults if they're not set
        mode = args.mode if args.mode != None else "simple"
        darkening_coefficient = float(args.darkening_coefficient) if args.darkening_coefficient != None else (0.6 if mode == "rainbow" else 0.4)
        alpha_blend = float(args.alpha_blend) if args.alpha_blend != None else 1
        beta_blend = float(args.beta_blend) if args.beta_blend else (1 - alpha_blend if args.alpha_blend != None else (0.4 if mode == "rainbow" else 0.7))
        neighbourhood_size = int(args.neighbourhood_size) if args.neighbourhood_size else 4
        gaussian_sigma = float(args.gaussian_sigma) if args.gaussian_sigma != None else 2.2

        print("Problem 1: Light Leak Filters")
        print(f"Parameters: mode={mode}, darkening_coefficient={darkening_coefficient}, blending_coefficient={alpha_blend}")
        print(f"Optionals: beta_blend={beta_blend}, neighbourhood_size={neighbourhood_size} ({2 * neighbourhood_size + 1}), gaussian_sigma={gaussian_sigma}")
        print("Generating image...")

        start = time.time()

        # Generate the image for problem 1
        output = problem1(inputFile, mode, darkening_coefficient, alpha_blend, beta_blend, gaussian_n = neighbourhood_size, gaussian_sigma = gaussian_sigma)
        displayTitle = f"Problem 1, mode={mode}, dc={darkening_coefficient}, bc={alpha_blend}"

        end = time.time()
        timeTaken = end - start
    elif problem == 2:
        # Gets the arguments for problem 2
        mode = args.mode if args.mode != None else "simple"
        alpha_blend = float(args.alpha_blend) if args.alpha_blend != None else 0.6
        channels = [int(x) for x in args.channels] if args.channels != None else [0, 1]

        print("Problem 2: Pencil/Charcoal Effect")
        print(f"Parameters: mode={mode}, blending_coefficient={alpha_blend}")

        joined = ",".join([str(x) for x in channels])
        displayTitle = f"Problem 2, mode={mode}, bc={alpha_blend}"

        # Handles any extra stuff for the coloured mode
        if mode.lower() == "coloured_pencil":
            print(f"Coloured Mode Only: channels={joined}")
            displayTitle = f"{displayTitle}, channels={joined}"

        print("Generating image...")
        start = time.time()

        # Generate the image
        output = problem2(inputFile, mode, alpha_blend, channels = channels)

        end = time.time()
        timeTaken = end - start
    elif problem == 3:
        # Gets the arguments for problem 3
        neighbourhood_size = int(args.neighbourhood_size) if args.neighbourhood_size else 2
        gaussian_sigma_spatial = float(args.gaussian_sigma_spatial) if args.gaussian_sigma_spatial != None else 6
        gaussian_sigma_intensity = float(args.gaussian_sigma_intensity) if args.gaussian_sigma_intensity != None else 6

        print("Problem 3: Smoothing and Beautifying Filter")
        print(f"Parameters: neighbourhood_size={neighbourhood_size} ({2 * neighbourhood_size + 1}), gaussian_sigma_spatial={gaussian_sigma_spatial}, gaussian_sigma_intensity={gaussian_sigma_intensity}")

        print("Generating image...")
        start = time.time()

        # Generates problem 3's image
        output = problem3(inputFile, neighbourhood_size, gaussian_sigma_intensity, gaussian_sigma_spatial)
        displayTitle = f"Problem 3, n={neighbourhood_size}, ss={gaussian_sigma_spatial}, si={gaussian_sigma_intensity}"

        end = time.time()
        timeTaken = end - start
    elif problem == 4:
        # We have different parts for q4
        demonstration_mode = int(args.demonstration_mode) if args.demonstration_mode else 0
        swirl_radius = float(args.swirl_radius) if args.swirl_radius != None else 170
        swirl_angle = float(args.swirl_angle) if args.swirl_angle != None else math.pi / 2
        swirl_direction = int(args.swirl_direction) if args.swirl_direction != None else -1
        bilinear = args.bilinear if args.bilinear != None else True
        print(f"Parameters: swirl_radius={swirl_radius}, swirl_angle={swirl_angle}, swirl_direction={swirl_direction}, bilinear={bilinear}")

        if demonstration_mode == 0:
            # Default, just perform the swirl
            print("Problem 4a: Face Swirl")

            print("Generating image...")
            start = time.time()

            output = problem4_normal(inputFile, swirl_radius, swirl_angle, swirl_direction = swirl_direction, bilinear = bilinear)
            displayTitle = f"Problem 4a, r={swirl_radius}, theta={swirl_angle}, d={swirl_direction}, bl={bilinear}"

            end = time.time()
            timeTaken = end - start
        elif demonstration_mode == 1:
            # Low pass filtering
            lpf_n = float(args.lpf_n) if args.lpf_n != None else 1
            lpf_K = float(args.lpf_K) if args.lpf_K != None else 50

            print("Problem 4b: Face Swirl with Low Pass Filtering")
            print(f"Part B: lpf_n={lpf_n}, lpf_K={lpf_K}")

            print("Generating image...")
            start = time.time()

            output = problem4_blur_demonstration(inputFile, swirl_radius, swirl_angle, swirl_direction = swirl_direction, bilinear = bilinear, n = lpf_n, K = lpf_K)
            multiple_images = True
            displayTitle = f"Problem 4b, r={swirl_radius}, theta={swirl_angle}, d={swirl_direction}, bl={bilinear}"

            end = time.time()
            timeTaken = end - start
        elif demonstration_mode == 2:
            # Reverse and difference
            print("Problem 4c: Face Swirl with Reverse and Difference")
            print("Generating image...")
            start = time.time()

            output = problem4_inverse_demonstration(inputFile, swirl_radius, swirl_angle, swirl_direction = swirl_direction, bilinear = bilinear)
            multiple_images = True
            displayTitle = f"Problem 4c, r={swirl_radius}, theta={swirl_angle}, d={swirl_direction}, bl={bilinear}"

            end = time.time()
            timeTaken = end - start
        elif demonstration_mode == 3:
            lpf_n = float(args.lpf_n) if args.lpf_n != None else 1
            lpf_K = float(args.lpf_K) if args.lpf_K != None else 50

            print("Problem 4: All Parts")
            print(f"Part B: lpf_n={lpf_n}, lpf_K={lpf_K}")
            print("Generating image...")

            start = time.time()

            output = problem4(inputFile, swirl_radius, swirl_angle, swirl_direction = swirl_direction, bilinear = bilinear, n = lpf_n, K = lpf_K)
            multiple_images = True
            displayTitle = f"Problem 4, All Parts"

            end = time.time()
            timeTaken = end - start
        else:
            print(f"Unknown part specified for problem 4 ({demonstration_mode})")
    else:
        print(f"Cannot find problem {problem}")
        return

    print(f"Generated in {timeTaken} seconds")

    # Display the image or write it to a file
    if display:
        if multiple_images:
            display_images(output)
        else:
            cv2.imshow(displayTitle, output)
            cv2.waitKey(0)
    else:
        outImage = output

        if multiple_images:
            outImage = np.concatenate([img[1] for img in outImage], axis=1)

        cv2.imwrite(outputFile, outImage)
        print(f"Written image to {outputFile}")

# Only run the command line utility if it is the main program
if __name__ == "__main__":
    main()
