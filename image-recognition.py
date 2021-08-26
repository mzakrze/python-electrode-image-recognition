import numpy as np
import cv2
import math
import os
from skimage import measure
import matplotlib.pyplot as plt

def get_list_all_images():
    return sorted(os.listdir('images'), key=image_name_to_number)
    return ["0.PNG"]

def image_name_to_number(image_name):
    return int(image_name[:-len('.PNG')])

def calculate_diameter(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    canny = cv2.Canny(blurred, 100, 200)
    #circles
    points = np.argwhere(canny > 0)
    center, radius = cv2.minEnclosingCircle(points)
    diameter = 2 * radius
    return diameter

def calculate_min_max_dist(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    regions = measure.regionprops(binary)
    circle = regions[0]
    yc, xc = circle.centroid

    max_dist = 0
    min_dist = 999999999999999
    all_contours = contours[0]
    for cnt in all_contours:
        y = cnt[0][0]
        x = cnt[0][1]
        dist = math.sqrt((yc - y)**2 + (xc - x)**2)
        if dist > max_dist:
            max_dist = dist
        if dist < min_dist:
            min_dist = dist

    return min_dist, max_dist

def calculate_black_area(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # threshold
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV +
                           cv2.THRESH_OTSU)[1]
    # apply morphology open with a circular shaped kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    threshold = 20
    white = 0
    black = 0
    height, width = binary.shape[:2]
    for y in range(height):
        for x in range(width):
            if (img[y][x][0] < threshold and img[y][x][1] < threshold and img[y][x][2] < threshold):
                black += 1
            else:
                white += 1
    return black / (white + black) * 100

    # https://www.geeksforgeeks.org/opencv-counting-the-number-of-black-and-white-pixels-in-the-image/
    # black_pixels = np.sum(binary == 0)
    # white_pixels = np.sum(binary == 255)
    # all_pixels = np.sum(binary)
    # print(white_pixels)
    # print(black_pixels)
    # print(all_pixels)
    # return black_pixels / (white_pixels + black_pixels) * 100

def debug_show_processed_image(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # threshold
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV +
                           cv2.THRESH_OTSU)[1]
    # apply morphology open with a circular shaped kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # find contour and draw on input (for comparison with circle)
    cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = cnts[0]
    result = img.copy()
    cv2.drawContours(result, [c], -1, (0, 255, 0), 1)
    # find radius and center of equivalent circle from binary image and draw circle
    # see https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
    regions = measure.regionprops(binary)
    circle = regions[0]
    yc, xc = circle.centroid
    radius = circle.equivalent_diameter / 2.0
    print("radius =", radius, " center =", xc, ",", yc)
    xx = int(round(xc))
    yy = int(round(yc))
    rr = int(round(radius))
    cv2.circle(result, (xx, yy), rr, (0, 0, 255), 1)
    # write result to disk
    cv2.imwrite("dark_circle_fit.png", result)
    # display it
    cv2.imshow("image", img)
    cv2.imshow("thresh", thresh)
    cv2.imshow("binary", binary)
    cv2.imshow("result", result)

# debug_show_processed_image(cv2.imread('images/708.PNG'))

def main():
    image_no = []
    image_diameter = []
    image_min_dist = []
    image_max_dist = []
    image_black_area = []

    for image_name in get_list_all_images():
        img = cv2.imread('images/' + str(image_name))

        image_no.append(image_name_to_number(image_name))

        diameter = calculate_diameter(img)
        print('Diameter {}: {}'.format(image_name, diameter))
        image_diameter.append(diameter)

        min_dist, max_dist = calculate_min_max_dist(img)
        print('Min dist {}: {}'.format(image_name, min_dist))
        print('Max dist {}: {}'.format(image_name, max_dist))
        image_min_dist.append(min_dist)
        image_max_dist.append(max_dist)

        black_area = calculate_black_area(img)
        print('Black area % {}: {}'.format(image_name, black_area))
        image_black_area.append(black_area)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.title('Electrode diameter')
    plt.plot(image_no, image_diameter)
    plt.show()

    plt.title('Black area %')
    plt.plot(image_no, image_black_area)
    plt.show()

    plt.title("Electrode min&max distance from center")
    plt.plot(image_no, image_min_dist, label="min dist")
    plt.plot(image_no, image_max_dist, label="max dist")
    plt.show()

    # figure, axis = plt.subplots(3)
    #
    # axis[0].plot(image_no, image_diameter)
    # axis[0].set_title("Electrode diameter")
    #
    # axis[1].plot(image_no, image_min_dist, label="min dist")
    # axis[1].plot(image_no, image_max_dist, label="max dist")
    # axis[1].set_title("Electrode min&max distance from center")
    #
    # axis[2].plot(image_no, image_black_area)
    # axis[2].set_title('Black area %')
    #
    # plt.show()

main()