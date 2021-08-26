import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def get_list_all_images():
    return sorted(os.listdir('images'), key=lambda x: int(x[:-len('.PNG')]))


for x in get_list_all_images():
    img = cv2.imread('images/' + str(x))
    if img is None:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 25)
    canny = cv2.Canny(blurred, 100, 200)
    #circles
    points = np.argwhere(canny > 0)
    center, radius = cv2.minEnclosingCircle(points)
    diameter = 2 * radius
    print('Diameter {}: {}'.format(x, diameter))

cv2.waitKey(0)
cv2.destroyAllWindows()

