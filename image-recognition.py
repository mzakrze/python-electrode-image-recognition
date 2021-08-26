import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def get_list_all_images():
    return sorted(os.listdir('images'), key=image_name_to_number)

def image_name_to_number(image_name):
    return int(image_name[:-len('.PNG')])

image_no = []
image_diameter = []

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

    image_no.append(image_name_to_number(x))
    image_diameter.append(diameter)

cv2.waitKey(0)
cv2.destroyAllWindows()


plt.plot(image_no, image_diameter)
plt.show()