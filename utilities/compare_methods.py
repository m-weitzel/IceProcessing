import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pickle
from IceSizing import MicroImg

folder = '/uni-mainz.de/homes/maweitze/CCR/01Mar/'
# folder = '../../../Dropbox/Dissertation/Ergebnisse/EisMainz/CNN/Training/3103M1/'

# label_path_list = ('Ice-1-104_features.png', 'Ice-2_features.png')
# img_path_list = ('Ice-1-104.png','Ice-2-172.png')

img_path = 'Ice-5.png'
# img_path = 'Ice-2-172.png'
# img_path = 'Ice-1-104.png'
label_filter = 'Bin'

# for label_path, img_path in zip(label_path_list, img_path_list):
# label_img = MicroImg('label', folder, label_path, (label_filter, 0), maxsize=np.inf)

# img_mean = cv2.meanStdDev(cv2.cvtColor(label_img.initial_image, cv2.COLOR_BGR2GRAY))
# thresh = [img_mean[0]-c*img_mean[1] for c in (0, 1/4, 1/2, 3/4, 1)]
# thresh = float(img_mean[0]-0.5*img_mean[1])
# print(str(img_mean))

# filter_list = (('Bin', 0), ('Bin', 80), ('Bin', 100), ('Bin', 127), ('Bin', 140))
# filter_list = (('Bin', 130), ('Bin', 150), ('Bin', 170), ('Bin', 190))
# filter_list = [('Otsu', int(t)) for t in thresh]
# filter_list = (('Bin', thresh[2]), ('Otsu', thresh[2]), ('Adaptive', 1001), ('Canny', 0))
# filter_list = [('Adaptive', int(t)) for t in (401, 601, 801, 1001, 1201, 1401)]

thresh = 127
# filter_list = (('Bin', thresh), ('Otsu', thresh), ('Adaptive', 1001), ('Canny', 0), ('Gradient', 0), ('kmeans', 0, 1))
# filter_list = (('Otsu', thresh), ('Adaptive', 1001), ('Canny', 0), ('Gradient', 0), ('kmeans', 0, 1))
# filter_list = (('kmeans', 5, 1), ('kmeans', 3, 1), ('kmeans', 4, 1))
filter_list = (('Gradient', 0), ('Gradient', 1))

# f, axarr = plt.subplots(np.floor_divide(len(filter_list), 3)+1, 3)

cont_list = list()

for i, filter in enumerate(filter_list):
    img_data = MicroImg('ice', folder, img_path, filter, maxsize=np.inf, dilation=50, fill_flag=False, min_dist_to_edge=0)
    #img_data.image = cv2.cvtColor(img_data.image, cv2.COLOR_BGR2GRAY)

    # object_detected_img = img_data.image.copy()
    object_detected_img = np.ones(img_data.initial_image.shape)
    conts = img_data.contours
    cont_list.append(conts)
    tmp = list()

    for c in conts:
        if cv2.contourArea(c) > 750:
            tmp.append(c)

    conts = tmp

    cv2.drawContours(object_detected_img, conts, -1, (0), 2)
    for c in conts:
        cv2.fillPoly(object_detected_img, pts=[c], color=(0))

    cols = ((255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255))

img = img_data.initial_image

for col, cont in zip(cols, cont_list):
    cv2.drawContours(img, cont, -1, col, 5)

plt.imshow(img)
cv2.imwrite('../../../Desktop/0808Ice11_comp.png', img)
plt.show()
cv2.waitKey(0)