import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pickle
from IceSizing import MicroImg

folder = '../../../Dropbox/Dissertation/Ergebnisse/EisMainz/CNN/Training/3103M1/'

# label_path_list = ('Ice-1-104_features.png', 'Ice-2_features.png')
# img_path_list = ('Ice-1-104.png','Ice-2-172.png')

label_path = 'Ice-2_features.png'
img_path = 'Ice-2-172.png'

label_filter = 'Bin'

# for label_path, img_path in zip(label_path_list, img_path_list):
label_img = MicroImg('label', folder, label_path, (label_filter, 0), maxsize=np.inf)

img_mean = cv2.meanStdDev(cv2.cvtColor(label_img.initial_image, cv2.COLOR_BGR2GRAY))
# thresh = [img_mean[0]-c*img_mean[1] for c in (0, 1/4, 1/2, 3/4, 1)]
thresh = float(img_mean[0]-0.5*img_mean[1])
# print(str(img_mean))

# filter_list = (('Bin', 0), ('Bin', 80), ('Bin', 100), ('Bin', 127), ('Bin', 140))
# filter_list = (('Bin', 130), ('Bin', 150), ('Bin', 170), ('Bin', 190))
# filter_list = [('Otsu', int(t)) for t in thresh]
# filter_list = (('Bin', thresh[2]), ('Otsu', thresh[2]), ('Adaptive', 1001), ('Canny', 0))
# filter_list = [('Adaptive', int(t)) for t in (401, 601, 801, 1001, 1201, 1401)]
filter_list = (('Bin', thresh), ('Otsu', thresh), ('Adaptive', 1001), ('Canny', 0), ('Gradient', 0), ('kmeans', 0, 1))

# f, axarr = plt.subplots(np.floor_divide(len(filter_list), 3)+1, 3)
f, axarr = plt.subplots(2, 3)

cont_list = list()

for i, filter in enumerate(filter_list):
    img_data = MicroImg('ice', folder, img_path, filter, maxsize=np.inf)
    #img_data.image = cv2.cvtColor(img_data.image, cv2.COLOR_BGR2GRAY)

    # object_detected_img = img_data.image.copy()
    object_detected_img = np.ones(img_data.initial_image.shape)
    conts = img_data.contours
    tmp = list()

    for c in conts:
        if cv2.contourArea(c) > 750:
            tmp.append(c)

    conts = tmp

    cv2.drawContours(object_detected_img, conts, -1, (0), 2)
    for c in conts:
        cv2.fillPoly(object_detected_img, pts=[c], color=(0))

    # cv2.imshow('im_a', object_detected_img)

    label_conts = label_img.contours
    label_map = label_img.initial_image.copy()

    _, label_map = cv2.threshold(label_map, 127, 255, cv2.THRESH_BINARY)

    cv2.drawContours(label_map, label_conts, -1, (0), 2)

    for c in label_conts:
        cv2.fillPoly(label_map, pts=[c], color=(0))

    in_img_and_label = (label_map*object_detected_img).astype('uint8')

    in_label_not_in_img = object_detected_img*255 - label_map
    in_label_not_in_img[in_label_not_in_img < 0] = 0

    in_img_not_in_label = label_map - object_detected_img*255
    in_img_not_in_label[in_img_not_in_label < 0] = 0

    im_cont = img_data.initial_image.copy()
    cv2.drawContours(im_cont, conts, -1, (0,0,0), 2)
    cont_list.append(conts)
    cv2.drawContours(im_cont, label_conts, -1, (255, 255, 255), 2)
    # cv2.namedWindow(filter[0]+str(filter[1]), cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(filter[0]+str(filter[1]), 768, 768)
    # cv2.imshow(filter[0]+str(filter[1]), im_cont)

    axarr[np.floor_divide(i, 3), np.mod(i,3)].imshow(im_cont)
    axarr[np.floor_divide(i, 3), np.mod(i, 3)].set_title(str(filter))

    false_pos = np.count_nonzero(in_img_not_in_label)/np.count_nonzero(label_map)*100
    false_neg = np.count_nonzero(in_label_not_in_img)/np.count_nonzero(label_map)*100

    print('FP'+filter[0]+': '+"{:.2f}".format(false_pos)+'%')
    print('FN '+filter[0]+': '+"{:.2f}".format(false_neg)+'%')
    print('Sum of Errors '+filter[0]+str(filter[1])+': '+"{:.2f}".format(false_pos+false_neg)+'%')

plt.show()
cv2.waitKey(0)