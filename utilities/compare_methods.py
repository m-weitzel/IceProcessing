""" Compares different object detection methods applied to the same ice crystal image, overlaying them as diversely colored
contours on top of the image itself. Is also capable of loading a "ground truth" image and adding its contours as a reference.
The method "eval_contour_quality" can then quantitatively compare how the compared methods misdetect the objects."""


import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utilities.IceSizing import MicroImg


def main():
    label_img = False

    # folder = '/uni-mainz.de/homes/maweitze/CCR/01Mar/'
    folder = '../../../Dropbox/Dissertation/Ergebnisse/EisMainz/CNN/Training/3103M1/'

    # label_path_list = ('Ice-1-104_features.png', 'Ice-2_features.png')
    # img_path_list = ('Ice-1-104.png','Ice-2-172.png')

    # img_path = 'Ice-4.png'
    img_path = 'Ice-1_color.png'
    if label_img:
        label_path = 'Ice-1_label.png'
        label_filter = 'Bin'

    # for label_path, img_path in zip(label_path_list, img_path_list):
        label_img = MicroImg('label', folder, label_path, (label_filter, 0), maxsize=np.inf, dilation=0, fill_flag=False)

    # img_mean = cv2.meanStdDev(cv2.cvtColor(label_img.initial_image, cv2.COLOR_BGR2GRAY))
    # thresh = [img_mean[0]-c*img_mean[1] for c in (0, 1/4, 1/2, 3/4, 1)]
    # thresh = float(img_mean[0]-0.5*img_mean[1])
    # print(str(img_mean))

    # filter_list = (('Bin', 0), ('Bin', 80), ('Bin', 100), ('Bin', 127), ('Bin', 140))
    # filter_list = [('Otsu', int(t)) for t in thresh]
    # filter_list = [('Adaptive', int(t)) for t in (401, 601, 801, 1001, 1201, 1401)]

    thresh = 0
    # filter_list = (('Bin', thresh), ('Otsu', thresh), ('Adaptive', 1001), ('Canny', 0), ('Gradient', 0), ('kmeans', 0, 1))
    # filter_list = list()
    # filter_list.append(('Bin', thresh))

    filter_list = (('Otsu', 255), ('Adaptive', 1001))

    # f, axarr = plt.subplots(np.floor_divide(len(filter_list), 3)+1, 3)

    cont_list = list()

    for i, filter in enumerate(filter_list):
        img_data = MicroImg('ice', folder, img_path, filter, maxsize=np.inf, dilation=20, fill_flag=False, min_dist_to_edge=0)
        # img_data.image = cv2.cvtColor(img_data.image, cv2.COLOR_BGR2GRAY)

        # object_detected_img = img_data.image.copy()
        object_detected_img = np.ones(img_data.initial_image.shape)
        conts = img_data.contours
        cont_list.append(conts)
        tmp = list()

        for c in conts:
            if cv2.contourArea(c) > 750:
                tmp.append(c)

        conts = tmp
        del tmp

        cv2.drawContours(object_detected_img, conts, -1, 0, 2)
        for c in conts:
            cv2.fillPoly(object_detected_img, pts=[c], color=0)

        if label_img:
            eval_contour_quality(img_data, object_detected_img, label_img, filter)

    cols = ((255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255))
    col_label = (255, 255, 255)
    img = img_data.initial_image

    for col, cont in zip(cols, cont_list):
        cv2.drawContours(img, cont, -1, col, 2)

    tmp = list()

    if label_img:
        for c in label_img.contours:
            if cv2.contourArea(c) > 750:
                tmp.append(c)

        cv2.drawContours(img, tmp, -1, col_label, 4)

    plt.imshow(img)
    plt.legend()
    for col, this_filter in zip([[c/255 for c in co] for co in cols], filter_list):
        plt.plot([], c=col, label=this_filter[0])

    if label_img:
        plt.plot([], c=[c/255 for c in col_label], label='Ground truth')

    plt.savefig(os.path.join(folder, '/uni-mainz.de/homes/maweitze/Dropbox/Dissertation/Präsentationen und Berichte/Holo Group + Cloud Physics Seminars Mai 2018', str(thresh)+'.png'), bbox_inches='tight',
                dpi=300)

    plt.show()
    cv2.waitKey(0)


def eval_contour_quality(img_data, object_detected_img, label_img, filter):

    label_conts = label_img.contours
    label_map = label_img.initial_image.copy()

    _, label_map = cv2.threshold(label_map, 127, 255, cv2.THRESH_BINARY)

    cv2.drawContours(label_map, label_conts, -1, (0), 2)

    for c in label_conts:
        cv2.fillPoly(label_map, pts=[c], color=(0))

    in_img_and_label = (label_map * object_detected_img).astype('uint8')

    in_label_not_in_img = object_detected_img * 255 - label_map
    in_label_not_in_img[in_label_not_in_img < 0] = 0

    in_img_not_in_label = label_map - object_detected_img * 255
    in_img_not_in_label[in_img_not_in_label < 0] = 0

    im_cont = img_data.initial_image.copy()
    cv2.drawContours(im_cont, img_data.contours, -1, (0, 0, 0), 2)
    cv2.drawContours(im_cont, label_conts, -1, (255, 255, 255), 2)
    # cv2.namedWindow(filter[0]+str(filter[1]), cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(filter[0]+str(filter[1]), 768, 768)
    # cv2.imshow(filter[0]+str(filter[1]), im_cont)

    false_pos = np.count_nonzero(in_img_not_in_label) / np.count_nonzero(label_map) * 100
    false_neg = np.count_nonzero(in_label_not_in_img) / np.count_nonzero(label_map) * 100

    print('\nFalse Positive ' + filter[0] + ': ' + "{:.2f}".format(false_pos) + '%')
    print('False Negative ' + filter[0] + ': ' + "{:.2f}".format(false_neg) + '%')
    print('Sum of Errors ' + filter[0] + str(filter[1]) + ': ' + "{:.2f}".format(false_pos + false_neg) + '%')


if __name__ == '__main__':
    main()