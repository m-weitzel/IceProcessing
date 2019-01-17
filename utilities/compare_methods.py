""" Compares different object detection methods applied to the same ice crystal image, overlaying them as diversely colored
contours on top of the image itself. Is also capable of loading a "ground truth" image and adding its contours as a reference.
The method "eval_contour_quality" can then quantitatively compare how the compared methods misdetect the objects."""


import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utilities.IceSizing import MicroImg
from utilities.make_pretty_figure import imshow_in_figure
from utilities.savefig_central import savefig_ipa
from scipy.spatial import distance as dist


def main():
    label_img = True

    # folder = '/uni-mainz.de/homes/maweitze/CCR/01Mar/'
    # folder = '../../../Dropbox/Dissertation/Ergebnisse/EisMainz/CNN/Training/3103M1/'
    # folder = '/ipa2/holo/mweitzel/Windkanal/Ice/CCR/2804/M1'
    folder = '/ipa/holo/mweitzel/Ergebnisse/EisMainz/CNN/Training/3103M1'

    pixel_size = 0.33

    # label_path_list = ('Ice-1-104_features.png', 'Ice-2_features.png')
    # img_path_list = ('Ice-1-104.png','Ice-2-172.png')

    # img_path = 'Ice-4.png'
    img_path = 'Ice-2_color_cropped.png'

    if label_img:
        label_path = 'Ice-2_label_cropped.png'
        label_filter = 'Bin'

    # for label_path, img_path in zip(label_path_list, img_path_list):
        label_img = MicroImg('label', folder, label_path, pixel_size, (label_filter, 0), maxsize=np.inf, dilation=0, fill_flag=False, min_dist_to_edge=5)
        label_centerpts = [l['Center Points'] for l in label_img.data]

    # img_mean = cv2.meanStdDev(cv2.cvtColor(label_img.initial_image, cv2.COLOR_BGR2GRAY))
    # thresh = [img_mean[0]-c*img_mean[1] for c in (0, 1/4, 1/2, 3/4, 1)]
    # thresh = float(img_mean[0]-0.5*img_mean[1])
    # print(str(img_mean))

    # filter_list = (('Bin', 0), ('Bin', 80), ('Bin', 100), ('Bin', 127), ('Bin', 140))
    # filter_list = [('Otsu', int(t)) for t in thresh]
    # filter_list = [('Adaptive', int(t)) for t in (401, 601, 801, 1001, 1201, 1401)]

    thresh = 0
    filter_list = (('Bin', thresh), ('Otsu', thresh), ('Adaptive', 1001), ('kmeans', 0, 1), ('Canny', 0))#, ('Gradient', 0))#, ('kmeans', 0, 1))
    # filter_list = list()
    # filter_list.append(('Canny', 0))
    # filter_list.append(('Bin', thresh))
    # filter_list.append(('kmeans', 4, 2))
    # filter_list.append(('Adaptive', 1001))
    # filter_list.append(('Otsu', 0))
    # filter_list = (('kmeans', 0, 1, 'Single'), ('kmeans', 0, 1, 'Multi'))

    # filter_list = (('Otsu', 255), ('Adaptive', 1001))

    # f, axarr = plt.subplots(np.floor_divide(len(filter_list), 3)+1, 3)

    image_data_list = list()

    for i, t_filter in enumerate(filter_list):
        img_data = MicroImg('ice', folder, img_path, pixel_size, t_filter, maxsize=np.inf, dilation=20, fill_flag=False, min_dist_to_edge=5)
        # img_data.image = cv2.cvtColor(img_data.image, cv2.COLOR_BGR2GRAY)

        # object_detected_img = img_data.image.copy()
        object_detected_img = np.ones(img_data.initial_image.shape)
        conts = img_data.contours
        majsizs = [i['Long Axis'] for i in img_data.data]
        minsizs = [i['Short Axis'] for i in img_data.data]
        centerpts = [i['Center Points'] for i in img_data.data]

        tmp_c = list()
        tmp_maj = list()
        tmp_min = list()
        tmp_ct = list()

        for c, maj, mino, ct in zip(conts, majsizs, minsizs, centerpts):
            if cv2.contourArea(c) > 250:
                tmp_c.append(c)
                tmp_maj.append(maj)
                tmp_min.append(mino)
                tmp_ct.append(ct)

        conts = tmp_c
        majsizs = tmp_maj
        minsizs = tmp_min
        centerpts = tmp_ct

        del tmp_c, tmp_maj, tmp_min, tmp_ct

        cv2.drawContours(object_detected_img, conts, -1, 0, 2)
        for c in conts:
            cv2.fillPoly(object_detected_img, pts=[c], color=0)

        image_data_list.append(img_data)

        del t_filter, conts, majsizs, minsizs, object_detected_img, centerpts, img_data

    proxs = sort_by_dist_to_label(image_data_list, label_centerpts, max_dist=100)

    cols = ((255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255))
    col_label = (0, 0, 0)
    img = image_data_list[0].initial_image

    fig_siz, ax_siz = imshow_in_figure(figspan=(15, 8), dpi=100)

    label_majsizs = [l['Long Axis'] for l in label_img.data]
    xax = np.arange(len(label_majsizs))

    # ax_siz.scatter(xax, label_majsizs, c='k', marker='s', s=50)
    for i, (col, p, f) in enumerate(zip(cols, proxs, filter_list)):
        cps = p['closest_particles']
        xax_in = [2*j-(i-1)/len(proxs)+1 for j in np.arange(len(label_majsizs))]
        # ax_siz.scatter(xax_in, [cp['Long Axis'] for cp in cps], c=[c/255 for c in col])
        # ax_siz.scatter(xax_in, [cp['Long Axis']-la for cp, la in zip(cps, label_majsizs)], c=[c/255 for c in col], edgecolors='k')

        size_diffs = [cp['Long Axis']-la if len(cp) > 0 else 0 for cp, la in zip(cps, label_majsizs)]

        ax_siz.bar(xax_in, size_diffs, 1/len(proxs), label=f[0], color=[c/255 for c in col])
        e_mean = np.mean([np.abs(s) for s in size_diffs])
        print('Average sizing error - {0}: {1:.2f} um'.format(f[0], e_mean))
        ax_siz.axhline(e_mean, color=[c/255 for c in col])
        ax_siz.set_xticks(xax_in)
        ax_siz.set_xticklabels(([str(int(x/2))for x in xax_in]))

        ax_siz.set_xlim([0, 23.8])

        ax_siz.set_ylim([-10, 10])
        ax_siz.set_ylabel('Size deviation in $\mu$m')

    axsiz_legend = ax_siz.legend(frameon=True, scatterpoints=1, fontsize=20)

    for col, method_property in zip(cols, image_data_list):
        if label_img:
            eval_contour_quality(method_property.contours, img, method_property.bin_img, label_img, method_property.thresh_type[0])
        cv2.drawContours(img, method_property.contours, -1, col, 2)
        # ax_siz.scatter(np.arange(len(method_property['majsizs'])), method_property['majsizs'], c=tuple(c/255 for c in col))

    if label_img:
        tmp = list()
        for i, (c, d) in enumerate(zip(label_img.contours, label_img.data)):
            if cv2.contourArea(c) > 750:
                tmp.append(c)
                cv2.putText(img, str(i), (int(d['Center Points'][0]), int(d['Center Points'][1])), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (255, 255, 255), 2)
        cv2.drawContours(img, tmp, -1, col_label, 4)

        # ax_siz.scatter(np.arange(len(method_property['majsizs'])), method_property['majsizs'], c='k')

    fig, ax_img = imshow_in_figure(img, grid=False, hide_axes=True)
    for col, method_property in zip(cols, image_data_list):
        ax_img.plot([], c=[c/255 for c in col], label=method_property.thresh_type[0])

    if label_img:
        ax_img.plot([], c=[c/255 for c in col_label], label='Ground truth')

    plt.legend(loc='lower right')

    savefig_ipa(fig, 'thresh_contour_comparison')
    savefig_ipa(fig_siz, 'thresh_size_comparison')

    plt.show()


def eval_contour_quality(detected_conts, initial_img, object_detected_img, label_img, filter):

    label_conts = label_img.contours
    label_map = label_img.bin_img

    # _, label_map = cv2.threshold(label_map, 127, 1, cv2.THRESH_BINARY)
    # label_map = cv2.cvtColor(label_map, cv2.COLOR_BGR2GRAY)

    # cv2.drawContours(label_map, label_conts, -1, (0), 2)
    #
    # for c in label_conts:
    #     cv2.fillPoly(label_map, pts=[c], color=(0))

    in_img_and_label = (label_map * object_detected_img).astype('uint8')

    in_img_not_in_label = np.maximum(object_detected_img.astype(np.int16) - label_map.astype(np.int16), np.zeros_like(label_map))

    in_label_not_in_img = np.maximum((label_map*255).astype(np.int16) - object_detected_img.astype(np.int16), np.zeros_like(label_map))

    # im_cont = initial_img
    # cv2.drawContours(im_cont, detected_conts, -1, (0, 0, 0), 2)
    # cv2.drawContours(im_cont, label_conts, -1, (255, 255, 255), 2)
    # cv2.namedWindow(filter[0]+str(filter[1]), cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(filter[0]+str(filter[1]), 768, 768)
    # cv2.imshow(filter[0]+str(filter[1]), im_cont)

    # plt.figure()
    # plt.imshow(in_label_not_in_img)
    # plt.title('Missed')
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(in_img_not_in_label)
    # plt.title('Too much')

    false_pos = np.count_nonzero(in_img_not_in_label) / np.count_nonzero(label_map) * 100
    false_neg = np.count_nonzero(in_label_not_in_img) / np.count_nonzero(label_map) * 100

    print('\nFalse Positive ' + filter+ ': ' + "{:.2f}".format(false_pos) + '%')
    print('False Negative ' + filter + ': ' + "{:.2f}".format(false_neg) + '%')
    print('Sum of Errors ' + filter + ': ' + "{:.2f}".format(false_pos + false_neg) + '%')


def sort_by_dist_to_label(idata, label_centerpts, max_dist=100):

    closest_particles = list()

    for id in idata:
        id_dict = dict()
        id_dict['filter'] = id.thresh_type[0]
        id_dict['closest_particles'] = []
        id_dict['label_centerpt'] = []
        for l_ct in label_centerpts:
            if len(id.data) > 0:
                dists = [dist.euclidean(c['Center Points'], l_ct) for c in id.data]

                sl = list(zip(dists, id.data))
                sl.sort(key=lambda x: abs(x[0]))

                if sl[0][0] < max_dist:
                    id_dict['closest_particles'].append(sl[0][1])
                    id.data = np.asarray([s[1] for s in sl[1:]])
                else:
                    id_dict['closest_particles'].append([])

                id_dict['label_centerpt'].append(l_ct)

            else:
                id_dict['closest_particles'].append([])
                id_dict['label_centerpt'].append(l_ct)

        closest_particles.append(id_dict)
        del id_dict

    return closest_particles


if __name__ == '__main__':
    main()