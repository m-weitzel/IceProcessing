""" Compares different object detection methods applied to the same ice crystal image, overlaying them as diversely colored
contours on top of the image itself. Is also capable of loading a "ground truth" image and adding its contours as a reference.
The method "eval_contour_quality" can then quantitatively compare how the compared methods misdetect the objects."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IceSizing import MicroImg
from make_pretty_figure import imshow_in_figure, savefig_ipa
from scipy.spatial import distance as dist
import logging


def main():
    label_img = True

    # folder = '/uni-mainz.de/homes/maweitze/CCR/01Mar/'
    # folder = '../../../Dropbox/Dissertation/Ergebnisse/EisMainz/CNN/Training/3103M1/'
    # folder = '/ipa2/holo/mweitzel/Windkanal/Ice/CCR/2804/M1'
    folder = '/ipa/holo/mweitzel/Ergebnisse/EisMainz/CNN/Training/3103M1'

    pixel_size = 0.33
    toggle_logging = True

    if toggle_logging:
        from time import strftime
        import os
        date = strftime("%d%b%y")
        time = strftime("%-H.%M.%S")

        path = '/ipa/holo/mweitzel/Ergebnisse/figdump'

        datepath = os.path.join(path, date)

        try:
            os.mkdir(datepath)
        except FileExistsError:
            pass
        log_fn = os.path.join(datepath, 'compare_method_log'+time+'.txt')
        logging.basicConfig(filename=log_fn, level=logging.DEBUG)

    # label_path_list = ('Ice-1-104_features.png', 'Ice-2_features.png')
    # img_path_list = ('Ice-1-104.png','Ice-2-172.png')

    # img_path = 'Ice-4.png'
    img_path = 'Ice-2_color_cropped.png'
    # img_path = 'Ice-1_color.png'

    if label_img:
        label_path = 'Ice-2_label_cropped.png'
        # label_path = 'Ice-1_label.png'
        label_filter = 'Bin'

    # for label_path, img_path in zip(label_path_list, img_path_list):
        label_img = MicroImg('label', folder, label_path, pixel_size, (label_filter, 0), maxsize=np.inf, dilation=0, fill_flag=False, min_dist_to_edge=0)
        label_centerpts = [l['Center Points'] for l in label_img.data]

    # img_mean = cv2.meanStdDev(cv2.cvtColor(label_img.initial_image, cv2.COLOR_BGR2GRAY))
    # thresh = [img_mean[0]-c*img_mean[1] for c in (0, 1/4, 1/2, 3/4, 1)]
    # thresh = float(img_mean[0]-0.5*img_mean[1])
    # print(str(img_mean))

    # filter_list = (('Bin', 0), ('Bin', 80), ('Bin', 100), ('Bin', 127), ('Bin', 140))
    # filter_list = [('Otsu', int(t)) for t in thresh]
    # filter_list = [('Adaptive', int(t)) for t in (401, 601, 801, 1001, 1201, 1401)]

    thresh = 0
    filter_list = (('Bin', thresh), ('Otsu', thresh), ('Adaptive', 1001), ('kmeans', 5, (0, 1, 2), 101), ('Canny', 0))#, ('Gradient', 0))#, ('kmeans', 0, 1))
    # filter_list = [('kmeans', 10, (0, 1, 2, 3, 4, 5), 5), ('kmeans', 10, (0, 1, 2, 3), 15), ('kmeans', 10, (0, 1, 2, 3), 25), ('kmeans', 10, (0, 1, 2, 3, 4, 5), 35), ('kmeans', 10, (0, 1, 2, 3, 4, 5), 45)]
    # filter_list = [('Otsu', thresh)]
    # filter_list = list()
    # filter_list.append(('Canny', 0))
    # filter_list = [('Bin', thresh)]
    # filter_list.append(('kmeans', 0, 1, 'Multi'))
    # filter_list.append(('Adaptive', 1001))
    # filter_list.append(('Otsu', 0))
    # filter_list = (('kmeans', 2, 0, 'Multi'), ('kmeans', 2, 1, 'Multi'), ('kmeans', 3, 0, 'Multi'), ('kmeans', 3, 1, 'Multi'), ('kmeans', 3, 2, 'Multi'),
    # ('kmeans', 4, 0, 'Multi'), ('kmeans', 4, 1, 'Multi'), ('kmeans', 4, 2, 'Multi'), ('kmeans', 4, 3, 'Multi'))
    # filter_list = (('kmeans', 3, 0, 'Multi'), ('kmeans', 3, (0, 1), 'Multi'), ('kmeans', 3, (0, 1, 2), 'Multi'))
    # filter_list = (('kmeans', 3, 0, 'Multi'), ('kmeans', 3, (0, 1), 'Multi'), ('kmeans', 3, (0, 1, 2), 'Multi'))
    # filter_list = (('kmeans', 4, 0, 'Multi'), ('kmeans', 4, 1, 'Multi'), ('kmeans', 4, 2, 'Multi'), ('kmeans', 4, 3, 'Multi'))
    # filter_list = (('kmeans', 3, (0, 1)), ('kmeans', 4, (0, 1)))
    # filter_list = [('kmeans', 6, (0, 1, 2, 3)),('kmeans', 7, (0, 1, 2, 3)),('kmeans', 8, (0, 1, 2, 3, 4)),('kmeans', 9, (0, 1, 2, 3, 4, 5)),('kmeans', 10, (0, 1, 2, 3, 4, 5)),('kmeans', 15, (0, 1, 2, 3, 4, 5, 6, 7)),]
                   # ('kmeans', 6, (0, 1, 2, 3), 'Multi'), ('kmeans', 7, (0, 1, 2, 3), 'Multi'), ('kmeans', 8, (0, 1, 2, 3, 4), 'Multi'), ('kmeans', 9, (0, 1, 2, 3, 4), 'Multi'), ('kmeans', 10, (0, 1, 2, 3, 4), 'Multi'), ('kmeans', 15, (0, 1, 2, 3, 4, 5), 'Multi')]
    # filter_list = [('kmeans', 3, 0), ('kmeans', 3, 1), ('kmeans', 3, 2)]

    # filter_list = [('kmeans', 3, 1)]*5
    # filter_list = (('Otsu', 255), ('Adaptive', 1001))

    # f, axarr = plt.subplots(np.floor_divide(len(filter_list), 3)+1, 3)

    image_data_list = list()

    for i, t_filter in enumerate(filter_list):
        img_data = MicroImg('ice', folder, img_path, pixel_size, t_filter, maxsize=np.inf, dilation=20, fill_flag=False, min_dist_to_edge=0)
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
            cv2.fillPoly(object_detected_img, pts=[c], color=(255, 0, 0))

        image_data_list.append(img_data)
        # plt.figure()
        # plt.imshow(img_data.processed_image)
        del t_filter, conts, majsizs, minsizs, object_detected_img, centerpts, img_data

    proxs = sort_by_dist_to_label(image_data_list, label_centerpts, max_dist=100)

    cols = ((255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255))
    col_label = (0, 0, 0)
    img = image_data_list[0].initial_image

    fig_siz, ax_siz = imshow_in_figure(figspan=(15, 8), dpi=100)

    # label_majsizs = [l['Long Axis'] for l in label_img.data]
    label_majsizs = [2*np.sqrt(float(a['Area'])/np.pi) for a in label_img.data]
    xax = np.arange(len(label_majsizs))

    # ax_siz.scatter(xax, label_majsizs, c='k', marker='s', s=50)
    for i, (col, p, f) in enumerate(zip(cols, proxs, filter_list)):
        cps = p['closest_particles']
        xax_in = [2*j-(i-1)/len(proxs)+1 for j in np.arange(len(label_majsizs))]
        # ax_siz.scatter(xax_in, [cp['Long Axis'] for cp in cps], c=[c/255 for c in col])
        # ax_siz.scatter(xax_in, [cp['Long Axis']-la for cp, la in zip(cps, label_majsizs)], c=[c/255 for c in col], edgecolors='k')
        # siz = [cp['Long Axis'] for cp in cps]
        siz = [2*np.sqrt(float(a['Area'])/np.pi) for a in cps]

        # size_diffs = [cp['Long Axis']-la if len(cp) > 0 else 0 for cp, la in zip(cps, label_majsizs)]
        size_diffs = [s-la for s,la in zip(siz, label_majsizs)]

        ax_siz.bar(xax_in[:-1], size_diffs[:-1], 1/len(proxs), label=f[0], color=[c/255 for c in col])
        e_mean = np.mean([np.abs(s) for s in size_diffs[:-1]])
        e_mean_rel = np.mean([np.abs(s/la)*100 for s, la in zip(size_diffs, label_majsizs)])
        print('Average absolute sizing error - {0}: {1:.2f} um'.format(f[0], e_mean))
        print('Average relative sizing error - {0}: {1:.2f}%'.format(f[0], e_mean_rel))
        ax_siz.axhline(e_mean, color=[c/255 for c in col])
        ax_siz.set_xticks(xax_in)
        ax_siz.set_xticklabels(([str(int(x/2)+1)for x in xax_in[:-1]]))
        ax_siz.set_xlim([0, xax_in[-1]-0.5])
        # ax_siz.set_xlim([0, 23.8])

        ax_siz.set_ylim([-12, 12])
        ax_siz.set_xlabel('Crystal no.')
        ax_siz.set_ylabel(r'Size deviation in $\mu$m')

    axsiz_legend = ax_siz.legend(frameon=True, scatterpoints=1, fontsize=20)

    for col, method_property in zip(cols, image_data_list):
        if label_img:
            eval_contour_quality(method_property.contours, img, method_property.bin_img, label_img, method_property.thresh_type)
        cv2.drawContours(img, method_property.contours, -1, col, 2)
        # ax_siz.scatter(np.arange(len(method_property['majsizs'])), method_property['majsizs'], c=tuple(c/255 for c in col))

    if label_img:
        tmp = list()
        for i, (c, d) in enumerate(zip(label_img.contours[:-1], label_img.data[:-1])):
        # for i, (c, d) in enumerate(zip(image_data_list[0].contours, image_data_list[0].data)):
            if (cv2.contourArea(c) > 750) & (d['Center Points'][0] < (2000)):
                tmp.append(c)
                cv2.putText(img, str(i+1), (int(d['Center Points'][0]-50), int(d['Center Points'][1]+50)), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 0, 0), 15)
                cv2.putText(img, str(i+1), (int(d['Center Points'][0]-50), int(d['Center Points'][1]+50)), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (250, 250, 250), 8)

        # cv2.drawContours(img, tmp, -1, col_label, 4)

        # ax_siz.scatter(np.arange(len(method_property['majsizs'])), method_property['majsizs'], c='k')

    fig, ax_img = imshow_in_figure(img, grid=False, hide_axes=True)
    rect = patches.Rectangle((50, 1200), 150, 40, linewidth=1, edgecolor='k', facecolor=(0.8, 0, 0, 1))
    ax_img.add_patch(rect)

    for col, method_property in zip(cols, image_data_list):
        ax_img.plot([], c=[c/255 for c in col], label=method_property.thresh_type[0])

    # if label_img:
        # ax_img.plot([], c=[c/255 for c in col_label], label='Marked by operator')

    plt.legend(loc='lower right', fontsize=24)

    savefig_ipa(fig, 'thresh_contour_comparison')
    savefig_ipa(fig_siz, 'thresh_size_comparison')

    # plot_clusters(image_data_list)

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

    msg_fp = 'False Positive {0}: {1:.2f}%'.format(filter, false_pos)
    print(msg_fp)
    logging.info(msg_fp)
    msg_fn = 'False Negative {0}: {1:.2f}%'.format(filter, false_neg)
    print(msg_fn)
    logging.info(msg_fn)
    msg_soe = 'Sum of Errors {0}: {1:.2f}%'.format(filter, false_pos+false_neg)
    print(msg_soe)
    logging.info(msg_soe)


def sort_by_dist_to_label(idata, label_centerpts, max_dist=100):

    closest_particles = list()

    for id in idata:
        id_dict = dict()
        id_dict['filter'] = id.thresh_type[0]
        id_dict['closest_particles'] = []
        id_dict['label_centerpt'] = []

        data = id.data
        for l_ct in label_centerpts:
            if len(data) > 0:
                dists = [dist.euclidean(c['Center Points'], l_ct) for c in data]

                sl = list(zip(dists, data))
                sl.sort(key=lambda x: abs(x[0]))

                if sl[0][0] < max_dist:
                    id_dict['closest_particles'].append(sl[0][1])
                    data = np.asarray([s[1] for s in sl[1:]])
                else:
                    id_dict['closest_particles'].append([])

                id_dict['label_centerpt'].append(l_ct)

            else:
                id_dict['closest_particles'].append([])
                id_dict['label_centerpt'].append(l_ct)

        closest_particles.append(id_dict)
        del id_dict

    return closest_particles


def plot_clusters(image_data_list):
    img_cluster = np.zeros_like(image_data_list[0].thresh_img)
    for i, im in enumerate(image_data_list):
        img_cluster = (img_cluster+(i+1)*im.thresh_img) * (im.thresh_img.astype(bool) ^ img_cluster.astype(bool))

    from matplotlib import colorbar, colors, cm

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # ax.imshow(img_cluster)
    # ax.colorbar(ticks=range(N), ax=ax)

    # cmap = colors.ListedColormap(['red', 'white', 'blue'])
    # cmap = plt.cm.jet
    # cmaplist = [cmap(i) for i in range(cmap.N)]
    # cmap = colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    # bounds = np.linspace(0, len(image_data_list), len(image_data_list)+1)
    # norm = colors.BoundaryNorm(bounds, cmap.N)
    # img = plt.imshow(img_cluster, norm=norm, cmap=cmap)
    # plt.colorbar(img, cmap=cmap, norm=norm, ticks=range(len(image_data_list)))
    # cb = colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    L = len(image_data_list)
    bounds = np.arange(len(image_data_list))
    cmap = cm.get_cmap("jet", lut=len(bounds)+1)
    cmap_bounds = np.arange(len(bounds)+2) - 0.5
    norm = colors.BoundaryNorm(cmap_bounds, cmap.N)
    ax = plt.imshow(img_cluster, cmap=cmap, norm=norm)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    cbar = plt.colorbar(p, ticks=np.arange(1, L+1), orientation="horizontal")
    cbar.clim(0.5, 4.5)
    # cbar.set_ticklabels(["Cluster {}".format(n) for n in np.arange(L)])


if __name__ == '__main__':
    main()
