import numpy as np
from IceSizing import MicroImg
import find_couples
from matplotlib import pyplot as plt
import cv2

img_ice = MicroImg('Ice', 'img/ice/3103M1/', 'Ice-4.png', 'Otsu')
img_drop = MicroImg('Drop', '/uni-mainz.de/homes/maweitze/Dropbox/Dissertation/Ergebnisse/EisMainz/3103/M1/',
                        'Drops-4.png_withCircles.png', 'Canny')

pairs_list, img_comp = find_couples.main(img_ice, img_drop)

list_couples = list(map(lambda x: (x.ice_center, x.drop_center), pairs_list))

plt.scatter(list_couples[0], list_couples[1])
x = [1, 1000]
y = x
plt.plot(x, y)
plt.xlim(0, 1000)
plt.show()

cv2.namedWindow('Comparison', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Comparison', 768, 768)
cv2.imshow('Comparison', img_comp)
cv2.waitKey(0)



# def process_folder(folder, filter_type):
#     img_list = os.listdir(folder)
#
#     img_object_list = []
#     # i = 0
#
#     for img in img_list:
#         img_object = MicroImg('ice', folder, img, filter_type)
#         this_image = img_object.initial_image.copy()
#         # cnts = img_object.contours
#         # dims = []
#         # for c in cnts:
#         #     this_dimensions, this_image = draw_box_from_conts(c, this_image, 3)
#         #     dims.append(this_dimensions)
#         #
#         # dims = list(filter(None, dims))
#
#         dims = img_object.dimensions
#
#         img_object.dimensions = np.asarray(dims)
#         img_object.initial_image = this_image
#         img_object_list.append(img_object)
#
#         # plot_folder(img_object)
#         # i = i+1
#     return img_object_list
#
#
# def plot_folder(microimg):
#     cv2.namedWindow(microimg.filename, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(microimg.filename, 768, 768)
#
#     cv2.imshow(microimg.filename, microimg.initial_image)
#     cv2.waitKey(0)
#
#
# def process_img_obj(img_object):
#     dims = [img_o.dimensions for img_o in img_object]
#     dims = np.concatenate(list(filter(lambda c: c.shape != (0,), dims)), axis=0)
#     ls = dims[:, 0]
#     ws = dims[:, 1]
#     cnt_area = dims[:, 2]
#     aspect_ratio = ls/ws
#     area_ratio = cnt_area/(ls*ws)
#     center_point = (dims[:, 3], dims[:, 4])
#     return ls, ws, cnt_area, aspect_ratio, area_ratio, center_point
#
#
# def main():
#     # folder_list = ('img/ice/2203M1', 'img/ice/2203M2', 'img/ice/2203M3','img/ice/3103M1','img/ice/0604M1', 'img/ice/0604M2', 'img/ice/0604M1', 'img/ice/2804M1', 'img/ice/2804M2')
#     # folder_list = ('img/ice/2203M1', 'img/ice/2203M2', 'img/ice/2203M3')
#     folder_list = ('img/ice/0604M2', 'img/ice/2203M1')
#
#     folder_count = 1
#     folder_num = list()
#     folder_avg_area = list()
#     folder_avg_ws = list()
#     folder_avg_aspr = list()
#     folder_avg_area_ratio = list()
#
#     for folder_path in folder_list:
#         img_object_bin = process_folder(folder_path, 'Bin')
#         (Ls_bin, Ws_bin, cnt_area_bin, aspr_bin, ar_bin, ctrpt_bin) = process_img_obj(img_object_bin)
#         print('Binary completed.')
#
#         img_object_otsu = process_folder(folder_path, 'Otsu')
#         (Ls_otsu, Ws_otsu, cnt_area_otsu, aspr_otsu, ar_otsu, ctrpt_otsu) = process_img_obj(img_object_otsu)
#         print('Otsu completed.')
#
#         img_object_canny = process_folder(folder_path, 'Canny')
#         (Ls_canny, Ws_canny, cnt_area_canny, aspr_canny, ar_canny, ctrpt_canny) = process_img_obj(img_object_canny)
#         print('Canny completed.')
#
#         img_object_adapt = process_folder(folder_path, 'Adaptive')
#         (Ls_adapt, Ws_adapt, cnt_area_adapt, aspr_adapt, ar_adapt, ctrpt_adapt) = process_img_obj(img_object_adapt)
#         print('Adaptive completed.')
#
#         # pickle.dump(((Ls_bin, Ws_bin, cnt_area_bin, aspr_bin, ar_bin), (Ls_otsu, Ws_otsu, cnt_area_otsu, aspr_otsu, ar_otsu),
#         #             (Ls_canny, Ws_canny, cnt_area_canny, aspr_canny, ar_canny), (Ls_adapt, Ws_adapt, cnt_area_adapt, aspr_adapt, ar_adapt)),
#         #             open(folder_path[-6:], "wb"))
#
#         # plt.figure(folder_count)
#         #
#         # plt.subplot(2, 3, 1)
#         # plt.hist([cnt_area_bin, cnt_area_canny, cnt_area_otsu, cnt_area_adapt], histtype='bar', ec='black', label=['Binary', 'Canny', 'Otsu', 'Adaptive'], alpha=0.5)
#         # # plt.hist(cnt_area_otsu, histtype='bar', ec='black', label='Otsu', alpha=0.5)
#         # # plt.hist(cnt_area_bin, histtype='bar', ec='black', label='Binary', alpha=0.5)
#         # # plt.hist(cnt_area_canny, histtype='bar', ec='black', label='Canny', alpha=0.5)
#         # plt.title('Contour Area')
#         # plt.legend(loc='upper right')
#         #
#         # plt.subplot(2, 3, 2)
#         # plt.hist([Ls_bin, Ls_canny, Ls_otsu, Ls_adapt], histtype='bar', ec='black', label=['Binary', 'Canny', 'Otsu', 'Adaptive'], alpha=0.5)
#         # # plt.hist(Ls_otsu, histtype='bar', ec='black', label='Otsu', alpha=0.5)
#         # # plt.hist(Ls_bin, histtype='bar', ec='black', label='Binary', alpha=0.5)
#         # # plt.hist(Ls_canny, histtype='bar', ec='black', label='Canny', alpha=0.5)
#         # plt.title('Major Axis')
#         # plt.legend(loc='upper right')
#         #
#         # plt.subplot(2, 3, 3)
#         # plt.hist([Ws_bin, Ws_canny, Ws_otsu, Ws_adapt], histtype='bar', ec='black', label=['Binary', 'Canny', 'Otsu', 'Adaptive'], alpha=0.5)
#         # # plt.hist(Ws_otsu, histtype='bar', ec='black', label='Otsu', alpha=0.5)
#         # # plt.hist(Ws_bin, histtype='bar', ec='black', label='Binary', alpha=0.5)
#         # # plt.hist(Ws_canny, histtype='bar', ec='black', label='Canny', alpha=0.5)
#         # plt.title('Minor Axis')
#         # plt.legend(loc='upper right')
#         #
#         # plt.subplot(2, 3, 4)
#         # plt.hist([aspr_bin, aspr_canny, aspr_otsu, aspr_adapt], bins=[1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 4], histtype='bar', ec='black', label=['Binary', 'Canny', 'Otsu', 'Adaptive'], alpha=0.5)
#         # # plt.hist(Ls_otsu/Ws_otsu, histtype='bar', ec='black', label='Otsu', alpha=0.5)
#         # # plt.hist(Ls_bin/Ws_bin, histtype='bar', ec='black', label='Binary', alpha=0.5)
#         # # plt.hist(Ls_canny/Ws_canny, histtype='bar', ec='black', label='Canny', alpha=0.5)
#         # plt.title('Aspect Ratio')
#         # plt.legend(loc='upper right')
#         #
#         # plt.subplot(2, 3, 5)
#         # plt.hist([ar_bin, ar_canny, ar_otsu, ar_adapt], ec='black', label=['Binary', 'Canny', 'Otsu', 'Adaptive'], alpha=0.5)
#         # # plt.hist(cnt_area_otsu/(Ls_otsu*Ws_otsu), ec='black', label='Otsu', alpha=0.5)
#         # # plt.hist(cnt_area_bin/(Ls_bin*Ws_bin), ec='black', label='Binary', alpha=0.5)
#         # # plt.hist(cnt_area_canny/(Ls_canny*Ws_canny), ec='black', label='Canny', alpha=0.5)
#         # plt.title('Normalized Area')
#         # plt.legend(loc='upper right')
#         #
#         # plt.suptitle(folder_path)
#         #
#         folder_avg_area.append((cnt_area_bin.mean(), cnt_area_canny.mean(), cnt_area_otsu.mean(), cnt_area_adapt.mean()))
#         folder_avg_area_ratio.append((ar_bin.mean(), ar_canny.mean(), ar_otsu.mean(), ar_adapt.mean()))
#         folder_avg_aspr.append((aspr_bin.mean(), aspr_canny.mean(), aspr_otsu.mean(), aspr_adapt.mean()))
#         folder_avg_ws.append((Ws_bin.mean(), Ws_canny.mean(), Ws_otsu.mean(), Ws_adapt.mean()))
#         folder_num.append((len(Ls_bin), len(Ls_canny), len(Ls_otsu), len(Ls_adapt)))
#
#         folder_count += 1
#         print(folder_path + ' completed.')
#
#     figure = plt.figure(folder_count)
#
#     labels_filter = ['Binary', 'Canny', 'Otsu', 'Adaptive']
#     xs = np.arange(0, len(labels_filter), 1)
#     folder_avg_area = list(map(list, zip(*folder_avg_area)))
#     folder_avg_area_ratio = list(map(list, zip(*folder_avg_area_ratio)))
#     folder_avg_aspr = list(map(list, zip(*folder_avg_aspr)))
#     folder_avg_ws = list(map(list, zip(*folder_avg_ws)))
#     folder_num = list(map(list, zip(*folder_num)))
#
#     subplot1 = figure.add_subplot(2, 3, 1)
#     for nums, label_flt in zip(folder_num, labels_filter):
#         subplot1.plot(nums, 'o', label=label_flt)
#     subplot1.set_xticks(xs)
#     subplot1.set_xticklabels(folder_list)
#     subplot1.set_title('Number of objects')
#     plt.legend()
#
#     subplot2 = figure.add_subplot(2, 3, 2)
#     for avg_Ws, label_flt in zip(folder_avg_ws, labels_filter):
#         subplot2.plot(avg_Ws, 'o', label=label_flt)
#     subplot2.set_xticks(xs)
#     subplot2.set_xticklabels(folder_list)
#     subplot2.set_title('Average major axis')
#     plt.legend()
#
#     subplot3 = figure.add_subplot(2, 3, 3)
#     for avg_area, label_flt in zip(folder_avg_area, labels_filter):
#         subplot3.plot(avg_area, 'o', label=label_flt)
#     subplot3.set_xticks(xs)
#     subplot3.set_xticklabels(folder_list)
#     subplot3.set_title('Average area')
#     plt.legend()
#
#     subplot4 = figure.add_subplot(2, 3, 4)
#     for avg_aspr, label_flt in zip(folder_avg_aspr, labels_filter):
#         subplot4.plot(avg_aspr, 'o', label=label_flt)
#     subplot4.set_xticks(xs)
#     subplot4.set_xticklabels(folder_list)
#     subplot4.set_title('Average aspect ratio')
#     plt.legend()
#
#     subplot5 = figure.add_subplot(2, 3, 5)
#     for avg_ar, label_flt in zip(folder_avg_area_ratio, labels_filter):
#         subplot5.plot(avg_ar, 'o', label=label_flt)
#     subplot5.set_xticks(xs)
#     subplot5.set_xticklabels(folder_list)
#     subplot5.set_title('Average area ratio')
#     plt.legend()
#
#     plt.show()