import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import os
from matplotlib import pyplot as plt
import pickle


class Particle:
    def __init__(self, type_phase, image_path, partial_image, contour):
        self.type_phase = type_phase
        self.image_path = image_path
        self.partial_image = partial_image
        self.contour = contour


class MicroImg:
    def __init__(self, type_phase, folder, filename, thresh_type):
        self.type_phase = type_phase
        self.folder = folder
        self.filename = filename
        self.image = cv2.imread(self.full_path())
        self.particles = ()
        self.thresh_type = thresh_type
        self.contours = self.process_image(self.image, self.thresh_type)
        self.dimensions = None

    @staticmethod
    def process_image(image_1, thresh_type):

        # load the image, convert it to grayscale, and blur it slightly

        gray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        thresh = threshold_image(gray, thresh_type)

        dilation = 20
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]

        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
        (cnts, _) = contours.sort_contours(cnts)

        return cnts

    def full_path(self):
        return self.folder+'/'+self.filename


def midpoint(pt_a, pt_b):
    return (pt_a[0] + pt_b[0]) * 0.5, (pt_a[1] + pt_b[1]) * 0.5


def threshold_image(image, thresh_type):
    if thresh_type == "Canny":
        canny_low = 8
        thresh = cv2.Canny(image, canny_low, canny_low*3)
    elif thresh_type == "Bin":
        threshold = 127
        rt, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    elif thresh_type == "Otsu":
        threshold = 127
        rt, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif thresh_type == "Adaptive":
        block_size = 767
        adpt_constant = 7
        thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, adpt_constant)
    else:
        raise NameError('Invalid Threshold Method')
    return thresh


def get_box_metrics(contour, img, pixels_per_metric):
    if cv2.contourArea(contour) < 750:
        return [], img

    # compute the rotated bounding box of the contour

    box = cv2.minAreaRect(contour)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    if (box > 2040).any() or (box < 8).any():
        return [], img

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)

    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    # img_box = (min(box[:, 0]), max(box[:, 0]), min(box[:, 1]), max(box[:, 1]))
    # partial_img = img[int(img_box[0]):int(img_box[1]), int(img_box[2]):int(img_box[3])]

    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the img
    cv2.circle(img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
    cv2.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

    # compute the Euclidean distance between the midpoints
    d_a = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    d_b = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    l = np.max([d_a, d_b])
    w = np.min([d_a, d_b])

    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixels_per_metric is None:
        # pixels_per_metric = dB / args["width"]
        pixels_per_metric = 3

    # compute the size of the object
    dim_l = l / pixels_per_metric
    dim_w = w / pixels_per_metric
    area = cv2.contourArea(contour) / (pixels_per_metric ** 2)

    dimensions = (dim_l, dim_w, area)

    # draw the object sizes on the img
    cv2.putText(img, "{:.1f}um".format(d_a / pixels_per_metric),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(img, "{:.1f}um".format(d_b / pixels_per_metric),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    # show the output img

    cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2)
    cv2.drawContours(img, contour, -1, (255, 0, 0), 2)
    # cv2.imshow(img_path, window)
    # cv2.waitKey(0)

    return dimensions, img


def process_folder(folder, filter_type):
    img_list = os.listdir(folder)

    img_object_list = []
    # i = 0

    for img in img_list:
        img_object = MicroImg('ice', folder, img, filter_type)
        this_image = img_object.image.copy()
        cnts = img_object.contours
        dims = []
        for c in cnts:
            this_dimensions, this_image = get_box_metrics(c, this_image, 3)
            dims.append(this_dimensions)

        dims = list(filter(None, dims))
        img_object.dimensions = np.asarray(dims)
        img_object.image = this_image
        img_object_list.append(img_object)

        # plot_folder(img_object)
        # i = i+1
    return img_object_list


def plot_folder(microimg):
    cv2.namedWindow(microimg.filename, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(microimg.filename, 768, 768)

    cv2.imshow(microimg.filename, microimg.image)
    cv2.waitKey(0)


def process_img_obj(img_object):
    dims = [img_o.dimensions for img_o in img_object]
    dims = np.concatenate(list(filter(lambda c: c.shape != (0,), dims)), axis=0)
    ls = dims[:, 0]
    ws = dims[:, 1]
    cnt_area = dims[:, 2]
    aspect_ratio = ls/ws
    area_ratio = cnt_area/(ls*ws)
    return ls, ws, cnt_area, aspect_ratio, area_ratio


#folder_list = ('img/ice/3103M1', 'img/ice/2804M1', 'img/ice/2804M2')
folder_list = ('img/ice/2203M1', 'img/ice/2203M2', 'img/ice/2203M3')

folder_count = 1
folder_num = list()
folder_avg_area = list()
folder_avg_Ws = list()
folder_avg_aspr = list()
folder_avg_area_ratio = list()


for folder_path in folder_list:
    img_object_Bin = process_folder(folder_path, 'Bin')
    (Ls_bin, Ws_bin, cnt_area_bin, aspr_bin, ar_bin) = process_img_obj(img_object_Bin)

    img_object_Otsu = process_folder(folder_path, 'Otsu')
    (Ls_otsu, Ws_otsu, cnt_area_otsu, aspr_otsu, ar_otsu) = process_img_obj(img_object_Otsu)

    img_object_Canny = process_folder(folder_path, 'Canny')
    (Ls_canny, Ws_canny, cnt_area_canny, aspr_canny, ar_canny) = process_img_obj(img_object_Canny)

    img_object_Adapt = process_folder(folder_path, 'Adaptive')
    (Ls_adapt, Ws_adapt, cnt_area_adapt, aspr_adapt, ar_adapt) = process_img_obj(img_object_Adapt)

    pickle.dump(((Ls_bin, Ws_bin, cnt_area_bin, aspr_bin, ar_bin), (Ls_otsu, Ws_otsu, cnt_area_otsu, aspr_otsu, ar_otsu),
                (Ls_canny, Ws_canny, cnt_area_canny, aspr_canny, ar_canny), (Ls_adapt, Ws_adapt, cnt_area_adapt, aspr_adapt, ar_adapt)),
                open(folder_path[-6:], "wb"))

    plt.figure(folder_count)

    plt.subplot(2, 3, 1)
    plt.hist([cnt_area_bin, cnt_area_canny, cnt_area_otsu, cnt_area_adapt], histtype='bar', ec='black', label=['Binary', 'Canny', 'Otsu', 'Adaptive'], alpha=0.5)
    # plt.hist(cnt_area_otsu, histtype='bar', ec='black', label='Otsu', alpha=0.5)
    # plt.hist(cnt_area_bin, histtype='bar', ec='black', label='Binary', alpha=0.5)
    # plt.hist(cnt_area_canny, histtype='bar', ec='black', label='Canny', alpha=0.5)
    plt.title('Contour Area')
    plt.legend(loc='upper right')

    plt.subplot(2, 3, 2)
    plt.hist([Ls_bin, Ls_canny, Ls_otsu, Ls_adapt], histtype='bar', ec='black', label=['Binary', 'Canny', 'Otsu', 'Adaptive'], alpha=0.5)
    # plt.hist(Ls_otsu, histtype='bar', ec='black', label='Otsu', alpha=0.5)
    # plt.hist(Ls_bin, histtype='bar', ec='black', label='Binary', alpha=0.5)
    # plt.hist(Ls_canny, histtype='bar', ec='black', label='Canny', alpha=0.5)
    plt.title('Major Axis')
    plt.legend(loc='upper right')

    plt.subplot(2, 3, 3)
    plt.hist([Ws_bin, Ws_canny, Ws_otsu, Ws_adapt], histtype='bar', ec='black', label=['Binary', 'Canny', 'Otsu', 'Adaptive'], alpha=0.5)
    # plt.hist(Ws_otsu, histtype='bar', ec='black', label='Otsu', alpha=0.5)
    # plt.hist(Ws_bin, histtype='bar', ec='black', label='Binary', alpha=0.5)
    # plt.hist(Ws_canny, histtype='bar', ec='black', label='Canny', alpha=0.5)
    plt.title('Minor Axis')
    plt.legend(loc='upper right')

    plt.subplot(2, 3, 4)
    plt.hist([aspr_bin, aspr_canny, aspr_otsu, aspr_adapt], bins=[1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 4], histtype='bar', ec='black', label=['Binary', 'Canny', 'Otsu', 'Adaptive'], alpha=0.5)
    # plt.hist(Ls_otsu/Ws_otsu, histtype='bar', ec='black', label='Otsu', alpha=0.5)
    # plt.hist(Ls_bin/Ws_bin, histtype='bar', ec='black', label='Binary', alpha=0.5)
    # plt.hist(Ls_canny/Ws_canny, histtype='bar', ec='black', label='Canny', alpha=0.5)
    plt.title('Aspect Ratio')
    plt.legend(loc='upper right')

    plt.subplot(2, 3, 5)
    plt.hist([ar_bin, ar_canny, ar_otsu, ar_adapt], ec='black', label=['Binary', 'Canny', 'Otsu', 'Adaptive'], alpha=0.5)
    # plt.hist(cnt_area_otsu/(Ls_otsu*Ws_otsu), ec='black', label='Otsu', alpha=0.5)
    # plt.hist(cnt_area_bin/(Ls_bin*Ws_bin), ec='black', label='Binary', alpha=0.5)
    # plt.hist(cnt_area_canny/(Ls_canny*Ws_canny), ec='black', label='Canny', alpha=0.5)
    plt.title('Normalized Area')
    plt.legend(loc='upper right')

    plt.suptitle(folder_path)

    folder_avg_area.append((cnt_area_bin.mean(), cnt_area_canny.mean(), cnt_area_otsu.mean(), cnt_area_adapt.mean()))
    folder_avg_area_ratio.append((ar_bin.mean(), ar_canny.mean(), ar_otsu.mean(), ar_adapt.mean()))
    folder_avg_aspr.append((aspr_bin.mean(), aspr_canny.mean(), aspr_otsu.mean(), aspr_adapt.mean()))
    folder_avg_Ws.append((Ws_bin.mean(), Ws_canny.mean(), Ws_otsu.mean(), Ws_adapt.mean()))
    folder_num.append((len(Ls_bin), len(Ls_canny), len(Ls_otsu), len(Ls_adapt)))

    folder_count += 1

figure = plt.figure(folder_count)

labels_filter = ['Binary', 'Canny', 'Otsu', 'Adaptive']
xs = np.arange(0, len(labels_filter), 1)
folder_avg_area = list(map(list, zip(*folder_avg_area)))
folder_avg_area_ratio = list(map(list, zip(*folder_avg_area_ratio)))
folder_avg_aspr = list(map(list, zip(*folder_avg_aspr)))
folder_avg_Ws = list(map(list, zip(*folder_avg_Ws)))
folder_num = list(map(list, zip(*folder_num)))


subplot1 = figure.add_subplot(2, 3, 1)
for nums, label_flt in zip(folder_num, labels_filter):
    subplot1.plot(nums, 'o', label=label_flt)
subplot1.set_xticks(xs)
subplot1.set_xticklabels(folder_list)
subplot1.set_title('Number of objects')
plt.legend()

subplot2 = figure.add_subplot(2, 3, 2)
for avg_Ws, label_flt in zip(folder_avg_Ws, labels_filter):
    subplot2.plot(avg_Ws, 'o', label=label_flt)
subplot2.set_xticks(xs)
subplot2.set_xticklabels(folder_list)
subplot2.set_title('Average major axis')
plt.legend()

subplot3 = figure.add_subplot(2, 3, 3)
for avg_area, label_flt in zip(folder_avg_area, labels_filter):
    subplot3.plot(avg_area, 'o', label=label_flt)
subplot3.set_xticks(xs)
subplot3.set_xticklabels(folder_list)
subplot3.set_title('Average area')
plt.legend()

subplot4 = figure.add_subplot(2, 3, 4)
for avg_aspr, label_flt in zip(folder_avg_aspr, labels_filter):
    subplot4.plot(avg_aspr, 'o', label=label_flt)
subplot4.set_xticks(xs)
subplot4.set_xticklabels(folder_list)
subplot4.set_title('Average aspect ratio')
plt.legend()

subplot5 = figure.add_subplot(2, 3, 5)
for avg_ar, label_flt in zip(folder_avg_area_ratio, labels_filter):
    subplot5.plot(avg_ar, 'o', label=label_flt)
subplot5.set_xticks(xs)
subplot5.set_xticklabels(folder_list)
subplot5.set_title('Average area ratio')
plt.legend()

plt.show()
