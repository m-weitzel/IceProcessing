""" Contains the MicroImg class.
Central code for the sizing of microscopic images of ice crystals and water drops.
The MicroImg constructor expects the following inputs:
type_phase: Phase of the objects in the image. 'Drop' lets thresholding default to binary color thresholding
folder: Folder containing the image
filename: File name of the image to be investigated
thresh_type: 2-tuple of the threshold identifier ('Bin', 'Otsu', 'Adaptive', 'Canny', 'kmeans', 'Gradient') and
            an associated parameter (e.g. threshold value for "Bin" or window size for "adaptive"
minsize, maxsize: minimum and maximum size of detected objects
dilation: width of dilation and erosion in pixels
fill_flag: boolean toggling whether holes within objects are to be manually filled or not
min_dist_to_edge: particles closer to the image edges are disregarded
optional_object_filter: manual string containing a condition used to filter out detected objects
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from sklearn.cluster import KMeans
import os


class MicroImg:
    def __init__(self, type_phase, folder, filename, thresh_type=(None, 0), minsize=750, maxsize=100000, dilation=30, fill_flag=True, min_dist_to_edge=4, optional_object_filter_condition='False'):
        self.type_phase = type_phase
        self.folder = folder
        self.filename = filename
        self.pixels_per_metric = 3
        self.mindisttoedge = min_dist_to_edge
        self.minsize = minsize
        self.maxsize = maxsize
        self.dilation = dilation
        self.fill_flag = fill_flag
        self.initial_image = cv2.imread(self.full_path())
        # self.initial_image = cv2.Laplacian(cv2.imread(self.full_path()), cv2.CV_64F)
        self.thresh_type = thresh_type
        self.thresh_img = self.binarize_image()
        self.contours = self.get_contours_from_img()
        self.bin_img = self.fill_contours()
        self.streak_flag = (type_phase == 'Streak')
        self.data, self.processed_image = self.get_data_and_process(optional_object_filter_condition, self.streak_flag)

    def get_contours_from_img(self):

        cnts = cv2.findContours(self.thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]

        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
        if len(cnts) > 0:
            (cnts, _) = contours.sort_contours(cnts)

        filtered_contours = list()
        for c in cnts:
            c_area = cv2.contourArea(c)
            if (c_area > self.minsize) & (c_area < self.maxsize):
                box = cv2.minAreaRect(c)
                box = cv2.boxPoints(box)
                box = np.array(box, dtype="int")

                # if ~((box > (self.thresh_img.shape[0]-self.mindisttoedge)).any() or (box < self.mindisttoedge).any()):
                filtered_contours.append(c)
        return filtered_contours

    def get_data_and_process(self, optional_object_filter_condition, streak_flag):
        data = list()
        img = self.initial_image.copy()
        for c in self.contours:

            this_data = dict()
            this_center, la = cv2.minEnclosingCircle(c)
            this_data['Area'] = cv2.contourArea(c) / (self.pixels_per_metric**2)
            this_data['File Name'] = self.filename

            this_data['Orientation'] = 0
            this_data['CSP'] = 0
            this_data['Short Axis'] = 1

            cv2.drawContours(img, c, -1, (0, 255, 0), 2)

            # this_center = (int(la), int(this_data['Center Points'][1]))

            cv2.circle(img, (int(this_center[0]), int(this_center[1])), 5, (0, 0, 255), 3)
            cv2.circle(img, (int(this_center[0]), int(this_center[1])), int(la), (255, 0, 0), 3)

            this_data['Long Axis'] = 2*la/self.pixels_per_metric
            this_data['Center Points'] = this_center
            data.append(this_data)

            # this_data, img = draw_box_from_conts(c, img, self.pixels_per_metric, optional_object_filter_condition, streak_flag)
            # if this_data:
            #     csp = this_data['Area'] * this_data['Short Axis'] \
            #           * (2 * this_data['Long Axis']+2 * this_data['Short Axis'])\
            #           / (cv2.arcLength(c, True)/self.pixels_per_metric)
            #     new_data = {'CSP': csp, 'File Name': self.filename}
            #     this_data.update(new_data)
            #     data.append(this_data)

        data = list(filter(None, data))
        return np.asarray(data), img

    def full_path(self):
        return os.path.join(self.folder, self.filename)

    def binarize_image(self):

        img = self.initial_image

        if (self.type_phase == 'Drop') | (self.type_phase == 'drop'):

            blue = img[:, :, 2]
            rt, thresh = cv2.threshold(blue, 250, 255, cv2.THRESH_BINARY)
        else:
            if len(img.shape) > 2:
                # img = cv2.convertScaleAbs(img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            gray = cv2.GaussianBlur(gray, (15, 15), 0)

            if self.thresh_type[0] == "Canny":
                if self.thresh_type[1] == 0:
                    canny_low = 6
                else:
                    canny_low = self.thresh_type[1]
                if len(self.thresh_type) > 2:
                    if self.thresh_type[2] == 0:
                        canny_high = 3*canny_low
                    else:
                        canny_high = self.thresh_type[2]
                else:
                    canny_high = 3*canny_low

                thresh = cv2.Canny(gray, canny_low, canny_high, L2gradient=True)
            elif self.thresh_type[0] == "Bin":
                if self.thresh_type[1] != 0:
                    threshold = self.thresh_type[1]
                else:
                    threshold = gray.mean()-0.5*gray.std()
                    print('Using threshold value of {} (mean grey value minus 0.5*sigma)'.format(threshold))
                if np.sign(self.thresh_type[1]) == -1:
                    print('Received negative threshold value, using inverted thresholding.')
                    rt, thresh = cv2.threshold(gray, -threshold, 255, cv2.THRESH_BINARY)
                else:
                    rt, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            elif self.thresh_type[0] == "Otsu":
                if np.sign(self.thresh_type[1]) == -1:
                    print('Received negative threshold value, using inverted thresholding.')
                    rt, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    print('Otsu value: {}'.format(rt))
                else:
                    rt, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    print('Otsu value: {}'.format(rt))
            elif self.thresh_type[0] == "Adaptive":
                block_size = self.thresh_type[1]
                if block_size == 0:
                    block_size = 751

                adpt_constant = 7
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, adpt_constant)
            elif self.thresh_type[0] == "Gradient":
                # grad_img = np.uint8(np.sqrt(np.uint8(cv2.Sobel(gray, -1, 0, 1)) ** 2 + np.uint8(cv2.Sobel(gray, -1, 1, 0)) ** 2))
                grad_img = cv2.Laplacian(gray, cv2.CV_8U)

                grad_img = cv2.GaussianBlur(grad_img, (15, 15), 0)
                # thresh = cv2.adaptiveThreshold(grad_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 501, 0)
                rt, thresh = cv2.threshold(grad_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            elif self.thresh_type[0] == "kmeans":
                # if self.thresh_type[3] == "Multi":
                #     print('\nUsing multidimensional cluster for kmeans.')
                #     # grad_img = np.uint8(
                #     #     np.sqrt(np.uint8(cv2.Sobel(gray, -1, 0, 1)) ** 2 + np.uint8(cv2.Sobel(gray, -1, 1, 0)) ** 2))
                #     grad_img = cv2.Laplacian(gray, cv2.CV_8U)
                #     p_r = cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(gray))))))))
                #     p_r = np.abs(p_r-gray)
                #
                #     im_list = gray.reshape(gray.shape[0] * gray.shape[1])
                #     grad_list = grad_img.reshape(grad_img.shape[0] * grad_img.shape[1])
                #     p_r_list = p_r.reshape(p_r.shape[0] * p_r.shape[1])
                #     comp_list = list()
                #     for i, g, p in zip(im_list, grad_list, p_r_list):
                #         comp_list.append((i, g, p))
                #     clustered_data = np.asarray(comp_list)
                #
                # else:
                print('\nUsing only greyscale intensity for kmeans.')
                clustered_data = img.reshape(gray.shape[0] * gray.shape[1], 3)

                if self.thresh_type[1] == 0:
                    n = 2
                else:
                    n = self.thresh_type[1]
                try:
                    pos_label = self.thresh_type[2]
                except IndexError:
                    pos_label = 0

                kmeans = KMeans(n_clusters=n)
                kmeans.fit(clustered_data)

                centroids = kmeans.cluster_centers_
                # centroids = ms.cluster_centers_
                # labels = ms.labels_
                labels = kmeans.labels_

                la = [1 if ((l == pos_label) or (l == 1)) else 0 for l in labels]

                thresh = np.reshape(la, [img.shape[0], img.shape[1]]).astype('uint8')

            # elif self.thresh_type[0] == "Color":
            #     lower_range = np.array([110, 50, 50])
            #     upper_range = np.array([130, 255, 255])
            #
            #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            #     hsv = cv2.GaussianBlur(hsv, (7, 7), 0)
            #
            #     thresh = cv2.inRange(hsv, lower_range, upper_range)
            else:
                raise NameError('Invalid Threshold Method')

            # load the initial_image, convert it to grayscale, and blur it slightly

            dilation = self.dilation
            if dilation > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
                thresh = cv2.dilate(thresh, kernel, iterations=1)
                thresh = cv2.erode(thresh, kernel, iterations=1)

            h, w = thresh.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)

            if self.fill_flag:
                thresh_floodfill = thresh.copy()
                cv2.floodFill(thresh_floodfill, mask, (0, 0), 255);
                thresh_floodfill_inv = cv2.bitwise_not(thresh_floodfill)
                if np.any(thresh_floodfill > 0):
                    print('Floodfill had any effect whatsoever.')
                thresh_filled = thresh | thresh_floodfill_inv

                if np.any((thresh|thresh_floodfill_inv) < 255):
                    thresh = thresh_filled
                else:
                    print('Fill holes failed.')

        return thresh

    def fill_contours(self):

        ii = cv2.cvtColor(np.zeros_like(self.initial_image), cv2.COLOR_BGR2GRAY)
        cv2.drawContours(ii, self.contours, -1, (255, 0, 0), 2)
        for c in self.contours:
            cv2.fillPoly(ii, pts=[c], color=(255, 0, 0))

        return ii


def midpoint(pt_a, pt_b):
    return (pt_a[0] + pt_b[0]) * 0.5, (pt_a[1] + pt_b[1]) * 0.5


def draw_box_from_conts(contour, img, pixels_per_metric, optional_object_filter_condition=False, streak_flag=False):
    img_processed = img.copy()
    # if cv2.contourArea(contour) < 750:
    #     return [], img

    box = cv2.minAreaRect(contour)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # if (box > 2040).any() or (box < 8).any():
    #     return [], img

    box = perspective.order_points(box)

    for (x, y) in box:
        cv2.circle(img_processed, (int(x), int(y)), 5, (0, 0, 255), -1)

    (tl, tr, br, bl) = box
    # img_box = (min(box[:, 0]), max(box[:, 0]), min(box[:, 1]), max(box[:, 1]))
    # partial_img = img[int(img_box[0]):int(img_box[1]), int(img_box[2]):int(img_box[3])]

    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    center_point = midpoint((tlblX, tlblY), (trbrX, trbrY))

    cv2.circle(img_processed, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(img_processed, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(img_processed, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(img_processed, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    cv2.line(img_processed, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
    cv2.line(img_processed, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

    d_a = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    d_b = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    if d_a > d_b:
        orientation = np.arctan((blbrX-tltrX)/(blbrY-tltrY))
    else:
        orientation = np.arctan((tlblX-trbrX)/(tlblY-trbrY))

    l = np.max([d_a, d_b])
    w = np.min([d_a, d_b])

    if pixels_per_metric is None:
        pixels_per_metric = 3

    dim_l = l / pixels_per_metric
    dim_w = w / pixels_per_metric

    area = cv2.contourArea(contour) / (pixels_per_metric ** 2)

    # if dim_l > 100:
    #     return [], img

    if eval(optional_object_filter_condition):
        return [], img
    else:

        data = {'Long Axis': dim_l, 'Short Axis': dim_w, 'Area': area, 'Center Points': (center_point[0], center_point[1]), 'Orientation': orientation}

        if streak_flag:
            if d_a > d_b:
                cv2.putText(img_processed, "{:.1f}um".format(d_a / pixels_per_metric),
                            (int(tltrX - 150), int(tltrY+80)), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (90, 90, 255), 4)
            else:

                cv2.putText(img_processed, "{:.1f}um".format(d_b / pixels_per_metric),
                            (int(trbrX + 10), int(trbrY-100)), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (90, 90, 255), 4)
        else:
            # if d_a > d_b:
            cv2.putText(img_processed, "{:.1f}um".format(d_a / pixels_per_metric),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
            # else:
            cv2.putText(img_processed, "{:.1f}um".format(d_b / pixels_per_metric),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)

        cv2.drawContours(img_processed, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.drawContours(img_processed, contour, -1, (255, 0, 0), 2)
        # cv2.imshow('a', img_processed)
        # cv2.waitKey(0)

        return data, img_processed


def main():
    None


if __name__ == "__main__":
    main()