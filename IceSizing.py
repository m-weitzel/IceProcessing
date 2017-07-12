import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import os
from matplotlib import pyplot as plt


class MicroImg:
    def __init__(self, type_phase, folder, filename, thresh_type=None):
        self.type_phase = type_phase
        self.folder = folder
        self.filename = filename
        self.pixels_per_metric = 3
        # self.initial_image = cv2.cvtColor(cv2.imread(self.full_path()), cv2.COLOR_BGR2GRAY)
        self.initial_image = cv2.imread(self.full_path())
        self.particles = ()
        self.thresh_type = thresh_type
        self.bin_img = self.binarize_image()
        self.contours = self.get_contours_from_img()
        self.dimensions, self.processed_image = self.get_dims_and_process()

    def get_contours_from_img(self):

        cnts = cv2.findContours(self.bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1]

        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
        (cnts, _) = contours.sort_contours(cnts)

        return cnts

    def get_dims_and_process(self):
        dims = list()
        img = self.initial_image.copy()
        for c in self.contours:
            this_dimensions, img = draw_box_from_conts(c, img, self.pixels_per_metric)
            dims.append(this_dimensions)

        dims = list(filter(None, dims))
        return np.asarray(dims), img

    def full_path(self):
        return self.folder+'/'+self.filename

    def binarize_image(self):

        img = self.initial_image

        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self.thresh_type == "Canny":
            canny_low = 6
            thresh = cv2.Canny(gray, canny_low, canny_low*3)
        elif self.thresh_type == "Bin":
            threshold = gray.mean()-1.5*gray.std()
            rt, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        elif self.thresh_type == "Otsu":
            threshold = gray.mean()-1.5*gray.std()
            rt, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        elif self.thresh_type == "Adaptive":
            block_size = 767
            # block_size = 751
            adpt_constant = 7
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, adpt_constant)
        else:
            raise NameError('Invalid Threshold Method')

        # load the initial_image, convert it to grayscale, and blur it slightly

        dilation = 20
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        return thresh


def midpoint(pt_a, pt_b):
    return (pt_a[0] + pt_b[0]) * 0.5, (pt_a[1] + pt_b[1]) * 0.5


def draw_box_from_conts(contour, img, pixels_per_metric):
    img_processed = img.copy()
    if cv2.contourArea(contour) < 750:
        return [], img

    box = cv2.minAreaRect(contour)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    if (box > 2040).any() or (box < 8).any():
        return [], img

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

    l = np.max([d_a, d_b])
    w = np.min([d_a, d_b])

    if pixels_per_metric is None:
        pixels_per_metric = 3

    dim_l = l / pixels_per_metric
    dim_w = w / pixels_per_metric

    area = cv2.contourArea(contour) / (pixels_per_metric ** 2)

    # if dim_l > 100:
    #     return [], img

    dimensions = (dim_l, dim_w, area, center_point[0], center_point[1])

    cv2.putText(img_processed, "{:.1f}um".format(d_a / pixels_per_metric),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(img_processed, "{:.1f}um".format(d_b / pixels_per_metric),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    cv2.drawContours(img_processed, [box.astype("int")], -1, (0, 255, 0), 2)
    cv2.drawContours(img_processed, contour, -1, (255, 0, 0), 2)
    # cv2.imshow('a', img_processed)
    # cv2.waitKey(0)

    return dimensions, img_processed

def main():
    None

if __name__ == "__main__":
    main()