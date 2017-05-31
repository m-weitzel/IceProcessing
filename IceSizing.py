import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import os


class MicroImg:

    def __init__(self, type_phase, folder, filename):
        self.type_phase = type_phase
        self.folder = folder
        self.filename = filename
        self.image = cv2.imread(self.full_path())
        self.contours = ""

    def full_path(self):
        return self.folder+'/'+self.filename


def midpoint(pt_a, pt_b):
    return (pt_a[0] + pt_b[0]) * 0.5, (pt_a[1] + pt_b[1]) * 0.5


def threshold_image(image, thresh_type):
    if thresh_type == "Canny":
        canny_low = 8
        thresh = cv2.Canny(image, canny_low, canny_low*3)
    elif thresh_type == "BinaryInv":
        threshold = 127
        rt, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    elif thresh_type == "BinaryInvOtsu":
        threshold = 127
        rt, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        raise NameError('Invalid Threshold Method')
    return thresh


def process_image(img, thresh_type):

    # load the image, convert it to grayscale, and blur it slightly

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    img_box = (min(box[:, 0]), max(box[:, 0]), min(box[:, 1]), max(box[:, 1]))
    partial_img = img[int(img_box[0]):int(img_box[1]), int(img_box[2]):int(img_box[3])]

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
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    L = np.max([dA, dB])
    W = np.min([dA, dB])

    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixels_per_metric is None:
        # pixels_per_metric = dB / args["width"]
        pixels_per_metric = 3

    # compute the size of the object
    dimL = L / pixels_per_metric
    dimW = W / pixels_per_metric
    area = cv2.contourArea(contour) / (pixels_per_metric ** 2)

    dimensions = (dimL, dimW, area)

    # draw the object sizes on the img
    cv2.putText(img, "{:.1f}um".format(dA / pixels_per_metric),
          (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
          0.65, (255, 255, 255), 2)
    cv2.putText(img, "{:.1f}um".format(dB / pixels_per_metric),
      (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
      0.65, (255, 255, 255), 2)
    # show the output img

    cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2)
    cv2.drawContours(img, contour, -1, (255,0,0), 2)
    # cv2.imshow(img_path, window)
    # cv2.waitKey(0)

    return dimensions, img




img_folder = 'img/ice/3103M1/'
img_list = os.listdir(img_folder)
img_object = MicroImg('ice', img_folder, img_list[0])

this_image = img_object.image.copy()

cnts = process_image(this_image, 'Canny')

cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Test', 768, 768)

dims = []

for c in cnts:
    this_dimensions, this_image = get_box_metrics(c, this_image, 3)
    dims = [dims,this_dimensions]
    # cv2.drawContours(this_image,c,

cv2.imshow('Test', this_image)
cv2.waitKey(0)

dims = dims[1:]
