from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import os


def midpoint(pt_a, pt_b):
    return (pt_a[0] + pt_b[0]) * 0.5, (pt_a[1] + pt_b[1]) * 0.5

# construct the argument parse and parse the argume nts
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to the input image")
# ap.add_argument("-w", "--width", type=float, required=True,
# 	help="width of the left-most object in the image (in inches)")
# args = vars(ap.parse_args())

cnt_area = np.zeros(0)
Ls = cnt_area.copy()
Ws = cnt_area.copy()
curr_folder = 'img/ice/3103M1/'

img_array = list()

for img_path in os.listdir(curr_folder):

    if img_path[-3:] != "png":
        continue

    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread(curr_folder + img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges

    threshold = 127
    canny_low = 8
    thresh = cv2.Canny(gray, canny_low, canny_low*3); method = 'Canny'
    # rt,thresh = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY_INV)
    # rt,thresh = cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU); method='InvOtsu'
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 767, 7); method='adaptiveGaussianInv'

    dilation = 20
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    #
    # find contours in the edge map
    #
    # cv2.namedWindow(img_path+"_thresh",cv2.WINDOW_NORMAL)
    # cv2.imshow(img_path+"_thresh",thresh)
    # cv2.resizeWindow(img_path, 768,768)

    # cv2.imshow(img_path+"_thresh",thresh)

    # cv2.waitKey(0)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    cv2.namedWindow(img_path, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(img_path, 768, 768)

    # cv2.imshow("imwin", image)

    orig = image.copy()

    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 750:
            continue

        # compute the rotated bounding box of the contour

        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        if (box > 2040).any() or (box < 8).any():
           continue

        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)

        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        img_box = (min(box[:, 0]), max(box[:, 0]), min(box[:, 1]), max(box[:, 1]))
        partial_img = image[int(img_box[0]):int(img_box[1]), int(img_box[2]):int(img_box[3])]

        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        L = np.max([dA, dB])
        W = np.min([dA, dB])

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            # pixelsPerMetric = dB / args["width"]
            pixelsPerMetric = 3

        # compute the size of the object
        dimL = L / pixelsPerMetric
        dimW = W / pixelsPerMetric
        area = cv2.contourArea(c) / (pixelsPerMetric ** 2)

        Ls = np.append(Ls, dimL)
        Ws = np.append(Ws, dimW)
        cnt_area = np.append(cnt_area, area)

        # draw the object sizes on the image
        cv2.putText(orig, "{:.1f}um".format(dA / pixelsPerMetric),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}um".format(dB / pixelsPerMetric),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        # show the output image

        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.drawContours(orig, c, -1, (255, 0, 0), 2)
        # cv2.imshow(img_path, orig)
        # cv2.waitKey(0)

    cv2.namedWindow(img_path+"_2", cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow('imwin', 720, 720)

    cv2.imshow(img_path, orig)
    # img_array.append(orig)
# cv2.namedWindow('HistArea', cv2.WINDOW_AUTOSIZE)
# plt.subplot(2,3,1)
# plt.hist(cnt_area, histtype='bar', ec='black')
# plt.title('Contour Area')
# plt.subplot(2,3,2)
# plt.hist(Ls, histtype='bar', ec='black')
# plt.title('A')
# plt.subplot(2,3,3)
# plt.hist(Ws, histtype='bar', ec='black')
# plt.title('B')
# plt.subplot(2,3,4)
# plt.hist(Ls/Ws, histtype='bar', ec='black')
# plt.title('Aspect Ratio')
# plt.subplot(2,3,5)
# plt.hist(cnt_area/(Ls*Ws), ec='black')
# plt.title('Normalized Area')
#
# plt.suptitle('Properties of '+str(len(os.listdir('img/ice')))+' files with threshold '+str(threshold)+' and method '+str(method)+' and dilation of'+str(dilation))
# plt.show()

cv2.waitKey(0)
