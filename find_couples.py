import cv2
import numpy as np
# from IceSizing import MicroImg


class IceDropCouple:
    def __init__(self, ice_cent, drop_cent):
        self.ice_center = ice_cent
        # self.ice_contour = ice_cont
        self.drop_center = drop_cent
        # self.drop_contour = drop_cont

        self.x_dist, self.y_dist, self.ice_drop_dist = self.get_distances()

    def get_distances(self):
        if self.drop_center:
            x_dist = self.ice_center[0]-self.drop_center[0]
            y_dist = self.ice_center[1]-self.drop_center[1]
            ice_drop_dist = np.sqrt(x_dist**2+y_dist**2)
        else:
            x_dist = None
            y_dist = None
            ice_drop_dist = None

        return x_dist, y_dist, ice_drop_dist


def find_closest_drop(crystal, drop_list, x_shift=-200, max_y_dist=200):
    if (crystal[0]-x_shift) > 2048:
        return None
    else:

        x_dist_list = list()
        y_dist_list = list()

        for inner_drop in drop_list:
            current_x_shift = inner_drop[0] - crystal[0]
            current_y_dist = abs(inner_drop[1] - crystal[1])

            x_dist_list.append(current_x_shift)
            y_dist_list.append(current_y_dist)

        list_pairs = list(zip([x + x_shift for x in x_dist_list], y_dist_list, drop_list))
        list_pairs.sort(key=lambda x: x[1])
        # result = (list_pairs[0])
        list_shortened = list_pairs[:4]
        list_shortened.sort(key=lambda x: np.sqrt([x[0]**2+x[1]**2]))
        result = list_shortened[0]

        if abs(result[1]) < max_y_dist:
            return result[2]
        else:
            return None


def main(ice_contours, drop_contours, x_shift_guess=-200, maxdist=500):

    ice_crystal_list = list()
    drop_list = list()
    x_shift_list = list()
    pairs_list = list()

    for c in ice_contours:
        if cv2.contourArea(c) > 1000:
            m = cv2.moments(c)
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])

            ice_crystal = (cx, cy)

            ice_crystal_list.append(ice_crystal)
            # ice_objs.append(this_co)

            # cv2.circle(img_comparison, ice_crystal, 7, (0, 255, 0), -1)
            # cv2.drawContours(img_comparison, [c], -1, (0, 255, 0), 2)
            # cv2.putText(img_comparison, "{:.1f}um".format(int(cy)),
            #            (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
            #            0.65, (255, 255, 255), 2)

    for c in drop_contours:
        if cv2.contourArea(c) > 1000:
            m = cv2.moments(c)
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])

            drop = (cx, cy)

            drop_list.append(drop)

            # cv2.circle(img_comparison, drop, 7, (0, 255, 0), -1)
            # cv2.drawContours(img_comparison, [c], -1, (0, 0, 255), 2)

# First run for getting x shift
    for crystal in ice_crystal_list:
        if len(x_shift_list) > 4:
            closest_drop = find_closest_drop(crystal, drop_list, np.median(x_shift_list), maxdist)
        else:
            closest_drop = find_closest_drop(crystal, drop_list, x_shift_guess, maxdist)
        if closest_drop:
            x_shift_list.append(crystal[0]-closest_drop[0])

# Second run for finding couples
    for crystal in ice_crystal_list:
        closest_drop = find_closest_drop(crystal, drop_list, np.median(x_shift_list), maxdist)

        if closest_drop:
            pairs_list.append(IceDropCouple(crystal, closest_drop))

    # for pair in pairs_list:
    #     if pair.drop_center:
    #         cv2.circle(img_comparison, pair.ice_center, 7, (0, 255, 0), -1)
    #         cv2.circle(img_comparison, pair.drop_center, 7, (255, 0, 0), -1)
    #         cv2.line(img_comparison, pair.drop_center, pair.ice_center, (255, 255, 255))
    return pairs_list

# if __name__ == "__main__":
#     main()