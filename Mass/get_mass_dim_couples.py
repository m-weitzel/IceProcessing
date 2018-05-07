""" Interactive ice crystal detection script that shows the user an ice image (from folder) with adjustable sliders for threshold
and dilation properties. Subsequently, corresponding drop image is shown and matching of drop to crystl can be adjusted.
Couple images are saved after every image. mass_dim_data.dat and graph.png are only saved if, after the last couple,
saving flag is confirmed by user input."""


import os
import pickle
from copy import deepcopy

import cv2
import numpy as np
from utilities.IceSizing import MicroImg
import Mass.find_couples as find_couples
from matplotlib import pyplot as plt


def main():
    folder = '/uni-mainz.de/homes/maweitze/CCR/01Mar/'
    file_list = os.listdir(folder)

    ice_file_list = list()
    drop_file_list = list()

    for filename in file_list:
        if 'Ice' in filename:
            ice_file_list.append(filename)
        elif 'withCircles' in filename:
            drop_file_list.append(filename)

    assert (len(ice_file_list) > 0), 'No files found.'
    assert (len(drop_file_list) > 0), 'No drop files found.'

    ice_file_list.sort()
    drop_file_list.sort()

    crystal_list = list()
    remove_list = list()

    try:
        tmp = pickle.load(open(folder+'mass_dim_data.dat', 'rb'))
        if len(tmp) == 6 or len(tmp) == 5:
            (x_shift_global_list, y_shift_global_list, _, _, _) = tmp
            remove_global_list = [list() for f in ice_file_list]
        # elif len(tmp) == 3:
        else:
            x_shift_global_list = tmp['x_shift']
            y_shift_global_list = tmp['y_shift']
            try:
                remove_global_list = tmp['remove']
            except KeyError:
                remove_global_list = [list() for f in ice_file_list]
        # else:
        #     print(len(tmp))

    except (FileNotFoundError, EOFError):
        print('No old data file found, starting from scratch.')
        x_shift_global_list = [0]*len(ice_file_list)
        y_shift_global_list = [0]*len(ice_file_list)
        remove_global_list = [list() for f in ice_file_list]

    for ice_file, drop_file, x_shift, y_shift, remove, i in \
            zip(ice_file_list, drop_file_list, x_shift_global_list, y_shift_global_list, remove_global_list, np.arange(1, len(ice_file_list)+1)):

        window_title = ('Comparison' + str(abs(int(ice_file[-6:-4]))))
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 768, 768)

        start_val_thresh = 63
        cv2.createTrackbar('AdaptiveWindow', window_title, start_val_thresh, 1000, nothing)
        cv2.createTrackbar('Dilation', window_title, 30, 100, nothing)
        while 1:
            adaptive_window = cv2.getTrackbarPos('AdaptiveWindow', window_title)
            dilation = cv2.getTrackbarPos('Dilation', window_title)
            # img_ice = MicroImg('Ice', folder, ice_file, ('Adaptive', 2*adaptive_window+1), 750, 100000, dilation)
            # img_ice = MicroImg('Ice', folder, ice_file, ('Bin', 2*adaptive_window+1), 750, 100000, dilation)
            img_ice = MicroImg('Ice', folder, ice_file, ('Gradient', 0), maxsize=np.inf, dilation=dilation, fill_flag=False, min_dist_to_edge=0)
            cv2.imshow(window_title, img_ice.processed_image)
            k = cv2.waitKey(1500) & 0xFF  # Set refresh time for plot to 5 ms
            if (k == 13) | (k == 10):  # Arbitrary end loop time
                break
            # else:
                # print(k)

        img_drop = MicroImg('Drop', folder, drop_file, ('Color', 0), 750, 100000)

        dims_ice_list = [x['Center Points'] for x in img_ice.data]
        dims_drops_list = [x['Center Points'] for x in img_drop.data]

        editing_flag = True

        while editing_flag:

            cv2.createTrackbar('X Shift', window_title, 1000+int(x_shift), 2000, nothing)

            while 1:
                img_comparison = img_ice.initial_image.copy()
                preview_img_contours = deepcopy(img_drop.contours)

                img_comparison = shift_contours(img_comparison, preview_img_contours, x_shift, y_shift)
                for c in img_ice.contours:
                    cv2.drawContours(img_comparison, c, -1, (0, 255, 0), 2)

                cv2.imshow(window_title, img_comparison)

                x_shift_list = list()
                pairs_list = list()

                x_shift = -1000+cv2.getTrackbarPos('X Shift', window_title)
                k = cv2.waitKey(500) & 0xFF  # Set refresh time for plot to 5 ms
                if (k == 13) | (k == 10):  # Arbitrary end loop time
                    break
                # else:
                #     print(k)

            # this_input = input('Keep '+str(x_shift)+', '+str(y_shift)+' as xshift, yshift or enter new (<-Drop: neg., Drop->: pos.).')
            # try:
            #     x_shift = int(this_input)
            #     try:
            #         y_shift = int(input('yshift:'))
            #     except ValueError:
            #         print('Invalid or empty input, keeping old yshift ' + str(y_shift) + '.')
            # except ValueError:
            #     print('Invalid or empty input, keeping old xshift '+str(x_shift)+', '+str(y_shift)+'.')

            # Actual call to finding matching drop for all crystals in current image
            for crystal in dims_ice_list:
                nearest_drop = find_couples.find_closest_drop(crystal, dims_drops_list, x_shift, y_shift, 150)
                if nearest_drop:
                    x_shift_list.append(crystal[0] - nearest_drop[0])
                    pairs_list.append((crystal, nearest_drop))
                    if len(x_shift_list) > 4:
                        x_shift = np.median(x_shift_list)

            print('Saving '+str(x_shift)+' as xshift, '+str(y_shift)+' as yshift.')

            img_comparison = img_ice.initial_image.copy()

            this_crystal_list = list()
            new_info_list = list()

            for crystal in img_ice.data:
                try:
                    k = [x[0] for x in pairs_list].index(crystal['Center Points'])
                    new_info = {}
                    for drop in img_drop.data:
                        if drop['Center Points'] == pairs_list[k][1]:
                            new_info = {'Drop Diameter': drop['Short Axis']}
                    new_info_list.append(new_info)
                    this_crystal_list.append(crystal)

                except ValueError:
                    print('No matching drop for crystal at '+str(crystal['Center Points'])+' found.')

            for c in img_ice.contours:
                cv2.drawContours(img_comparison, c, -1, (0, 255, 0), 2)

            preview_img_contours = deepcopy(img_drop.contours)
            img_comparison = shift_contours(img_comparison, preview_img_contours, x_shift, y_shift)

            img_preview = img_comparison.copy()

            for p, pair in enumerate(pairs_list):
                if pair[1]:
                    cv2.circle(img_preview, (int(pair[0][0]), int(pair[0][1])), 7, (0, 255, 0), -1)
                    cv2.circle(img_preview, (int(pair[1][0]+x_shift), int(pair[1][1]+y_shift)), 7, (255, 0, 0), -1)
                    cv2.line(img_preview, (int(pair[0][0]), int(pair[0][1])), (int(pair[1][0]+x_shift), int(pair[1][1]+y_shift)), (255, 255, 255))
                    cv2.putText(img_preview, str(p), (int(pair[0][0]), int(pair[0][1])), cv2.FONT_HERSHEY_SIMPLEX,
                                3, (255, 255, 255), 2)

            cv2.imshow(('Comparison'+str(abs(int(ice_file[-6:-4])))), img_preview)
            cv2.waitKey(1000)

            input_remove = input('Old removes: ['+"".join(str(remove))+'] - Enter new list of objects to remove or keep.')

            if input_remove:
                try:
                    this_remove_list = list(map(int, input_remove.split(',')))
                    this_remove_list.sort(reverse=True)
                except ValueError:
                    print('Input incorrect, using old/empty removes.')
                    this_remove_list = remove
            else:
                this_remove_list = remove

            for removable in this_remove_list:
                pairs_list.pop(removable)
                this_crystal_list.pop(removable)
                new_info_list.pop(removable)

            for p, pair in enumerate(pairs_list):
                if pair[1]:
                    cv2.circle(img_comparison, (int(pair[0][0]), int(pair[0][1])), 7, (0, 255, 0), -1)
                    cv2.circle(img_comparison, (int(pair[1][0] + x_shift), int(pair[1][1] + y_shift)), 7, (255, 0, 0), -1)
                    cv2.line(img_comparison, (int(pair[0][0]), int(pair[0][1])),
                             (int(pair[1][0] + x_shift), int(pair[1][1] + y_shift)), (255, 255, 255))

            cv2.imshow(('Comparison' + str(abs(int(ice_file[-6:-4])))), img_comparison)
            cv2.waitKey(1000)

            this_input = input('Save image and data as displayed?')
            if this_input=='Yes' or this_input=='yes' or this_input=='y':
                print('Saving.')
                editing_flag=False
            else:
                print('Not accepting, starting over...')

        for crystal, new_info in zip(this_crystal_list, new_info_list):
            crystal.update(new_info)

        for c in img_drop.contours:
            c[:, :, 0] += int(x_shift)
            c[:, :, 1] += int(y_shift)

        x_shift_global_list[i - 1] = x_shift
        y_shift_global_list[i - 1] = y_shift

        remove_list += [this_remove_list]
        crystal_list += this_crystal_list

        cv2.imwrite((folder+'/IDCouple-'+str(abs(int(ice_file[-6:-4])))+'.png'), img_comparison)

        cv2.destroyWindow('Comparison'+str(abs(int(ice_file[-6:-4]))))

    plt.scatter([x['Long Axis'] for x in crystal_list], [np.pi/6*x['Drop Diameter']**3 for x in crystal_list])
    # plt.xlim((0, 1.1*np.max(dim_list)))
    # plt.ylim((0, 1.1*np.max(mass_list)))

    save_flag = input('Save data to file?')
    if save_flag == 'Yes' or save_flag == 'yes':

        save_dict = {"x_shift": x_shift_global_list, "y_shift": y_shift_global_list, 'remove': remove_list, "crystal": crystal_list}
        pickle.dump(save_dict, open(folder + 'mass_dim_data.dat', 'wb'))
        print('Data saved in '+folder+'mass_dim_data.dat.')

        plt.savefig(folder + 'plots/graph.png')
        print('Graph saved in ' + folder + 'plots/mass_graph.png.')

    plt.show()


def nothing(x):
    pass


def shift_contours(img, contours, x_shift, y_shift):
    for c in contours:
        c[:, :, 0] += int(x_shift)
        c[:, :, 1] += int(y_shift)
        cv2.drawContours(img, c, -1, (255, 0, 0), 2)
    return img


if __name__ == "__main__":
    main()