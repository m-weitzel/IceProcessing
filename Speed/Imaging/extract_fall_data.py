import os
import sys
sys.path.append('../../utilities')
sys.path.append('../../Mass')
from utilities.IceSizing import MicroImg
import cv2
import pickle
import Mass.find_couples
import numpy as np


def initialize_data(fldr, fldr_list, pixel_size):

    list_of_file_data = list()

    print('No old data file found, starting from scratch.')
    try:
        os.mkdir(os.path.join(fldr, 'Fall/processed'))
    except FileExistsError:
        pass

    streak_filter_cond = 'dim_w > 4 or (np.asarray([b[0] for b in box])>(img.shape[1]-8)).any() \
                          or (np.asarray([b[1] for b in box])>(img.shape[0]-8)).any() or (box < 8).any()'

    for i, filename in enumerate(fldr_list):
        if '_cropped' in filename:
            cont_real = list()
            fall_dist = list()
            orientation = list()
            centerpt = list()
            time_list = list()

            img = MicroImg('Streak', os.path.join(fldr, 'Fall'), filename, pixel_size,
                           thresh_type=('Bin', -150), minsize=75, fill_flag=False, maxsize=50000, dilation=1,
                           optional_object_filter_condition=streak_filter_cond)

            dims = img.data
            conts = img.contours

            for dim, cont in zip(dims, conts):
                # if dim['Short Axis'] < 8:
                cont_real.append(cont)
                fall_dist.append(dim['Long Axis'])
                orientation.append(dim['Orientation'])
                centerpt.append(dim['Center Points'])
                time_list.append(i)

            img.contours = cont_real
            print('Done processing ' + filename + ', ' + str(i+1) + ' of ' + str(len(fldr_list)) + '.')
            # plt.imshow(img.processed_image)
            cv2.imwrite(os.path.join(fldr, 'Fall', 'processed', filename+'_processed.png'), img.processed_image)
            if fall_dist:
                list_of_file_data.append([i, filename, fall_dist, orientation, centerpt, time_list, list(np.ones(len(centerpt)))])

    # list_of_lists = (cont_real, fall_dist, orientation, centerpt, time_list)
    # pickle.dump(list_of_lists, open(fldr+'fall_speed_data.dat', 'wb'))

    # pickle.dump(list_of_file_data, open(os.path.join(fldr, 'fall_speed_data.dat'), 'wb'))

    print('Number of streaks before condensation: {}.'.format(len(list_of_file_data)))
    list_of_file_data = condense_streaks(list_of_file_data, pixel_size)
    print('Number of streaks after condensation: {}.'.format(len(list_of_file_data)))
    cont_list = list()
    fall_dist = list()
    orientation = list()
    centerpt = list()
    time_list = list()
    chainlength_list = list()

    for file in list_of_file_data:
        cont_list.extend(file[1])
        fall_dist.extend(file[2])
        orientation.extend(file[3])
        centerpt.extend(file[4])
        time_list.extend(file[5])
        chainlength_list.extend(file[6])

    list_of_lists = cont_list, fall_dist, orientation, centerpt, time_list, chainlength_list
    pickle.dump(list_of_lists, open(os.path.join(fldr, 'fall_speed_data.dat'), 'wb'))

    return list_of_lists


def condense_streaks(list_of_file_data, pxl_size):

    for i, (this_file, next_file) in enumerate(zip(list_of_file_data[0:-2], list_of_file_data[1:])):
        if this_file[0] == next_file[0]-1:                                                                                  # if the next file is actually the next file in chronological order
            predicted_ctrs = [(ct[0], ct[1]+fs/pxl_size) for fs, ct in zip(this_file[2], this_file[4])]                         # predict where the p center would be in the next image (frame rate 11.7 fps = 1/85000µs)
            for ctr in predicted_ctrs:                                                                                      # for each of the predicted ctr points try to find a closest
                found_flag = Mass.find_couples.find_closest_drop(ctr, next_file[4], 0, 0, 50)                                   # object in the ctrs of the next image within certain distance
                if found_flag:
                    index_this = predicted_ctrs.index(ctr)                                                                  # if one is found, save index of the current obj in current list
                    index_next = next_file[4].index(found_flag)                                                             # and the index of the found next obj in next list
                    for j in range(2, 6):
                        if j != 4:
                            # print('i={}, j={}, index_this={}, index_next={}'.format(i, j, index_this, index_next))
                            next_file[j][index_next] = np.mean((this_file[j][index_this], next_file[j][index_next]))        # but take mean for all other parameters
                        this_file[j].pop(index_this)
                    predicted_ctrs.pop(index_this)  # and remove the second object to not have duplicates
                    next_file[6][index_next] = this_file[6][index_this]+1
                    this_file[6].pop(index_this)
        else:
            pass
    return list_of_file_data