import os
import sys
sys.path.append('../utilities')
sys.path.append('../Mass')
from IceSizing import MicroImg
import cv2
import pickle
import find_couples
import numpy as np

def initialize_data(fldr, fldr_list):

    list_of_file_data = list()

    print('No old data file found, starting from scratch.')
    try:
        os.mkdir(fldr+'Fall/processed')
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

            img = MicroImg('Streak', fldr+'Fall', filename,
                           thresh_type=('Bin', -180), minsize=75, maxsize=10000, dilation=1, optional_object_filter_condition=streak_filter_cond)

            dims = img.data
            conts = img.contours



            for dim, cont in zip(dims, conts):
                # if dim['Short Axis'] < 8:
                cont_real.append(cont)
                fall_dist.append(dim['Long Axis'])
                orientation.append(dim['Orientation'])
                centerpt.append(dim['Center Points'])
                # time_list.append([i])

            img.contours = cont_real
            print('Done processing ' + filename + ', ' + str(i+1) + ' of ' + str(len(fldr_list)) + '.')
            # plt.imshow(img.processed_image)
            cv2.imwrite(fldr+'Fall/processed/'+filename+'_processed.png', img.processed_image)
            if fall_dist:
                list_of_file_data.append([i, filename, fall_dist, orientation, centerpt, np.ones(len(fall_dist))])

    # list_of_lists = (cont_real, fall_dist, orientation, centerpt, time_list)
    # pickle.dump(list_of_lists, open(fldr+'fall_speed_data.dat', 'wb'))

    pickle.dump(list_of_file_data, open(fldr+'fall_streak_data.dat', 'wb'))

    list_of_file_data = condense_streaks(list_of_file_data)

    # return fall_dist, orientation, centerpt, time_list
    return list_of_file_data


def condense_streaks(list_of_file_data):

    for this_file, next_file in zip(list_of_file_data[0:-2], list_of_file_data[1:]):
        if this_file[0] == next_file[0]-1:
            predicted_ctrs = [(ct[0], ct[1]+fs*3) for fs, ct in zip(this_file[2], this_file[4])]
            for ctr in predicted_ctrs:
                found_flag = find_couples.find_closest_drop(ctr, next_file[4], 0, 0, 200)
                if found_flag:
                    index_this = predicted_ctrs.index(ctr)
                    index_next = next_file[4].index(found_flag)
                    for i in range(2,5):
                        this_file[i][index_this] = np.mean((this_file[i][index_this], next_file[i][index_next]))
                        next_file[i].pop(index_next)
                    this_file[5][index_this] += 1
        else:
            pass

    return list_of_file_data