import numpy as np
from IceSizing import MicroImg
import find_couples
from matplotlib import pyplot as plt
import cv2
import pickle
import os
from copy import deepcopy

folder = '/uni-mainz.de/homes/maweitze/CCR/0604/M1/'
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

try:
    tmp = pickle.load(open(folder+'mass_dim_data.dat', 'rb'))
    if len(tmp) == 5:
        (x_shift_global_list, y_shift_global_list, _, _, _) = tmp
    elif len(tmp) == 3:
        x_shift_global_list = tmp['x_shift']
        y_shift_global_list = tmp['y_shift']
    else:
        print(len(tmp))

except (FileNotFoundError, EOFError):
    print('No old data file found, starting from scratch.')
    x_shift_global_list = [0]*len(ice_file_list)
    y_shift_global_list = [0]*len(ice_file_list)

for ice_file, drop_file, x_shift, y_shift, i in \
        zip(ice_file_list, drop_file_list, x_shift_global_list, y_shift_global_list, np.arange(1, len(ice_file_list)+1)):
    img_ice = MicroImg('Ice', folder, ice_file, ('Adaptive', 1001), 750, 100000)
    img_drop = MicroImg('Drop', folder, drop_file, ('Color', 0), 750, 100000)

    dims_ice_list = [x['Center Points'] for x in img_ice.data]
    dims_drops_list = [x['Center Points'] for x in img_drop.data]

    x_shift_list = list()

    img_comparison = img_ice.initial_image.copy()
    preview_img_contours = deepcopy(img_drop.contours)

    for c in preview_img_contours:
        c[:, :, 0] += int(x_shift)
        c[:, :, 1] += int(y_shift)
        cv2.drawContours(img_comparison, c, -1, (0, 255, 0), 2)

    cv2.namedWindow(('Comparison'+str(abs(int(ice_file[-6:-4])))), cv2.WINDOW_NORMAL)
    cv2.resizeWindow(('Comparison'+str(abs(int(ice_file[-6:-4])))), 768, 768)
    cv2.imshow(('Comparison' + str(abs(int(ice_file[-6:-4])))), img_comparison)
    cv2.waitKey(1000)

    pairs_list = list()

    this_input = input('Keep '+str(x_shift)+', '+str(y_shift)+' as xshift, yshift? If no, enter new xshift (Drop to left: negative, Drops to right: positive).')
    try:
        x_shift = int(this_input)
        try:
            y_shift = int(input('yshift:'))
        except ValueError:
            print('Invalid or empty input, keeping old yshift ' + str(y_shift) + '.')
    except ValueError:
        print('Invalid or empty input, keeping old xshift '+str(x_shift)+', '+str(y_shift)+'.')

    # Actual call to finding matching drop for all crystals in current image
    for crystal in dims_ice_list:
        nearest_drop = find_couples.find_closest_drop(crystal, dims_drops_list, x_shift, y_shift, 150)
        if nearest_drop:
            x_shift_list.append(crystal[0] - nearest_drop[0])
            pairs_list.append((crystal, nearest_drop))
            if len(x_shift_list) > 4:
                x_shift = np.median(x_shift_list)

    x_shift_global_list[i-1] = x_shift
    y_shift_global_list[i-1] = y_shift
    print('Saving '+str(x_shift)+' as xshift, '+str(y_shift)+' as yshift.')

    img_comparison = img_ice.initial_image.copy()

    this_crystal_list = list()

    for crystal in img_ice.data:
        try:
            k = [x[0] for x in pairs_list].index(crystal['Center Points'])
            for drop in img_drop.data:
                if drop['Center Points'] == pairs_list[k][1]:
                    new_info = {'Drop Diameter': drop['Short Axis']}
            crystal.update(new_info)
            this_crystal_list.append(crystal)
        except ValueError:
            print('No matching drop for crystal at '+str(crystal['Center Points'])+' found.')

    for c in img_ice.contours:
        cv2.drawContours(img_comparison, c, -1, (0, 255, 0), 2)

    for c in img_drop.contours:
        c[:, :, 0] += int(x_shift)
        c[:, :, 1] += int(y_shift)
        cv2.drawContours(img_comparison, c, -1, (255, 0, 0), 2)

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

    input_remove = input('Enter objects to remove.')

    if input_remove:
        remove_list = list(map(int, input_remove.split(',')))
        remove_list.sort(reverse=True)

        for removable in remove_list:
            pairs_list.pop(removable)
            this_crystal_list.pop(removable)

    crystal_list += this_crystal_list

    for p, pair in enumerate(pairs_list):
        if pair[1]:
            cv2.circle(img_comparison, (int(pair[0][0]), int(pair[0][1])), 7, (0, 255, 0), -1)
            cv2.circle(img_comparison, (int(pair[1][0] + x_shift), int(pair[1][1] + y_shift)), 7, (255, 0, 0), -1)
            cv2.line(img_comparison, (int(pair[0][0]), int(pair[0][1])),
                     (int(pair[1][0] + x_shift), int(pair[1][1] + y_shift)), (255, 255, 255))

    cv2.imshow(('Comparison' + str(abs(int(ice_file[-6:-4])))), img_comparison)
    cv2.waitKey(1000)

    cv2.imwrite((folder+'/IDCouple-'+str(abs(int(ice_file[-6:-4])))+'.png'), img_comparison)

    cv2.destroyWindow('Comparison'+str(abs(int(ice_file[-6:-4]))))

plt.scatter([x['Long Axis'] for x in crystal_list], [np.pi/6*x['Drop Diameter']**3 for x in crystal_list])
# plt.xlim((0, 1.1*np.max(dim_list)))
# plt.ylim((0, 1.1*np.max(mass_list)))

save_flag = input('Save data?')
if save_flag == 'Yes' or save_flag == 'yes':

    save_dict = {"x_shift": x_shift_global_list, "y_shift": y_shift_global_list, "crystal": crystal_list}
    pickle.dump(save_dict, open(folder + 'mass_dim_data.dat', 'wb'))
    print('Data saved in '+folder+'mass_dim_data.dat.')

    plt.savefig(folder + 'graph.png')
    print('Graph saved in ' + folder + 'graph.png.')

plt.show()

