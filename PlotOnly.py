from matplotlib import pyplot as plt
import os
import numpy as np
import pickle

full_folder = os.listdir('result_binaries')
folder_list = list(filter(lambda c: (c[-3] != ".") & (c[0] != ".") & (c != "img"), full_folder))

folder_count = 1
folder_num = list()
folder_avg_area = list()
folder_avg_Ws = list()
folder_avg_aspr = list()
folder_avg_area_ratio = list()

label_array = ['Canny', 'Otsu', 'Adaptive']

for folder_path in folder_list:
    (dims_bin, dims_otsu, dims_canny, dims_adapt) = pickle.load(open('result_binaries/'+folder_path, "rb"))

    plt.figure(folder_count)

    plt.subplot(2, 3, 1)
    plt.hist([dims_canny[2], dims_otsu[2], dims_adapt[2]], histtype='bar', ec='black', label=label_array, alpha=0.5)
    plt.title('Contour Area')
    plt.legend(loc='upper right')

    plt.subplot(2, 3, 2)
    plt.hist([dims_canny[0], dims_otsu[0], dims_adapt[0]], histtype='bar', ec='black', label=label_array, alpha=0.5)
    plt.title('Major Axis')
    plt.legend(loc='upper right')

    plt.subplot(2, 3, 3)
    plt.hist([dims_canny[1], dims_otsu[1], dims_adapt[1]], histtype='bar', ec='black', label=label_array, alpha=0.5)
    plt.title('Minor Axis')
    plt.legend(loc='upper right')

    plt.subplot(2, 3, 4)
    plt.hist([dims_canny[3], dims_otsu[3], dims_adapt[3]], histtype='bar', ec='black', label=label_array, alpha=0.5)
    plt.title('Aspect Ratio')
    plt.legend(loc='upper right')

    plt.subplot(2, 3, 5)
    plt.hist([dims_canny[4], dims_otsu[4], dims_adapt[4]], ec='black', label=label_array, alpha=0.5)
    plt.title('Normalized Area')
    plt.legend(loc='upper left')

    plt.suptitle(folder_path)

    folder_avg_area.append((dims_bin[2].mean(), dims_canny[2].mean(), dims_otsu[2].mean(), dims_adapt[2].mean()))
    folder_avg_area_ratio.append((dims_bin[4].mean(), dims_canny[4].mean(), dims_otsu[4].mean(), dims_adapt[4].mean()))
    folder_avg_aspr.append((dims_bin[3].mean(), dims_canny[3].mean(), dims_otsu[3].mean(), dims_adapt[3].mean()))
    folder_avg_Ws.append((dims_bin[1].mean(), dims_canny[1].mean(), dims_otsu[1].mean(), dims_adapt[1].mean()))
    folder_num.append((len(dims_bin[0]), len(dims_canny[0]), len(dims_otsu[0]), len(dims_adapt[0])))

    folder_count += 1

figure = plt.figure(folder_count)

labels_filter = ['Binary', 'Canny', 'Otsu', 'Adaptive']
x = np.arange(0, len(labels_filter), 1)
folder_avg_area = list(map(list, zip(*folder_avg_area)))
folder_avg_area_ratio = list(map(list, zip(*folder_avg_area_ratio)))
folder_avg_aspr = list(map(list, zip(*folder_avg_aspr)))
folder_avg_Ws = list(map(list, zip(*folder_avg_Ws)))
folder_num = list(map(list, zip(*folder_num)))


subplot1 = figure.add_subplot(2, 3, 1)
for nums, label_flt in zip(folder_num, labels_filter):
    subplot1.plot(nums, 'o-', label=label_flt)
subplot1.set_xticks(x)
subplot1.set_xticklabels(folder_list)
subplot1.set_title('Number of objects')
plt.legend()

subplot2 = figure.add_subplot(2, 3, 2)
for avg_Ws, label_flt in zip(folder_avg_Ws, labels_filter):
    subplot2.plot(avg_Ws, 'o-', label=label_flt)
subplot2.set_xticks(x)
subplot2.set_xticklabels(folder_list)
subplot2.set_title('Average major axis')
plt.legend()

subplot3 = figure.add_subplot(2, 3, 3)
for avg_area, label_flt in zip(folder_avg_area, labels_filter):
    subplot3.plot(avg_area, 'o-', label=label_flt)
subplot3.set_xticks(x)
subplot3.set_xticklabels(folder_list)
subplot3.set_title('Average area')
plt.legend()

subplot4 = figure.add_subplot(2, 3, 4)
for avg_aspr, label_flt in zip(folder_avg_aspr, labels_filter):
    subplot4.plot(avg_aspr, 'o-', label=label_flt)
subplot4.set_xticks(x)
subplot4.set_xticklabels(folder_list)
subplot4.set_title('Average aspect ratio')
plt.legend()

subplot5 = figure.add_subplot(2, 3, 5)
for avg_ar, label_flt in zip(folder_avg_area_ratio, labels_filter):
    subplot5.plot(avg_ar, 'o-', label=label_flt)
subplot5.set_xticks(x)
subplot5.set_xticklabels(folder_list)
subplot5.set_title('Average area ratio')
plt.legend()

plt.show()
