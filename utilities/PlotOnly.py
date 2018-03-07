from matplotlib import pyplot as plt
import os
import numpy as np
import pickle


full_folder = sorted(os.listdir('result_binaries'))
folder_list = list(filter(lambda c: (c[-3] != ".") & (c[0] != ".") & (c != "img"), full_folder))


folder_count = 1
folder_num = list()
folder_avg_area = list()
folder_mdn_area = list()
folder_std_area = list()

folder_avg_Ws = list()
folder_mdn_Ws = list()
folder_std_Ws = list()

folder_avg_aspr = list()
folder_mdn_aspr = list()
folder_std_aspr = list()

folder_avg_area_ratio = list()
folder_mdn_area_ratio = list()
folder_std_area_ratio = list()

label_array = ['Canny', 'Otsu', 'Adaptive']

for folder_path in folder_list:
    (dims_bin, dims_otsu, dims_canny, dims_adapt) = pickle.load(open('result_binaries/'+folder_path, "rb"))

    # plt.figure(folder_count)
    #
    # plt.subplot(2, 3, 1)
    # plt.hist([dims_canny[2], dims_otsu[2], dims_adapt[2]], histtype='bar', ec='black', label=label_array, alpha=0.5)
    # plt.title('Contour Area')
    # plt.legend(loc='upper right')
    #
    # plt.subplot(2, 3, 2)
    # plt.hist([dims_canny[0], dims_otsu[0], dims_adapt[0]], histtype='bar', ec='black', label=label_array, alpha=0.5)
    # plt.title('Major Axis')
    # plt.legend(loc='upper right')
    #
    # plt.subplot(2, 3, 3)
    # plt.hist([dims_canny[1], dims_otsu[1], dims_adapt[1]], histtype='bar', ec='black', label=label_array, alpha=0.5)
    # plt.title('Minor Axis')
    # plt.legend(loc='upper right')
    #
    # plt.subplot(2, 3, 4)
    # plt.hist([dims_canny[3], dims_otsu[3], dims_adapt[3]], histtype='bar', ec='black', label=label_array, alpha=0.5)
    # plt.title('Aspect Ratio')
    # plt.legend(loc='upper right')
    #
    # plt.subplot(2, 3, 5)
    # plt.hist([dims_canny[4], dims_otsu[4], dims_adapt[4]], ec='black', label=label_array, alpha=0.5)
    # plt.title('Normalized Area')
    # plt.legend(loc='upper left')
    #
    # plt.suptitle(folder_path)

    folder_num.append((len(dims_bin[0]), len(dims_canny[0]), len(dims_otsu[0]), len(dims_adapt[0])))

    folder_avg_area.append((dims_bin[2].mean(), dims_canny[2].mean(), dims_otsu[2].mean(), dims_adapt[2].mean()))
    folder_mdn_area.append((np.median(dims_bin[2]), np.median(dims_canny[2]), np.median(dims_otsu[2]), np.median(dims_adapt[2])))
    folder_std_area.append((np.std(dims_bin[2]), np.std(dims_canny[2]), np.std(dims_otsu[2]), np.std(dims_adapt[2])))

    folder_avg_area_ratio.append((dims_bin[4].mean(), dims_canny[4].mean(), dims_otsu[4].mean(), dims_adapt[4].mean()))
    folder_mdn_area_ratio.append((np.median(dims_bin[4]), np.median(dims_canny[4]), np.median(dims_otsu[4]), np.median(dims_adapt[4])))
    folder_std_area_ratio.append((np.std(dims_bin[4]), np.std(dims_canny[4]), np.std(dims_otsu[4]), np.std(dims_adapt[4])))

    folder_avg_aspr.append((dims_bin[3].mean(), dims_canny[3].mean(), dims_otsu[3].mean(), dims_adapt[3].mean()))
    folder_mdn_aspr.append((np.median(dims_bin[3]), np.median(dims_canny[3]), np.median(dims_otsu[3]), np.median(dims_adapt[3])))
    folder_std_aspr.append((np.std(dims_bin[3]), np.std(dims_canny[3]), np.std(dims_otsu[3]), np.std(dims_adapt[3])))

    folder_avg_Ws.append((dims_bin[1].mean(), dims_canny[1].mean(), dims_otsu[1].mean(), dims_adapt[1].mean()))
    folder_mdn_Ws.append((np.median(dims_bin[1]), np.median(dims_canny[1]), np.median(dims_otsu[1]), np.median(dims_adapt[1])))
    folder_std_Ws.append((np.std(dims_bin[1]), np.std(dims_canny[1]), np.std(dims_otsu[1]), np.std(dims_adapt[1])))

    folder_count += 1

figure = plt.figure(folder_count)

labels_filter = ['Binary', 'Canny', 'Otsu', 'Adaptive']
x = np.arange(0, len(folder_list), 1)
folder_avg_area = list(map(list, zip(*folder_avg_area)))
folder_mdn_area = list(map(list, zip(*folder_mdn_area)))
folder_std_area = list(map(list, zip(*folder_std_area)))

folder_avg_area_ratio = list(map(list, zip(*folder_avg_area_ratio)))
folder_mdn_area_ratio = list(map(list, zip(*folder_mdn_area_ratio)))
folder_std_area_ratio = list(map(list, zip(*folder_std_area_ratio)))

folder_avg_aspr = list(map(list, zip(*folder_avg_aspr)))
folder_mdn_aspr = list(map(list, zip(*folder_mdn_aspr)))
folder_std_aspr = list(map(list, zip(*folder_std_aspr)))

folder_avg_Ws = list(map(list, zip(*folder_avg_Ws)))
folder_mdn_Ws = list(map(list, zip(*folder_mdn_Ws)))
folder_std_Ws = list(map(list, zip(*folder_std_Ws)))

folder_num = list(map(list, zip(*folder_num)))


cmap = plt.get_cmap('jet')
col_mod = cmap(np.arange(1, len(labels_filter)+1)/len(labels_filter))


subplot1 = figure.add_subplot(2, 3, 1)
for col_idx, (nums, label_flt) in enumerate(zip(folder_num, labels_filter)):
    subplot1.plot(nums, 'o-', c=col_mod[col_idx],  label=label_flt)
subplot1.set_xticks(x)
subplot1.set_xticklabels(folder_list)
subplot1.set_ylim([0, 350])
subplot1.set_title('Number of objects')
plt.legend()

subplot2 = figure.add_subplot(2, 3, 2)
for col_idx, (mdn_Ws, avg_Ws, label_flt) in enumerate(zip(folder_mdn_Ws, folder_avg_Ws, labels_filter)):
    subplot2.plot(avg_Ws, 'o-', c=col_mod[col_idx], label=label_flt)
    subplot2.plot(mdn_Ws, '*--', c=col_mod[col_idx])
    #subplot2.plot(list(np.asarray(mdn_Ws)+np.asarray(std_Ws)), '*--', c=col_mod[col_idx])
    #subplot2.plot(list(np.asarray(mdn_Ws)-np.asarray(std_Ws)), '*--', c=col_mod[col_idx])
subplot2.set_xticks(x)
subplot2.set_xticklabels(folder_list)
subplot2.set_ylim([0, 55])
subplot2.set_title('Average major axis')
plt.legend()

subplot3 = figure.add_subplot(2, 3, 3)
for col_idx, (avg_area, mdn_area, label_flt) in enumerate(zip(folder_avg_area, folder_mdn_area, labels_filter)):
    subplot3.plot(avg_area, 'o-', c=col_mod[col_idx], label=label_flt)
    subplot3.plot(mdn_area, '*--', c=col_mod[col_idx])
subplot3.set_xticks(x)
subplot3.set_xticklabels(folder_list)
subplot3.set_ylim([250, 3500])
subplot3.set_title('Average area')
plt.legend()

subplot4 = figure.add_subplot(2, 3, 4)
for col_idx, (avg_aspr, mdn_aspr, label_flt) in enumerate(zip(folder_avg_aspr, folder_mdn_aspr, labels_filter)):
    subplot4.plot(avg_aspr, 'o-', c=col_mod[col_idx], label=label_flt)
    subplot4.plot(mdn_aspr, '*--', c=col_mod[col_idx])
subplot4.set_xticks(x)
subplot4.set_xticklabels(folder_list)
subplot4.set_ylim([1, 2.25])
subplot4.set_title('Average aspect ratio')
plt.legend()

subplot5 = figure.add_subplot(2, 3, 5)
for col_idx, (avg_ar, mdn_ar, label_flt) in enumerate(zip(folder_avg_area_ratio, folder_mdn_area_ratio, labels_filter)):
    subplot5.plot(avg_ar, 'o-', c=col_mod[col_idx], label=label_flt)
    subplot5.plot(mdn_ar, '*--', c=col_mod[col_idx])
subplot5.set_xticks(x)
subplot5.set_xticklabels(folder_list)
subplot5.set_ylim([0.4, 0.8])
subplot5.set_title('Average area ratio')
plt.legend()

plt.show()
