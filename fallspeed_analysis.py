import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.ticker as tck
from IceSizing import MicroImg
import os
import numpy as np
import pickle
from scipy.stats import norm
import cv2

folder = '/uni-mainz.de/homes/maweitze/CCR/0808/M1/'
pixel_size = 23.03  # in µm
exposure_time = 85000  # in µs

histogram_plt_flag = 0
orientation_polar_flag = 0
v_t_series_flag = 0
ori_scatter_flag = 1
centerpt_density_flag = 1


def main(fldr, pxl_size, exp_time, h_flag=1, op_flag=1, vt_flag=1, or_flag=1, dn_flag=1):

    fall_folder = fldr+'Fall'
    folder_list = sorted(os.listdir(fall_folder))
    folder_list = [f for f in folder_list if '.png' in f]

    try:
        fall_dist, orientation, centerpt, time_list = load_v_data(fldr)
    except FileNotFoundError:
        fall_dist, orientation, centerpt, time_list = initialize_data(fldr, folder_list)
    vs = np.asarray(fall_dist) * pxl_size / exp_time * 100  # in cm/s
    projected_vs = [v * np.cos(o) for (v, o) in zip(vs, orientation)]

    mass_data = load_mass_data(folder)

    if h_flag:
        mass_velocity_dim_histograms(projected_vs, mass_data, folder)
    if op_flag:
        orientation_polar_plot(orientation)
    if vt_flag:
        velocity_time_series(folder_list, time_list, projected_vs)
    if or_flag:
        orientation_scatter(centerpt, orientation)
    if dn_flag:
        centerpt_density(centerpt, orientation, pixel_size)
    plt.show()


def load_v_data(fldr):
    list_of_lists = pickle.load(open(fldr+'fall_speed_data.dat', 'rb'))
    # cont_real = list_of_lists[0]
    fall_dist = list_of_lists[1]
    orientation = list_of_lists[2]
    centerpt = list_of_lists[3]
    time_list = list_of_lists[4]

    return fall_dist, orientation, centerpt, time_list


def initialize_data(fldr, fldr_list):

    cont_real = list()
    fall_dist = list()
    orientation = list()
    centerpt = list()
    time_list = list()

    print('No old data file found, starting from scratch.')
    try:
        os.mkdir(fldr+'Fall/processed')
    except FileExistsError:
        pass

    streak_filter_cond = streak_filter_cond = 'dim_w > 4 or (np.asarray([b[0] for b in box])>(img.shape[1]-8)).any() \
                          or (np.asarray([b[1] for b in box])>(img.shape[0]-8)).any() or (box < 8).any()'

    for i, filename in enumerate(fldr_list):
        if '_cropped' in filename:
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
                time_list.append([i])

            img.contours = cont_real
            print('Done processing ' + filename + ', ' + str(i+1) + ' of ' + str(len(fldr_list)) + '.')
            # plt.imshow(img.processed_image)
            cv2.imwrite(fldr+'Fall/processed/'+filename+'_processed.png', img.processed_image)

    list_of_lists = (cont_real, fall_dist, orientation, centerpt, time_list)
    pickle.dump(list_of_lists, open(fldr+'fall_speed_data.dat', 'wb'))

    return fall_dist, orientation, centerpt, time_list


def load_mass_data(fldr):
    tmp = pickle.load(open(fldr + 'mass_dim_data.dat', 'rb'))

    area_eq_diam_list = list()
    max_diam_list = list()
    mass_list = list()
    dropdiam_list = list()

    for obj in tmp['crystal']:
        area_eq_diam_list.append(2 * np.sqrt(obj['Area'] / np.pi))
        # area_eq_diam_list.append(0.58*obj['Short Axis']/2*(1+0.95*(obj['Long Axis']/obj['Short Axis'])**0.75))
        max_diam_list.append(obj['Long Axis'])
        mass_list.append(np.pi / 6 * obj['Drop Diameter'] ** 3)
        dropdiam_list.append(obj['Drop Diameter'])

    return area_eq_diam_list, max_diam_list, mass_list, dropdiam_list


def mass_velocity_dim_histograms(vs, mass_data, fldr):

    n_bins = 25
    v_max = 3
    ae_max = 75
    mxdim_max = 105
    # mass_max = 87500
    dropdiam_max = 75

    area_eq_diam_list = mass_data[0]
    max_diam_list = mass_data[1]
    dropdiam_list = mass_data[3]

    def plot_param_hist(t_ax, list_of_vals, max_val, t_n_bins, unit):
        bins = max_val/n_bins*np.arange(t_n_bins)
        (mu, sigma) = norm.fit(list_of_vals)
        n, bins, _ = ax.hist(list_of_vals, bins=bins)
        dx = bins[1] - bins[0]
        scale = len(list_of_vals)*dx
        y = mlab.normpdf(bins, mu, sigma) * scale
        t_ax.plot(bins, y, 'g--', linewidth=2)
        t_ax.axvline(x=mu, ymax=max(y)/t_ax.get_ylim()[1], color='r', linewidth=2)
        t_ax.axvline(x=mu-sigma, ymax=max(y)/t_ax.get_ylim()[1], color='r', linewidth=2)
        t_ax.axvline(x=mu+sigma, ymax=max(y)/t_ax.get_ylim()[1], color='r', linewidth=2)
        t_ax.axhline(y=max(y), xmin=(mu-sigma)/t_ax.get_xlim()[1], xmax=(mu+sigma)/t_ax.get_xlim()[1], color='r', linewidth=2)

        t_ax.text(0.95*float(t_ax.get_xlim()[1]), 0.95*float(t_ax.get_ylim()[1]), 'Mean:${0:.3f} \pm {1:.3f}$ {2:s}'.format(mu, sigma, unit),
                  bbox=dict(facecolor='red', alpha=0.2), horizontalalignment='right', verticalalignment='top')

    _, axs = plt.subplots(2, 2, figsize=(20, 12.5))
    ax = axs[0][0]
    plot_param_hist(ax, vs, v_max, n_bins, 'cm/s')
    ax.set_title('Terminal Velocity in cm/s')
    ax.set_xlabel('Terminal Velocity in cm/s')
    ax.set_ylabel('Count')

    ax = axs[0][1]
    plot_param_hist(ax, area_eq_diam_list, ae_max, n_bins, '$\mu m$')
    ax.set_title('Area equivalent diameter in um')
    ax.set_xlabel('Area equivalent diameter in um')
    ax.set_ylabel('Count')

    ax = axs[1][1]
    plot_param_hist(ax, max_diam_list, mxdim_max, n_bins, '$\mu m$')
    ax.set_title('Maximum diameter in um')
    ax.set_xlabel('Maximum diameter in um')
    ax.set_ylabel('Count')

    ax = axs[1][0]
    plot_param_hist(ax, dropdiam_list, dropdiam_max, n_bins, '$\mu m$')
    ax.set_title('Drop diameter in um')
    ax.set_xlabel('Drop diameter in um')
    ax.set_ylabel('Count')

    plt.suptitle('Histogram Overview for '+fldr[-8:], fontsize=12)

    plt.savefig(fldr + 'histogram.png')


def orientation_polar_plot(orientation):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location("S")
    n_bins = 100
    width = (2*np.pi)/n_bins
    theta = np.linspace(-np.pi+np.pi/n_bins, np.pi+np.pi/n_bins, n_bins, endpoint=False)
    # max_height = 8
    # ori = [np.deg2rad(a) for a in np.sort(orientation) if ~np.isnan(np.deg2rad(a))]
    # ori = orientation
    radii = np.histogram(orientation, theta)
    bars = ax.bar(radii[1][:-1], radii[0], width=width, bottom=20)

    for r, bar in zip(orientation, bars):
        # bar.set_facecolor(plt.cm.jet(r/10.))
        bar.set_alpha(0.5)


def velocity_time_series(folder_list, time_list, projected_vs):
    temp_vals = list()
    mean_vals = np.zeros(len(folder_list))
    std_vals = np.zeros(len(folder_list))
    curr_val = 0

    for time, v in zip(time_list, projected_vs):
        if time == [curr_val]:
            temp_vals.append(v)
        if time != [curr_val]:
            if len(temp_vals) > 0:
                mean_vals[curr_val] = np.mean(temp_vals)
                std_vals[curr_val] = np.std(temp_vals)
                curr_val += 1
            else:
                mean_vals[curr_val] = np.nan
                std_vals[curr_val] = np.nan
                curr_val += 1
            while time != [curr_val]:
                mean_vals[curr_val] = np.nan
                std_vals[curr_val] = np.nan
                curr_val += 1
            temp_vals = list()
            temp_vals.append(v)

    n_bins = 101

    sum_run = np.nancumsum(np.insert(mean_vals, 0, 0))
    sum_std = np.nancumsum(np.insert(std_vals, 0, 0))

    rm_vs = np.zeros(len(mean_vals))
    # rm_std = np.zeros(len(folder_list))

    for n in np.arange(np.min([n_bins-1, len(mean_vals)])):
        rm_vs[n] = np.nanmean(mean_vals[:n])
        # rm_std[n] = np.nanstd(mean_vals[:n])
    rm_vs[n_bins-1:] = (sum_run[n_bins:]-sum_run[:-n_bins])/n_bins
    rm_std = (sum_std[n_bins:]-sum_std[:-n_bins])/n_bins

    f_start = folder_list[0]
    start_time = np.float16(f_start[21:23])*60+np.float16(f_start[24:26])+np.float16(f_start[27:30])/1000
    time_fl = [np.float16(f[21:23])*60+np.float16(f[24:26])+np.float16(f[27:30])/1000-start_time for f in folder_list if '.png' in f]

    plt.figure()
    plt.plot(time_fl, rm_vs, color='b')
    plt.plot(time_fl[np.int((n_bins-1)/2):np.int(-(n_bins-1)/2)], rm_vs[np.int((n_bins-1)/2):np.int(-(n_bins-1)/2)]+rm_std, color='b', linestyle='--')
    plt.plot(time_fl[np.int((n_bins-1)/2):np.int(-(n_bins-1)/2)], rm_vs[np.int((n_bins-1)/2):np.int(-(n_bins-1)/2)]-rm_std, color='b', linestyle='--')
    plt.title('Running Mean of Fall Speed over Time')
    plt.xlabel('Time in seconds')
    plt.ylabel('Running mean of fall velocity in cm/s')


def orientation_scatter(centerpt, orientation):
    center_x = [c[0] for c in centerpt]
    f, ax = plt.subplots(1)
    ax.scatter(center_x, [o/np.pi for o in orientation])
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.yaxis.set_major_locator(tck.MultipleLocator(base=0.5))

    ax.set_xlabel('x location of center point')
    ax.set_ylabel('Angle of fall streak')


def centerpt_density(centerpt, orientation, pxl_size):

    oris = sorted(zip(orientation, centerpt), key=lambda tup: tup[1][0])
    orientation_s = [o[0] for o in oris]
    centerpt_s = [o[1] for o in oris]

    bins_orientation = list()

    bin_size = 40
    max_x_bin = np.ceil(830/bin_size)
    max_y_bin = np.ceil(2048/bin_size)
    x_range = np.arange(max_x_bin+1)[1:]
    y_range = np.arange(max_y_bin+1)[1:]
    xs = x_range*bin_size
    ys = y_range*bin_size

    for i in x_range - 1:
        bins_orientation.append(list())
        for j in y_range - 1:
            bins_orientation[int(i)].append(list())

    for oc in zip(orientation_s, centerpt_s):
        set_flag = 0
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if (oc[1][0] < x) & (oc[1][1] < y):
                    bins_orientation[i][j].append(oc[0])
                    set_flag = 1
                    break
            if set_flag:
                break

    binned_o = np.zeros([len(xs), len(ys)])
    binned_n = np.zeros([len(xs), len(ys)])

    for i in x_range-1:
        for j in y_range-1:
            binned_o[int(i)][int(j)] = np.mean(bins_orientation[int(i)][int(j)])
            binned_n[int(i)][int(j)] = len(bins_orientation[int(i)][int(j)])

    plt.figure()
    plt.pcolor(xs * pxl_size, ys * pxl_size, np.flip(np.transpose(binned_n), 0) / np.max(binned_n))
    plt.colorbar()
    plt.xlabel('x in $\mu m$')
    plt.ylabel('y in $\mu m$')
    plt.title('Relative occurence of fall streak center points')


def get_angles(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurd = cv2.GaussianBlur(img, (15, 15), 0)
    grad = cv2.Laplacian(blurd, -1)
    # grad = cv2.Sobel(blurd, -1, 1, 0, ksize=15)

    # edges = cv2.Canny(grad, 3, 9, apertureSize=3)
    lines = cv2.HoughLines(grad, 1, np.pi / 180, 200)

    return lines[0][1]


if __name__ == '__main__':
    main(folder, pixel_size, exposure_time, histogram_plt_flag, orientation_polar_flag, v_t_series_flag, ori_scatter_flag, centerpt_density_flag)