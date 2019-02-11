""" Imports fall_speed_data.dat (fall streaks) from a specified folder and creates characteristic plots describing the average properties.
These plots show the characteristic spatial distribution of average fall speed, angle and other properties.
"""

import sys
import os
import matplotlib.mlab as mlab
import matplotlib.ticker as tck
import numpy as np
import pickle
from scipy.stats import norm
import cv2
import time
sys.path.append('/uni-mainz.de/homes/maweitze/PycharmProjects/MassDimSpeed/utilities')
sys.path.append('/uni-mainz.de/homes/maweitze/PycharmProjects/MassDimSpeed/Mass')
sys.path.append('/uni-mainz.de/homes/maweitze/PycharmProjects/MassDimSpeed/Speed')
from find_ccr_folder import find_ccr
from Imaging import extract_fall_data
from IceSizing import MicroImg

try:
    import matplotlib.pyplot as plt
    from make_pretty_figure import imshow_in_figure, create_hist, savefig_ipa
except ImportError:
    pass

folder = find_ccr()
folder = os.path.join(folder, 'Y2017/0908/M1')
pixel_size = 23.03  # in µm
exposure_time = 85000  # in µs

save_only_flag = 0

histogram_plt_flag = 1
orientation_polar_flag = 0
v_t_series_flag = 0
ori_scatter_flag = 0
centerpt_density_flag = 0


def main(fldr, pxl_size, exp_time, save_only=0, h_flag=1, op_flag=1, vt_flag=1, or_flag=1, dn_flag=1):

    fall_folder = os.path.join(fldr, 'Fall')
    folder_list = sorted(os.listdir(fall_folder))
    folder_list = [f for f in folder_list if '.png' in f]
    im_for_shape_acq = MicroImg('Streak', fall_folder, folder_list[0], pxl_size, ('Bin', 0))
    imsize = im_for_shape_acq.processed_image.shape

    plot_descriptor_list = list()

    try:
        fall_dist, orientation, centerpt, time_list = load_v_data(fldr)
    except FileNotFoundError:
        _, fall_dist, orientation, centerpt, time_list = extract_fall_data.initialize_data(fldr, folder_list, pxl_size)
        # list_of_file_data = extract_fall_data.initialize_data(fldr, folder_list)

    vs = np.asarray(fall_dist) / exp_time * 1000  # in mm/s
    projected_vs = [v * np.cos(o) for (v, o) in zip(vs, orientation)]
    print('Number of fall streaks: '+str(len(projected_vs)))

    t0 = time.time()
    if not save_only:
        if h_flag:
            mass_data = load_mass_data(folder)
            print('Number of mass data points: '+str(len(mass_data[0])))
            mass_velocity_dim_histograms(projected_vs, mass_data, folder, all_four=False)
            t1 = time.time()
            print('Time spent on Mass Velocity Histograms: {0:.1f} s'.format(t1-t0))
            plot_descriptor_list += ['histogram.png']
            t0 = time.time()
        if op_flag:
            orientation_polar_plot(orientation)
            t1 = time.time()
            print('Time spent on Orientation Polar Plot: {0:.1f} s'.format(t1-t0))
            plot_descriptor_list += ['orientation_polar.png']
            t0 = time.time()
        if vt_flag:
            velocity_time_series(folder_list, time_list, projected_vs)
            t1 = time.time()
            print('Time spent on Velocity Time Series: {0:.1f} s'.format(t1-t0))
            plot_descriptor_list += ['v_timeseries.png', 'v_count_of_streaks']
            t0 = time.time()
        if or_flag:
            orientation_scatter(centerpt, orientation)
            t1 = time.time()
            print('Time spent on Orientation Scatter Plot: {0:.1f} s'.format(t1-t0))
            plot_descriptor_list += ['orientation_scatter.png']
            t0 = time.time()
        if dn_flag:
            centerpt_density(centerpt, orientation, vs, imsize, pixel_size)
            t1 = time.time()
            # plot_descriptor_list += ['number_density.png', 'orientation_heatmap.png', 'quiver.png']
            plot_descriptor_list += ['number_density.png', 'quiver.png']
            print('Time spent on Centerpoint Density Plot: {0:.1f} s'.format(t1-t0))

        try:
            os.mkdir(os.path.join(fldr, 'plots/'))
        except FileExistsError:
            pass

        for i, p in zip(plt.get_fignums(), plot_descriptor_list):
            f = plt.figure(i)
            # plt.savefig(os.path.join(fldr, 'plots/'+plot_descriptor_list[i-1]))
            savefig_filepath = fldr[-7:]+'_'+p[:-4]
            savefig_ipa(f, savefig_filepath.replace('/', ''))

        plt.show()


def load_v_data(fldr):
    list_of_lists = pickle.load(open(os.path.join(fldr, 'fall_speed_data.dat'), 'rb'))
    # cont_real = list_of_lists[0]
    fall_dist = list_of_lists[1]
    orientation = list_of_lists[2]
    centerpt = list_of_lists[3]
    time_list = list_of_lists[4]

    return fall_dist, orientation, centerpt, time_list


def load_mass_data(fldr):
    tmp = pickle.load(open(os.path.join(fldr, 'mass_dim_data.dat'), 'rb'))

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


def mass_velocity_dim_histograms(vs, mass_data, fldr, all_four=True):

    n_bins = 15
    v_max = 8
    ae_max = 100
    mxdim_max = 120
    # mass_max = 87500
    dropdiam_max = 75

    area_eq_diam_list = mass_data[0]
    max_diam_list = mass_data[1]
    dropdiam_list = mass_data[3]

    mass_list = [np.pi/6*a**3/1000 for a in dropdiam_list]   # mass in milligrams

    def plot_param_hist(t_ax, list_of_vals, max_val, t_n_bins, unit):
        bins = max_val/n_bins*np.arange(t_n_bins)
        (mu, sigma) = norm.fit(list_of_vals)
        n, bins, _ = t_ax.hist(list_of_vals, bins=bins)
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

    fig = plt.figure(figsize=(16, 12.5))

    bins = np.arange(0, 150, 10)

    if all_four:
        axs = fig.subplots(2, 2)

        ax = axs[0][0]
        _, ax = create_hist(vs, ax=ax, bins=bins, maxval=v_max)
        # plot_param_hist(ax, vs, v_max, n_bins, 'mm/s')
        ax.set_title('Terminal Velocity in mm/s', fontsize=20)
        # ax.set_xlabel('Terminal Velocity in mm/s')

        ax = axs[0][1]
        _, ax = create_hist(area_eq_diam_list, ax=ax, bins=bins, maxval=ae_max)
        # plot_param_hist(ax, area_eq_diam_list, ae_max, n_bins, '$\mu m$')
        ax.set_title(r'Area equivalent diameter in $\mu m$', fontsize=20)
        # ax.set_xlabel('Area equivalent diameter in um')

        ax = axs[1][1]
        _, ax = create_hist(max_diam_list, ax=ax, bins=bins, maxval=mxdim_max)
        # plot_param_hist(ax, max_diam_list, mxdim_max, n_bins, '$\mu m$')
        ax.set_title(r'Maximum diameter in $\mu m$', fontsize=20)
        # ax.set_xlabel('Maximum diameter in um')

        ax = axs[1][0]
        # bins = 10*np.arange(19)
        _, ax = create_hist(mass_list, ax=ax, bins=bins)
        # plot_param_hist(ax, dropdiam_list, dropdiam_max, n_bins, '$\mu m$')
        ax.set_title('Crystal mass in mg', fontsize=20)
        # ax.set_xlabel('Drop diameter in um')

    else:
        axs = fig.subplots(2)

        ax = axs[0]
        bins = np.arange(0, 120, 5)
        _, ax = create_hist(vs, ax=ax, bins=bins, maxval=v_max, grid=False)
        ax.set_title('Terminal Velocity in mm/s', fontsize=20)

        ax = axs[1]
        bins = np.arange(0, np.max(area_eq_diam_list), 5)
        _, ax = create_hist(area_eq_diam_list, ax=ax, bins=bins, maxval=ae_max, grid=False)
        ax.set_title(r'Area equivalent diameter in $\mu m$', fontsize=20)

    # plt.suptitle('Histogram Overview for '+fldr[-8:], fontsize=24)

    # plt.savefig(fldr + 'histogram.png')


def orientation_polar_plot(orientation):

    # fig = plt.figure(figsize=(8, 8))
    fig = imshow_in_figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location("S")
    n_bins = 100
    width = (2*np.pi)/n_bins
    theta = np.linspace(-np.pi+np.pi/n_bins, np.pi+np.pi/n_bins, n_bins, endpoint=False)
    # max_height = 8
    # ori = [np.deg2rad(a) for a in np.sort(orientation) if not(np.isnan(np.deg2rad(a)))]
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
    streaks_in_pic = np.zeros(len(folder_list))
    curr_val = 0

    for t_time, v in zip(time_list, projected_vs):
        if t_time == curr_val:
            temp_vals.append(v)
        elif t_time > curr_val:
            while t_time > curr_val:
                # if t_time == curr_val:
                #     temp_vals.append(v)
                # else:
                    if len(temp_vals) > 1:
                        mean_vals[curr_val] = np.mean(temp_vals)
                        std_vals[curr_val] = np.std(temp_vals)
                        curr_val += 1
                        streaks_in_pic[curr_val] = len(temp_vals)
                        temp_vals = list()
                    else:
                        mean_vals[curr_val] = np.nan
                        std_vals[curr_val] = np.nan
                        curr_val += 1
                        streaks_in_pic[curr_val] = len(temp_vals)
                        temp_vals = list()

    n_bins = 25

    sum_run = np.nancumsum(np.insert(mean_vals, 0, 0))
    sum_std = np.nancumsum(np.insert(std_vals, 0, 0))

    rm_vs = np.zeros(len(mean_vals))
    # rm_std = np.zeros(len(folder_list))

    for n in np.arange(np.min([n_bins-1, len(mean_vals)])):
        rm_vs[n] = np.nanmean(mean_vals[:n])
        # rm_std[n] = np.nanstd(mean_vals[:n])
    # rm_vs[n_bins-1:] = (sum_run[n_bins:]-sum_run[:-n_bins])/n_bins
    # rm_std = (sum_std[n_bins:]-sum_std[:-n_bins])/n_bins

    nan_mask = np.isnan(mean_vals)
    K = np.ones(n_bins, dtype=int)
    rm_vs = np.convolve(np.where(nan_mask, 0, mean_vals), K, 'same')/np.convolve(~nan_mask, K, 'same')
    nan_mask = np.isnan(std_vals)
    rm_std = np.convolve(np.where(nan_mask, 0, std_vals), K, 'same')/np.convolve(~nan_mask, K, 'same')

    f_start = folder_list[0]
    start_time = np.float16(f_start[21:23])*60+np.float16(f_start[24:26])+np.float16(f_start[27:30])/1000
    time_fl = [np.float16(f[21:23])*60+np.float16(f[24:26])+np.float16(f[27:30])/1000-start_time for f in folder_list if '.png' in f]

    f, ax = imshow_in_figure(figspan=(18, 10))
    ax.plot(time_fl, rm_vs, color='b')
    # ax.plot(time_fl[np.int((n_bins-1)/2):np.int(-(n_bins-1)/2)], rm_vs[np.int((n_bins-1)/2):np.int(-(n_bins-1)/2)]+rm_std, color='b', linestyle='--')
    # ax.plot(time_fl[np.int((n_bins-1)/2):np.int(-(n_bins-1)/2)], rm_vs[np.int((n_bins-1)/2):np.int(-(n_bins-1)/2)]-rm_std, color='b', linestyle='--')
    # ax.set_ylim([0, np.max(rm_vs[np.int((n_bins-1)/2):np.int(-(n_bins-1)/2)]+rm_std)])
    ax.plot(time_fl, rm_vs+rm_std, color='b', linestyle='--')
    ax.plot(time_fl, rm_vs-rm_std, color='b', linestyle='--')
    ax.set_xlim([0, max(time_fl)])
    ax.set_ylim([0, 1.1*np.max(rm_vs+rm_std)])
    ax.set_title('Running mean of fall velocity over time', fontsize=20)
    ax.set_xlabel('Time since start of experiment in seconds')
    ax.set_ylabel('Running mean of fall velocity in mm/s', fontsize=20)

    f2, ax2 = imshow_in_figure(figspan=(18, 10))
    ax2.bar([t+0.5 for t in time_fl], streaks_in_pic)


def orientation_scatter(centerpt, orientation):
    center_x = [c[0] for c in centerpt]
    f, ax = plt.subplots(1)
    ax.scatter(center_x, [o/np.pi for o in orientation])
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.yaxis.set_major_locator(tck.MultipleLocator(base=0.5))

    ax.set_xlabel('x location of center point')
    ax.set_ylabel('Angle of fall streak')


def centerpt_density(centerpt, orientation, vs, imsize, pxl_size, relative=True):

    oris = sorted(zip(orientation, centerpt, vs), key=lambda tup: tup[1][0])
    orientation_s = [o[0] for o in oris]
    centerpt_s = [o[1] for o in oris]
    vs_s = [o[2] for o in oris]

    bins_orientation = list()
    bins_velocity = list()

    bin_size = 60
    max_x_bin = np.ceil(imsize[1]/bin_size)
    max_y_bin = np.ceil(imsize[0]/bin_size)
    x_range = np.arange(max_x_bin+1)[1:]
    y_range = np.arange(max_y_bin+1)[1:]
    xs = x_range*bin_size
    ys = y_range*bin_size

    for i in x_range - 1:
        bins_orientation.append(list())
        bins_velocity.append(list())
        for _ in y_range - 1:
            bins_orientation[int(i)].append(list())
            bins_velocity[int(i)].append(list())

    for oc in zip(orientation_s, centerpt_s, vs_s):
        set_flag = 0
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if (oc[1][0] < x) & (oc[1][1] < y):
                    bins_orientation[i][j].append(oc[0])
                    bins_velocity[i][j].append(oc[2])
                    set_flag = 1
                    break
            if set_flag:
                break

    binned_o = np.zeros([len(xs), len(ys)])
    binned_n = np.zeros([len(xs), len(ys)])
    binned_v = np.zeros([len(xs), len(ys)])

    for i in x_range-1:
        for j in y_range-1:
            this_o = np.median(bins_orientation[int(i)][int(j)])
            this_v = np.median(bins_velocity[int(i)][int(j)])
            if np.isnan(this_o) or (len(bins_orientation[int(i)][int(j)]) < 4):
                binned_o[int(i)][int(j)] = -3
                binned_v[int(i)][int(j)] = 0
            else:
                binned_o[int(i)][int(j)] = this_o
                binned_v[int(i)][int(j)] = this_v

            binned_n[int(i)][int(j)] = len(bins_orientation[int(i)][int(j)])

    if relative:
        sum_all_n = np.sum(binned_n)
        binned_n = [b/sum_all_n for b in [c for c in binned_n]]

    # f, ax = plt.subplots(figsize=(8, 14))
    f, ax = imshow_in_figure(figspan=(8, 14))
    ax.set_aspect('equal')
    ax.set_xlim([xs[0]*pxl_size/1000, xs[-1]*pxl_size/1000])
    ax.set_ylim([ys[0]*pxl_size/1000, ys[-1]*pxl_size/1000])
    f.canvas.draw()
    cmap = plt.cm.YlOrRd
    # cmap = plt.cm.Spectral_r
    cmap.set_under(color='white')
    im = ax.pcolor(xs * pxl_size/1000, ys * pxl_size/1000, np.flip(np.transpose(binned_n), 0), cmap=cmap, vmin=0.0001)
    cbar = f.colorbar(im)
    cbar.ax.tick_params(labelsize=20)
    # cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel('x in $mm$', fontsize=20)
    ax.set_ylabel('y in $mm$', fontsize=20)
    ax.set_title('PDF of fall streak distribution in sample volume', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # ax.axis([0, 1000, 0, 1000])
    # f, ax = plt.subplots(figsize=(8, 14))
    # ax.set_aspect('equal')
    # ax.set_xlim(xs[0]*pxl_size/1000, xs[-1]*pxl_size/1000)
    # ax.set_ylim(ys[0]*pxl_size/1000, ys[-1]*pxl_size/1000)
    # f.canvas.draw()
    cmap = plt.cm.Spectral_r
    cmap.set_under(color='white')
    # im = ax.pcolor(xs * pxl_size/1000, ys * pxl_size/1000, np.flip(np.transpose(np.rad2deg(binned_o)), 0), cmap=cmap, vmin=-30, vmax=30)
    # ax.set_xlabel('x in $mm$', fontsize=20)
    # ax.set_ylabel('y in $mm$', fontsize=20)
    # ax.set_title('Median fall streak orientation relative to verticality', fontsize=20)
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # cbar = f.colorbar(im, ticks=np.linspace(-45, 45, 7))
    # cbar.set_label('$\phi$ in $\degree$', fontsize=20)
    # ticklabels = list(np.linspace(-1, 1, 9))
    # ticklabels = [str(t)+'$/4\cdot\pi$' for t in ticklabels]
    # cbar.ax.set_yticklabels([str(t)+'$/4\cdot\pi$' for t in list(np.linspace(-1, 1, 9))])

    # f, ax = plt.subplots(figsize=(8, 14))
    f, ax = imshow_in_figure(figspan=(8, 14))
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    ax.set_xlim(xs[0] * pxl_size / 1000, xs[-1] * pxl_size / 1000)
    ax.set_ylim(ys[0] * pxl_size / 1000, ys[-1] * pxl_size / 1000)
    f.canvas.draw()
    X, Y = np.meshgrid(xs*pxl_size/1000, ys*pxl_size/1000)
    im_pc = ax.pcolor(xs * pxl_size/1000, ys * pxl_size/1000, np.flip(np.transpose(binned_v), 0), cmap=cmap, vmin=1.5, vmax=50)
    im_qv = ax.quiver(X, Y, np.sin(np.flip(np.transpose(binned_o), 0)*np.flip(np.transpose(binned_v), 0)),
                      -np.cos(np.flip(np.transpose(binned_o), 0))*np.flip(np.transpose(binned_v), 0))
    cbar = f.colorbar(im_pc, ticks=np.linspace(0, 50, 6))
    cbar.set_label('v in $mm/s$', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('x in $mm$', fontsize=20)
    ax.set_ylabel('y in $mm$', fontsize=20)
    ax.set_title('Quiver plot of mean orientation and fall speed', fontsize=20)


def get_angles(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurd = cv2.GaussianBlur(img, (15, 15), 0)
    grad = cv2.Laplacian(blurd, -1)
    # grad = cv2.Sobel(blurd, -1, 1, 0, ksize=15)

    # edges = cv2.Canny(grad, 3, 9, apertureSize=3)
    lines = cv2.HoughLines(grad, 1, np.pi / 180, 200)

    return lines[0][1]


if __name__ == '__main__':
    main(folder, pixel_size, exposure_time, save_only_flag, histogram_plt_flag, orientation_polar_flag, v_t_series_flag, ori_scatter_flag, centerpt_density_flag)