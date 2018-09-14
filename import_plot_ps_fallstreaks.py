""" Creates v(D) plot of all fall track (Holography) data ('streak_data.dat') from all folders listed in folder list."""

import os
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import pickle
from scipy import optimize
from Speed.Holography.generate_fallstreaks import ParticleStreak, FallParticle, refine_streaks
from utilities.plot_size_distribution import plot_size_dist
from itertools import cycle, compress, chain
from sklearn.metrics import r2_score
from matplotlib.widgets import CheckButtons
# from matplotlib import style
# style.use('dark_background')


def main():
    # Loading data ############################
    folder_list = (
        # '/ipa2/holo/mweitzel/HIVIS_Holograms/Prev23Feb/',  # Columnar, Irregular
        # '/ipa2/holo/mweitzel/HIVIS_Holograms/Meas28Feb/M2/',  # Dendritic
        # '/ipa2/holo/mweitzel/HIVIS_Holograms/Meas01Mar/',  # Dendritic
        # '/ipa2/holo/mweitzel/HIVIS_Holograms/Meas22May/',   # Columnar
        # '/ipa2/holo/mweitzel/HIVIS_Holograms/Meas23May/M2/',   # Columnar
        # '/ipa2/holo/mweitzel/HIVIS_Holograms/2905/ps/seq1/'
    )

    folder_list = list()

    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/CalibrationBeads07Jun/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/CalibrationBeads08Jun/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/Calibration20Aug/')
    folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/CalibrationDrops20Aug/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/Calibration22AugN1/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/Calibration22AugN2/')

    # # Properties for filtering streaks

    angle_leniency_deg = 5
    min_streak_length = 3   # for separation between short and long streaks
    fr_guess = 54          # guess for frame rate

    rho_o = 2500
    d_mean = 30e-6
    eta = 18.37*1e-6

    hist_plots = False
    calc_means = True
    plot_expected = False
    plot_powerlaws = False
    plot_stokes = True
    psd_flag = False
    plot_folder_means = True

    list_of_folder_dim_lists = list()
    list_of_folder_streak_lists = list()
    list_of_folder_v_lists = list()
    info_list = list()

    for folder in folder_list:
        tmp = pickle.load(open(os.path.join(folder, 'streak_data.dat'), 'rb'))
        this_folders_streak_list = tmp['streaks']
        print('Added {} streaks from {}.'.format(len(this_folders_streak_list), folder))
        this_folders_streak_list = refine_streaks(this_folders_streak_list, angle_leniency_deg)
        this_folders_streak_list = [a for a in this_folders_streak_list if (len(a.particle_streak) >= min_streak_length)]
        this_folders_mean_angle = np.mean([s.mean_angle for s in this_folders_streak_list])
        this_folders_dim_list = list()
        this_folders_v_list = list()
        for i, s in enumerate(this_folders_streak_list):
            this_folders_dim_list.append([p.majsiz * 1e6 for p in s.particle_streak])
            this_folders_v_list.append(s.get_projected_velocity(this_folders_mean_angle, fr_guess))
            info_list.append({'folder': folder, 'local_index': i, 'holonum': s.particle_streak[0].holonum})
        list_of_folder_dim_lists.append(this_folders_dim_list)
        list_of_folder_streak_lists.append(this_folders_streak_list)
        list_of_folder_v_lists.append(this_folders_v_list)

        print('Kept {} streaks longer than {}.'.format(len(this_folders_streak_list), min_streak_length))

    full_dim_list = list()
    full_dim_median_list = list()
    full_v_list = list()
    full_v_median_list = list()
    full_streak_list = list()

    for d, s, v in zip(list_of_folder_dim_lists, list_of_folder_streak_lists, list_of_folder_v_lists):
        for d1, s1, v1 in zip(d, s, v):
            full_dim_list.append(d1)
            full_dim_median_list.append(np.median(d1))
            full_v_list.append(v1)
            full_v_median_list.append(np.median(v1))
            full_streak_list.append(s1)

    indexes = list(range(len(full_dim_median_list)))
    indexes.sort(key=full_dim_median_list.__getitem__)
    full_dim_list = list(map(full_dim_list.__getitem__, indexes))
    full_dim_median_list = list(map(full_dim_median_list.__getitem__, indexes))
    full_streak_list = list(map(full_streak_list.__getitem__, indexes))
    info_list = list(map(info_list.__getitem__, indexes))
    full_v_list = list(map(full_v_list.__getitem__, indexes))
    full_v_median_list = list(map(full_v_median_list.__getitem__, indexes))

    streak_id = 0
    full_aspr_list = list()
    full_cap_list = list()
    full_im_list = list()
    full_streakid_list = list()
    full_habit_list = list()
    full_pos_list = list()
    for s in full_streak_list:
        full_aspr_list.append([p.majsiz / p.minsiz for p in s.particle_streak])  # aspect ratio of all p
        full_cap_list.append(
            ([0.134 * (0.58 * p.minsiz / 2 * (1 + 0.95 * (p.majsiz / p.minsiz) ** 0.75)) for p in
              s.particle_streak]))
        full_im_list.append([np.transpose(p.partimg) for p in s.particle_streak])
        full_streakid_list.append(streak_id)
        full_habit_list.append(s.streak_habit)
        full_pos_list.append([p.spatial_position for p in s.particle_streak])

        streak_id += 1

    full_aspr_median_list = [np.median(c) for c in full_aspr_list]
    different_habits = list(set(full_habit_list))
    selector_index_dict = dict()
    dim_dict = dict()
    v_median_dict = dict()
    aspr_dict = dict()

    if plot_powerlaws:
        plaw_by_habits = dict()
        plaw_vals_by_habits = dict()

    # Plotting

    for hab in different_habits:
        selector_index_dict[hab] = [1 if s.streak_habit == hab else 0 for s in full_streak_list]
        v_median_dict[hab] = list(compress(full_v_median_list, selector_index_dict[hab]))
        dim_dict[hab] = list(compress(full_dim_median_list, selector_index_dict[hab]))
        aspr_dict[hab] = list(compress(full_aspr_median_list, selector_index_dict[hab]))
        streaks_by_habit = list(compress(full_streak_list, selector_index_dict[hab]))
        info_by_habit = list(compress(info_list, selector_index_dict[hab]))

        if hist_plots:
            plot_hists_by_habit(hab, streaks_by_habit, dim_dict[hab], aspr_dict[hab], info_by_habit)
        if plot_powerlaws:
            plaw_vals_by_habits[hab] = fit_powerlaw(dim_dict[hab], v_median_dict[hab])
            powerlaw = lambda x, plaw_factor, plaw_exponent: plaw_factor * (x ** plaw_exponent)
            plaw_by_habits[hab] = powerlaw(dim_dict[hab], plaw_vals_by_habits[hab][0], plaw_vals_by_habits[hab][1])

        if psd_flag:
            if (len(dim_dict[hab]) > 50) | (hab == 'Particle_round '):
                n_bins = 20
                if len(dim_dict[hab]) < 100:
                    n_bins = 10
                (fig, ax) = plot_size_dist(dim_dict[hab], n_bins)
                ax.set_title('Size distribution {}'.format(hab), fontsize=24)

    ax = v_dim_scatter(selector_index_dict, full_dim_list, full_dim_median_list, full_v_median_list, full_v_list, different_habits, full_im_list,
                       full_streakid_list, info_list, full_pos_list)

    # ax2 = plot_best_vs_reynolds(v_median_dict['Column         '], dim_dict['Column         '], aspr_dict['Column         '], cap_flag=True)

    if calc_means:
        hab = different_habits[0]
        plot_mean_in_scatter(ax, list(compress(full_dim_list, selector_index_dict[hab])),
                             list(compress(full_v_list, selector_index_dict[hab])))

    if plot_expected:                               # viscosity at -20°C
        y_vel = -2*(d_mean/2)**2*9.81*(rho_o-1.34)/(9*eta)*1e3
        ax.axhline(-y_vel, color='k', lw=3, label='Expected value from Stokes, v={0:.2f} mm/s'.format(-y_vel))
        ax.legend(loc='upper left')

    if plot_powerlaws:
        for hab in different_habits:
            # ax.plot(dim_dict[hab], plaw_by_habits[hab])
            ax.plot(dim_dict[hab], plaw_by_habits[hab],
                    label='{0}, v={1:.2f}d^{2:.2f}, R^2={3:.3f}'.format(hab, plaw_vals_by_habits[hab][0], plaw_vals_by_habits[hab][1], plaw_vals_by_habits[hab][2]))
            ax.legend()

    if plot_stokes:
        ds = np.arange(np.round(1.2*np.max(dim_dict[hab])))
        stokes_v_over_d = [2*(d*1e-6/2)**2*9.81*(rho_o-1.2)/(9*eta)*1000 for d in ds]
        ax.plot(ds, stokes_v_over_d, label='Stokes', linewidth=3)

    plt.savefig(os.path.join(info_list[0]['folder'], 'plots/v_dim_scatter.png'))

    if plot_folder_means:
        fig_mean = plt.figure(figsize=(18, 10), dpi=100)
        ax_mean = fig_mean.add_subplot(111)
        folder_mean_vs = list()
        folder_std_vs = list()
        folder_mean_dims = list()
        folder_std_dims = list()
        for fvl, fdl in zip(list_of_folder_v_lists, list_of_folder_dim_lists):
            folder_mean_vs.append(np.mean([np.mean(vl) for vl in fvl]))
            folder_std_vs.append(np.std([np.mean(vl) for vl in fvl]))
            folder_mean_dims.append(np.mean([np.mean(dl) for dl in fdl]))
            folder_std_dims.append(np.std([np.mean(dl) for dl in fdl]))
        ax_mean.errorbar(np.arange(len(folder_mean_vs)), folder_mean_vs, folder_std_vs, [0]*len(folder_mean_vs), linestyle='none', marker='s', markersize=12, label='Means', color='k', capsize=5)
        ax_mean.set_xticks(np.arange(4))
        ax_mean.set_xticklabels([f[47:-1] for f in folder_list], fontsize=20)
        # ax_mean.set_xlim(0, 5)
        ax_mean.set_ylim(0, 110)
        ax_mean.grid(b=True, which='major', linestyle='-')
        ax_mean.set_xlabel('Measurement name', fontsize=20)
        ax_mean.set_ylabel('Mean velocity', fontsize=20)
        ax_mean.set_title('Fall speed of 30 $\mu m$ calibration glass spheres', fontsize=20)
        eta_m25 = 15.88*1e-6                                # viscosity at -205°C
        eta_m10 = 16.65*1e-6
        eta_p25 = 18.32*1e-6
        # y_vel_m25 = -2*(folder_mean_dims[0]*1e-6/2)**2*9.81*(rho_o-1.34)/(9*eta_m25)*1e3
        # y_vel_p25 = -2*(folder_mean_dims[-1]*1e-6/2)**2*9.81*(rho_o-1.34)/(9*eta_p25)*1e3
        # ax_mean.plot(np.arange(2), [-y_vel_m25]*2, color='b', lw=3, label='Expected value from Stokes, v={0:.2f} mm/s for T=-25°C'.format(-y_vel_m25))
        # ax_mean.plot(np.arange(2, 4), [-y_vel_p25]*2, color='r', lw=3, label='Expected value from Stokes, v={0:.2f} mm/s for T=+25°C'.format(-y_vel_p25))
        y_vel_m10 = -2*(folder_mean_dims[-1]*1e-6/2)**2*9.81*(rho_o-1.34)/(9*eta_p25)*1e3
        # y_vel_m10 = -2*(30*1e-6/2)**2*9.81*(rho_o-1.34)/(9*eta_p25)*1e3
        ax_mean.plot(np.arange(2), [-y_vel_m10]*2, color='b', lw=3, label='Expected value from Stokes for mean diameter {0:.2f} $\mu$m, v={1:.2f} mm/s for T=-10°C'.format(folder_mean_dims[-1], -y_vel_m10))
        ax_mean.legend(loc='upper left')

    plt.show()


def plot_best_vs_reynolds(v_list, dim_list, aspr_list, cap_flag=False):
    fig = plt.figure(figsize=(18, 10), dpi=100)
    almost_black = '#262626'

    if cap_flag:
        minsiz_list = [d/a for d, a in zip(dim_list, aspr_list)]
        cap_list = [0.58*w/2*(1+0.95*(l/w)**0.75) for w, l in zip(minsiz_list, dim_list)]
        best_list = [best_number(r * 1e-6, 3) for r in cap_list]
        reynolds_list = [reynolds_number(v / 1000, r * 1e-6) for v, r in zip(v_list, dim_list)]

    best_list = [best_number(r*1e-6, 3) for r in dim_list]
    reynolds_list = [reynolds_number(v/1000, r*1e-6) for v, r in zip(v_list, dim_list)]
    rey_best_fit = fit_powerlaw(reynolds_list, best_list)
    burgesser = [40 * r ** 1.36 for r in reynolds_list]

    br_ax = fig.add_subplot(111)
    br_ax.scatter(reynolds_list, best_list, label='Data', edgecolor=almost_black)
    br_ax.plot(reynolds_list, [rey_best_fit[0]*r**rey_best_fit[1] for r in reynolds_list], label='Data Fit')
    br_ax.plot(reynolds_list, burgesser, color='r', label='Burgesser Power Law')
    br_ax.legend(fontsize=16)
    br_ax.grid(b=True, which='major', linestyle='-')
    br_ax.grid(b=True, which='minor', linestyle='--', linewidth=0.5)
    br_ax.set_xlabel('Reynolds Number', fontsize=20)
    br_ax.set_ylabel('Best Number', fontsize=20)
    br_ax.set_title('Be vs. Re', fontsize=20)
    br_ax.set_xscale('log')
    br_ax.set_yscale('log')
    br_ax.set_xlim(0.7*min(reynolds_list), 1.7*max(reynolds_list))
    br_ax.set_ylim(0.2*min(best_list), 5*max(best_list))
    br_ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    br_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    return br_ax


def reynolds_number(v, d):
    rho = 1.341             # air, kg/m³ at -10°C
    eta = 17.1*1e-6         # air, Pa*s
    return v*d*rho/eta


def best_number(d, aspr):
    rho = 1.341
    eta = 17.1*1e-6
    g = 9.81
    a_m = 0.008
    b_m = 2.026
    m = a_m*(d*100)**b_m*1e-3
    best_n = 4*m*d*rho*g/(aspr*2*d*eta**2)
    return best_n


def plot_mean_in_scatter(ax, dim_list, v_list):
    mean_dim = np.mean(list(chain.from_iterable(dim_list)))
    mean_v = np.mean(list(chain.from_iterable(v_list)))
    std_dim = np.std(list(chain.from_iterable(dim_list)))
    std_v = np.std(list(chain.from_iterable(v_list)))

    ax.errorbar(mean_dim, mean_v, std_v, std_dim, marker='s', markersize=12, label='Means', color='k')
    ax.text(5, 175, 'v={0:.2f}+-{1:.2f}m/s, D={2:.2f}+-{3:.2f}\mu m'.format(mean_v, std_v, mean_dim, std_dim),
            bbox=dict(facecolor='green', alpha=0.2), fontsize=12)


def fit_powerlaw(x, y):

    # x = [this_x for this_x, this_y in zip(x, y) if not(np.isnan(this_y))]
    # y = [this_y for this_y in y if not(np.isnan(this_y))]

    logx = np.log10(x)
    logy = np.log10(y)

    fitfunc = lambda p, x: p[0]+p[1]*x
    errfunc = lambda p, x, y: (y-fitfunc(p,x))

    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit, args=(logx, logy), full_output=1)

    pfinal = out[0]
    covar = out[1]

    index = pfinal[1]
    amp = 10.0**pfinal[0]
    pl = amp * (x**index)

    r2 = r2_score(y, amp*x**index)

    return amp, index, r2


def v_dim_scatter(selector_list, dim_list, dim_median_list, v_median_list,
                  full_v_list, different_habits, im_list, streakid_list, info_list, pos_list):

    max_dim = 0
    for hab in different_habits:
        max_dim = np.max([max_dim, np.max(list(compress(dim_median_list, selector_list[hab])))])

    dims_spaced = np.arange(np.ceil(1.1 * max_dim / 10) * 10)
    locatelli_hobbs = 0.69*(dims_spaced*1e-3)**0.41*1e3

    maximum_vel = (2.22*2592-200)/2*60/1000
    almost_black = '#262626'
    fig = plt.figure(figsize=(18, 10), dpi=100)
    ax = fig.add_subplot(111)

    marker_dict = dict()
    marker_dict['Column         '] = 's'
    marker_dict['Aggregate      '] = (10, 1, 0)
    marker_dict['Particle_nubbly'] = 'v'
    marker_dict['Particle_round '] = 'o'
    marker_dict['Particle_round'] = 'o'
    marker_dict['Plate          '] = (6, 0, 0)
    marker_dict['Needle         '] = 'D'
    marker_dict['Dendrite       '] = '*'

    # for this_dim_list, this_v_list in zip(full_dim_list, full_v_list):
    #     line = ax.scatter(this_dim_list, this_v_list, alpha=1,
    #                edgecolors=almost_black, linewidth=1, zorder=0, picker=5)
    #     # ax.errorbar([(i+j)/2 for i, j in zip(bin_edges[:-1], bin_edges[1:])], avg_masses, yerr=mass_std, fmt='o')

    # print('\nMean diameter: {:.2f} +- {:.2f}'.format(np.mean(dim_list), np.std(dim_list)))
    # print('\nMean fall velocity: {:.2f} +- {:.2f}'.format(np.mean(v_list), np.std(v_list)))

    streakids_in_habits = dict()
    lines = list()

    for i, hab in enumerate(different_habits):
        t_dim = list(compress(dim_median_list, selector_list[hab]))
        t_v = list(compress(v_median_list, selector_list[hab]))
        lines.append(ax.scatter(t_dim, t_v, alpha=1,
                                edgecolors=almost_black, linewidth=1, zorder=0, picker=i,
                                # label='{} (N={})'.format(hab, sum(selector_list[hab])), marker=marker_dict[hab]))
                                label='v={0:.2f}$\pm${1:.2f}$m/s$\n D={2:.2f}$\pm${3:.2f}$\mu m$'.format(np.mean(t_v), np.std(t_v),
                                                                           np.mean(t_dim), np.std(t_dim)), marker = marker_dict[hab]))
        streakids_in_habits[hab] = list(compress(streakid_list, selector_list[hab]))

    ax.grid()
    # ax.plot(dims_spaced[1:], powerlaw(dims_spaced, amp_full, index_full)[1:], label='Power Law Full', linewidth=3, zorder=1)
    # ax.plot(dims_spaced, f, linewidth=3, label='Linear Capacitance Fit, v=aC, a={}]'.format(p1))
    # ax.set_xlim([0, np.max(dims_spaced)])
    # ax.set_xlim([20, 50])
    ax.set_xlabel('Maximum diameter in µm', fontsize=20)
    ax.set_ylim([0, 1.1 * np.max(v_median_list)])
    # ax.set_ylim([0, 1.1*maximum_vel])
    # ax.set_ylim([0, 115])
    ax.set_ylabel('Fall speed in mm/s', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.plot(dims_spaced, locatelli_hobbs, label='Locatelli+Hobbs 74', linewidth=2, color='b')
    # ax.axhline(maximum_vel, linewidth=2, color='k', label='Maximum measurable velocity')
    # ax.set_title('Fall speed vs. dimension for {}'.format(habit))
    ax.legend(fontsize=16)

    # check = CheckButtons(ax, different_habits, [True]*len(different_habits))
    # check.on_clicked(func(different_habits, lines))

    fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, dim_list, dim_median_list,
                                                              im_list, streakids_in_habits, info_list, pos_list, v_median_list, full_v_list, lines, different_habits))
    return ax


def func(label, different_habits, lines):
    i = different_habits.index(label)
    lines[i].set_visible(not lines[i].get_visible())


def onpick(event, dim_list, dim_median_list, im_list, streakid_list, info_list, pos_list, v_median_list, full_v_list, line_list, diff_habits):

    xlims = [-2.3, 2.3]
    ylims = [-2.9, 2.9]

    # n = len(event.ind)

    # if not n: return True

    hab = diff_habits[line_list.index(event.artist)]

    cmap = plt.get_cmap('bone')
    for subplotnum, dataind in enumerate(event.ind):
        print(dataind)
        fig_i = plt.figure()

        # try:
        global_streakid = streakid_list[hab][dataind]
        # except IndexError:
        #     print('{}, {}, {}'.format(hab, len(list(streakid_list[hab])), dataind))

        n = len(im_list[global_streakid])
        fig_fullshape = (2, n+2)

        fig_i.suptitle('{} No. {} in folder {}, local index {}'.format(hab, dataind,
                       info_list[streakid_list[hab][dataind]]['folder'][36:46], info_list[streakid_list[hab][dataind]]['local_index']), fontsize=14)

        iml = im_list[global_streakid]
        im_maxsize = max([im.shape for im in iml])

        for m, im in enumerate(iml):
            ax = plt.subplot2grid(fig_fullshape, (0, m))
            ax.imshow(np.abs(im), cmap=cmap)
            frame1 = plt.gca()
            frame1.axes.xaxis.set_ticklabels([])
            frame1.axes.yaxis.set_ticklabels([])
            ax.set_xlabel(m, fontsize=20)
            ax.set_ylim(0, im_maxsize[0])
            ax.set_xlim(0, im_maxsize[1])
            plt.subplots_adjust(wspace=0.8)
            # ax.set_xlabel('Index of particle in streak', fontsize=20)
            # ax.set_ylabel('Maximum diameter in µm', fontsize=20)
            # ax.set_title('Index evolution of particle size', fontsize=20)
        ax1 = plt.subplot2grid(fig_fullshape, (1, 0), colspan=n)
        fdl = dim_list[global_streakid]
        ax1.plot(fdl, c='b', lw=3)
        ax1.axhline(y=dim_median_list[global_streakid], ls='--', c='b')
        ax1.set_ylim(0, 1.1*np.max(fdl))
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax1.set_xlabel('Index', fontsize=20)
        ax1.set_ylabel('Particle Max Diameter ($\mu$m)', fontsize=20)

        ax2 = ax1.twinx()
        fvl = full_v_list[global_streakid]
        xrange = list(np.arange(1, len(fvl)+1)-0.5)
        ax2.plot(xrange, fvl, c='g', lw=3)
        ax2.axhline(y=v_median_list[global_streakid], ls='--', c='g')
        ax2.set_ylim(0, 1.1*np.max(fvl))
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax2.set_ylabel('Fall Speed (mm/s)', fontsize=20)

        fpl = pos_list[global_streakid]
        ax3 = plt.subplot2grid(fig_fullshape, (0, n), rowspan=2, colspan=2)
        ax3.scatter([p[0] for p in fpl], [p[1] for p in fpl])
        ax3.grid()
        ax3.set_xlim(xlims)
        ax3.set_ylim(ylims)
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position('right')
        ax3.set_xlabel('Particle x position in mm', fontsize=20)
        ax3.set_ylabel('Particle y (vertical) position in mm', fontsize=20)

        fig_i.show()
    return True


def plot_hists_by_habit(habit, streak_list, dim_median_list, aspr_list, info_list):

    # Plotting things ############################

    # Size distribution
    fig_h = plt.figure()
    ax = fig_h.add_subplot(111)
    histo = plt.hist(dim_median_list, edgecolor='black', linewidth=1.2)
    ax.set_title('Size distribution for {}'.format(habit))
    ax.set_xlabel('Maximum dimension in $\mu$m', fontsize=20)
    ax.set_ylabel('Count', fontsize=20)

    # Angle histogram plot
    fig_h = plt.figure()
    mean_angle_list = [s.mean_angle for s in streak_list]
    n_bins = 50
    ax = fig_h.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    angle_range = np.arange(-0.5 * np.pi, 0.5 * np.pi, np.pi / n_bins)
    angle_range = np.append(angle_range, 2 * np.pi)
    a = plt.hist(mean_angle_list, angle_range)
    ax.set_theta_zero_location("S")
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.bar(angle_range, a[1])
    ax.set_title('Angle histogram for {}'.format(habit))

    # Aspect ratio histogram plot
    fig_a = plt.figure()
    mean_aspr_list = [np.median(c) for c in aspr_list]
    n_bins = 20
    ax = fig_a.add_subplot(111)
    histo = plt.hist(mean_aspr_list, 1.0 + .25 * np.arange(20), edgecolor='black', linewidth=1.2)
    ax.set_title('Aspect ratio histogram for {}'.format(habit))
#
#     # z position histogram plot
#     is_in_folder = [[] for f in folder_list]
#     for pl, t_info in zip(full_pos_list, info_list):
#         zs = [p[2] for p in pl]
#         z_mean = np.mean(zs)
#         z_var = np.std(zs)
#         t_folder = t_info['folder']
#         is_in_folder[folder_list.index(t_folder)].append(z_mean)
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     bins = 0.005 * np.arange(12)
#
#     ax.hist(is_in_folder, bins,
#             weights=[np.zeros_like(np.asarray(fol)) + 1. / len(np.asarray(fol)) for fol in is_in_folder]
#             , alpha=0.5, label=folder_list, edgecolor='black', linewidth=1.3, align='left')
#     ax.legend(loc='upper right')
#     ax.set_title('z position for {}'.format(habit))
#     ax.grid()


if __name__ == "__main__":
    main()

