""" Creates v(D) plot of all fall track (Holography) data ('streak_data.dat') from all folders listed in folder list."""

import os
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import pickle
from Speed.Holography.generate_fallstreaks import ParticleStreak, FallParticle, refine_streaks, get_folder_framerate, eta
# from utilities.plot_size_distribution import plot_size_dist
from utilities.fit_powerlaw import fit_powerlaw
from itertools import cycle, compress, chain
# from matplotlib.widgets import CheckButtons
# from matplotlib import style
# style.use('dark_background')


def main():
    # Loading data ############################

    folder_list = list()

    # folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/Prev23Feb/')  # Columnar, Irregular
    # folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/Meas28Feb/M2/')  # Dendritic
    # folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/Meas01Mar/')  # Dendritic
    # folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/Meas22May/')   # Columnar
    # folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/Meas23May/M2/')   # Columnar
    folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/2905M1/')
    # folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/26Sep/')

    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/CalibrationBeads07Jun/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/CalibrationBeads08Jun/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/CalibrationDrops20Aug/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/Calibration22AugN2/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/Calibration06SepWarm/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/Calibration11SepStopfen/')
    # folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/Calibration11SepVentilated/')
    # folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/Calibration11SepNonVent/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/Calibration20Aug/')
    # folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/Calibration22AugN1/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/Calibration13SepSmall/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/Calibration13Sep54fps/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/Calibration19SepStopfen/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/Calibration20Sep/')

    # # Properties for filtering streaks

    angle_leniency_deg = 5
    min_streak_length = 3  # for separation between short and long streaks

    rho_o = 2500

    separate_by = 'habit'

    hist_plots = True
    calc_means = True
    plot_powerlaws = True
    plot_stokes = True
    plot_folder_means = False

    oversizing_correction = True

    list_of_folder_dim_lists = list()
    list_of_folder_streak_lists = list()
    list_of_folder_v_lists = list()
    info_list = list()
    temp_by_folder = list()

    for folder in folder_list:
        tmp = pickle.load(open(os.path.join(folder, 'streak_data.dat'), 'rb'))
        framerate = get_folder_framerate(tmp['folder'])
        this_folders_streak_list = tmp['streaks']
        print('Added {} streaks from {}.'.format(len(this_folders_streak_list), folder))
        this_folders_streak_list = refine_streaks(this_folders_streak_list, angle_leniency_deg)
        this_folders_streak_list = [a for a in this_folders_streak_list if (len(a.particle_streak) >= min_streak_length)]
        this_folders_mean_angle = np.mean([s.mean_angle for s in this_folders_streak_list])
        this_folders_dim_list = list()
        this_folders_v_list = list()
        for i, s in enumerate(this_folders_streak_list):
            this_folders_dim_list.append([p.majsiz * 1e6 for p in s.particle_streak])
            this_folders_v_list.append(s.get_projected_velocity(this_folders_mean_angle, framerate))
            info_list.append({'folder': folder, 'local_index': i, 'holonum': s.particle_streak[0].holonum})
            try:
                info_list[i]['temperature'] = tmp['temperature']
            except KeyError:
                pass

        list_of_folder_dim_lists.append(this_folders_dim_list)
        list_of_folder_streak_lists.append(this_folders_streak_list)
        list_of_folder_v_lists.append(this_folders_v_list)
        try:
            temp_by_folder.append(tmp['temperature'])
        except KeyError:
            pass

        print('Kept {} streaks longer than {}.'.format(len(this_folders_streak_list), min_streak_length))

    full_dim_list = list()
    full_dim_median_list = list()
    full_v_list = list()
    full_v_median_list = list()
    full_streak_list = list()

    for d, s, v in zip(list_of_folder_dim_lists, list_of_folder_streak_lists, list_of_folder_v_lists):
        for d1, s1, v1 in zip(d, s, v):
            if oversizing_correction:
                d1 = [td-1.5*2.22 for td in d1]
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
    selector_index_dict = dict()
    dim_dict = dict()
    v_median_dict = dict()
    aspr_dict = dict()

    if plot_powerlaws:
        plaw_by_habits = dict()
        plaw_vals_by_habits = dict()

    if separate_by == 'habit':
        different_separators = list(set(full_habit_list))
    elif separate_by == 'folder':
        different_separators = folder_list
    else:
        different_separators = ['']

    # Plotting

    for sep in different_separators:
        if separate_by == 'habit':
            selector_index_dict[sep] = [1 if s.streak_habit == sep else 0 for s in full_streak_list]
        elif separate_by == 'folder':
            selector_index_dict[sep] = [1 if i['folder'] == sep else 0 for i in info_list]
        else:
            print('Separator not found, showing full lists.')
            selector_index_dict[sep] = [1]*len(full_streak_list)
        v_median_dict[sep] = list(compress(full_v_median_list, selector_index_dict[sep]))
        dim_dict[sep] = list(compress(full_dim_median_list, selector_index_dict[sep]))
        aspr_dict[sep] = list(compress(full_aspr_median_list, selector_index_dict[sep]))
        streaks_by_separator = list(compress(full_streak_list, selector_index_dict[sep]))
        info_by_separator = list(compress(info_list, selector_index_dict[sep]))

        if hist_plots:
            plot_hists_by_habit(sep, streaks_by_separator, dim_dict[sep], aspr_dict[sep], info_by_separator)
        if plot_powerlaws:
            plaw_vals_by_habits[sep] = fit_powerlaw(dim_dict[sep], v_median_dict[sep])
            powerlaw = lambda x, plaw_factor, plaw_exponent: plaw_factor * (x ** plaw_exponent)
            plaw_by_habits[sep] = powerlaw(dim_dict[sep], plaw_vals_by_habits[sep][0], plaw_vals_by_habits[sep][1])

    ax = v_dim_scatter(selector_index_dict, full_dim_list, full_dim_median_list, full_v_median_list, full_v_list, different_separators, full_im_list,
                       full_streakid_list, info_list, full_pos_list)

    # ax2 = plot_best_vs_reynolds(v_median_dict['Column         '], dim_dict['Column         '], aspr_dict['Column         '], cap_flag=True)

    try:
        temp_stokes = info_list[0]['temperature']
    except KeyError:
        temp_stokes = -10

    if calc_means:
        for sep in different_separators:
            plot_mean_in_scatter(ax, list(compress(full_dim_list, selector_index_dict[sep])),
                                 list(compress(full_v_list, selector_index_dict[sep])))

    if plot_powerlaws:
        for sep in different_separators:
            # ax.plot(dim_dict[hab], plaw_by_habits[hab])
            ax.plot(dim_dict[sep], plaw_by_habits[sep],
                    label='{0}, v={1:.2f}d^{2:.2f}, R^2={3:.3f}'.format(sep, plaw_vals_by_habits[sep][0], plaw_vals_by_habits[sep][1], plaw_vals_by_habits[sep][2]))
            ax.legend()

    if plot_stokes:
        d_mean = 30e-6
        # y_vel_min = -2*((d_mean+1.2e-6)/2)**2*9.81*(rho_o-100-1.34)/(9*eta(temp_stokes-2.5))*1e3
        # y_vel_mean = -2*(d_mean/2)**2*9.81*(rho_o-1.34)/(9*eta(temp_stokes))*1e3
        # y_vel_max = -2*((d_mean-1.2e-6)/2)**2*9.81*(rho_o+100-1.34)/(9*eta(temp_stokes+2.5))*1e3
        # ax.axhline(-y_vel_mean, color='k', lw=3, label='Expected value from Stokes, v={0:.2f} mm/s'.format(-y_vel_mean))
        # ax.fill_between(np.arange(0, 1.2*np.max(full_dim_median_list)), -y_vel_min, -y_vel_max, facecolor='grey', alpha=0.4)
        # ax.legend(loc='upper left')

        ds = np.arange(np.round(1.2*np.max(dim_dict[sep])))
        stokes_vod_min = [2*((d+1.2)*1e-6/2)**2*9.81*((rho_o-100)-1.34)/(9*eta(temp_stokes-2.5))*1000 for d in ds]
        stokes_v_over_d = [2*(d*1e-6/2)**2*9.81*(rho_o-1.34)/(9*eta(temp_stokes))*1000 for d in ds]
        stokes_vod_max = [2*((d-1.2)*1e-6/2)**2*9.81*((rho_o+100)-1.34)/(9*eta(temp_stokes+2.5))*1000 for d in ds]
        ax.plot(ds, stokes_v_over_d, label='Stokes', linewidth=3, color='black')
        ax.fill_between(ds, stokes_vod_min, stokes_vod_max, facecolor='grey', alpha=0.6)

    ax.legend()
    savepath = os.path.join(info_list[0]['folder'], 'plots/v_dim_scatter.png')
    print('Saving plot to {}.'.format(savepath))
    plt.savefig(savepath)

    if plot_folder_means:
        fig_mean = plt.figure(figsize=(18, 10), dpi=100)
        fig_mean.canvas.set_window_title('Mean values for different folders')
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
        ax_mean.set_xticks(np.arange(len(folder_list)))
        ax_mean.set_xticklabels([f[47:-1] for f in folder_list], fontsize=20)
        # ax_mean.set_xlim(0, 5)
        ax_mean.set_ylim(0, 110)
        ax_mean.grid(b=True, which='major', linestyle='-')
        ax_mean.set_ylabel('Mean velocity', fontsize=20)
        ax_mean.set_title('Fall speed of 30 $\mu m$ calibration glass spheres', fontsize=20)
        r_mean = 30e-6
        y_vel = lambda T: -2*(r_mean/2)**2*9.81*(rho_o-1.34)/(9*eta(T))*1e3

        fig_box = plt.figure(figsize=(18, 10), dpi=100)
        ax_box = fig_box.add_subplot(111)
        fig_box.canvas.set_window_title('Box-whisker for different folders')
        if plot_stokes:
            hline_stokes = lambda T, ax_st, i: ax_st.plot([i-0.5, i+0.5], [-y_vel(T)]*2, color='b', lw=3,
                                                          label='Expected value from Stokes, v={0:.2f} mm/s for T={1:d}°C'.format(y_vel(T), T))
            for i, t in enumerate(temp_by_folder):
                hline_stokes(t, ax_mean, i)
                hline_stokes(t, ax_box, i+1)
        # ax_mean.plot(np.arange(2, 6)-0.5, [-y_vel_p25]*4, color='r', lw=3, label='Expected value from Stokes, v={0:.2f} mm/s for T=+25°C'.format(-y_vel_p25))
        # ax_mean.plot(np.arange(5, 7)-0.5, [-y_vel_m10]*2, color='g', lw=3, label='Expected value from Stokes, v={0:.2f} mm/s for T=-10°C'.format(-y_vel_m10))
        ax_mean.legend(loc='upper left')

        fs = list(set(v_median_dict))
        ax_box.boxplot([v_median_dict[p] for p in fs])
        ax_box.set_xticklabels([f[47:-1] for f in fs], fontsize=20)
        ax_box.grid(b=True, which='major', linestyle='-')

    plt.show()


def plot_best_vs_reynolds(v_list, dim_list, aspr_list, cap_flag=False):
    fig = plt.figure(figsize=(18, 10), dpi=100)
    fig.canvas.set_window_title('Best vs. Reynolds')
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

    ax.errorbar(mean_dim, mean_v, std_v, std_dim, marker='s', markersize=12, capsize=5, picker=2, mew=1, mec='k', elinewidth=2, ecolor='k')
    ax.text(5, 175, 'v={0:.2f}+-{1:.2f}m/s, D={2:.2f}+-{3:.2f}\mu m'.format(mean_v, std_v, mean_dim, std_dim),
            bbox=dict(facecolor='green', alpha=0.2), fontsize=12)


def v_dim_scatter(selector_list, dim_list, dim_median_list, v_median_list,
                  full_v_list, different_separators, im_list, streakid_list, info_list, pos_list):

    max_dim = 0
    for sep in different_separators:
        max_dim = np.max([max_dim, np.max(list(compress(dim_median_list, selector_list[sep])))])

    dims_spaced = np.arange(np.ceil(1.1 * max_dim / 10) * 10)
    # locatelli_hobbs = 0.69*(dims_spaced*1e-3)**0.41*1e3

    almost_black = '#262626'
    fig = plt.figure(figsize=(18, 10), dpi=100)
    fig.canvas.set_window_title('v-D scatter')
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

    try:
        for i, sep in enumerate(different_separators):
            t_dim = list(compress(dim_median_list, selector_list[sep]))
            t_v = list(compress(v_median_list, selector_list[sep]))
            lines.append(ax.scatter(t_dim, t_v, alpha=1,
                                    edgecolors=almost_black, linewidth=1, zorder=0, picker=i,
                                    # label='{} (N={})'.format(hab, sum(selector_list[hab])), marker=marker_dict[hab]))
                                    label='v={0:.2f}$\pm${1:.2f}$m/s$\n D={2:.2f}$\pm${3:.2f}$\mu m$'
                                    .format(np.mean(t_v), np.std(t_v), np.mean(t_dim), np.std(t_dim)), marker=marker_dict[sep]))
            streakids_in_habits[sep] = list(compress(streakid_list, selector_list[sep]))
    except KeyError:
        for i, sep in enumerate(different_separators):
            t_dim = list(compress(dim_median_list, selector_list[sep]))
            t_v = list(compress(v_median_list, selector_list[sep]))
            lines.append(ax.scatter(t_dim, t_v, alpha=1,
                                    edgecolors=almost_black, linewidth=1, zorder=0, picker=i,
                                    # label='{} (N={})'.format(sep, sum(selector_list[sep])), marker=marker_dict[sep]))
                                    label='v={0:.2f}$\pm${1:.2f}$m/s$\n D={2:.2f}$\pm${3:.2f}$\mu m$'
                                    .format(np.mean(t_v), np.std(t_v), np.mean(t_dim), np.std(t_dim))))
            streakids_in_habits[sep] = list(compress(streakid_list, selector_list[sep]))

    ax.grid()
    # ax.plot(dims_spaced[1:], powerlaw(dims_spaced, amp_full, index_full)[1:], label='Power Law Full', linewidth=3, zorder=1)
    # ax.plot(dims_spaced, f, linewidth=3, label='Linear Capacitance Fit, v=aC, a={}]'.format(p1))
    ax.set_xlim([0, np.max(dims_spaced)])
    # ax.set_xlim([20, 45])
    ax.set_xlabel('Maximum diameter in µm', fontsize=20)
    ax.set_ylim([0, 1.1 * np.max(v_median_list)])
    # ax.set_ylim([0, 1.1*maximum_vel])
    # ax.set_ylim([0, 115])
    ax.set_ylabel('Fall speed in mm/s', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.plot(dims_spaced, locatelli_hobbs, label='Locatelli+Hobbs 74', linewidth=2, color='b')
    # ax.axhline(maximum_vel, linewidth=2, color='k', label='Maximum measurable velocity')
    # ax.set_title('Fall speed vs. dimension for {}'.format(habit))
    ax.set_title('Fall speed vs. dimension for Calibration Beads of $30\mu$m diameter', fontsize=24)
    ax.legend(fontsize=20, loc='upper left')

    # check = CheckButtons(ax, different_habits, [True]*len(different_habits))
    # check.on_clicked(func(different_habits, lines))

    fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, dim_list, dim_median_list,
                                                              im_list, streakids_in_habits, info_list, pos_list, v_median_list, full_v_list, lines, different_separators))

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
        fig_i.canvas.set_window_title('Fall track detail')

        # try:
        global_streakid = streakid_list[hab][dataind]
        # except IndexError:
        #     print('{}, {}, {}'.format(hab, len(list(streakid_list[hab])), dataind))

        n = len(im_list[global_streakid])
        fig_fullshape = (2, n+2)

        fig_i.suptitle('{} No. {} in folder {}, local index {}'.format(hab, dataind,
                       info_list[streakid_list[hab][dataind]]['folder'][47:], info_list[streakid_list[hab][dataind]]['local_index']), fontsize=14)

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
    fig_h.canvas.set_window_title('Size Distribution')
    ax = fig_h.add_subplot(111)
    min_bin = max(0.5, (min(dim_median_list)//5-1)*5+0.5)
    max_bin = (max(dim_median_list)//5+2)*5+0.5
    bins = np.arange(min_bin, max_bin, 5)
    bins = np.arange(min_bin, max_bin, 1)

    histo = plt.hist(dim_median_list, bins, edgecolor='black', linewidth=1.2)
    ax.grid(b=True, which='major', linestyle='-')
    ax.set_axisbelow('True')
    ax.set_title('Size distribution for {}'.format(habit), fontsize=28)
    ax.set_title('Size distribution for 30 $\mu$m calibration beads, measured by HIVIS', fontsize=24)
    ax.set_xlabel('Maximum dimension in $\mu$m', fontsize=20)
    ax.set_ylabel('Count', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Angle histogram plot
    fig_h = plt.figure()
    fig_h.canvas.set_window_title('Angle Histogram')
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
    fig_a.canvas.set_window_title('Aspect Ratio Histogram')
    mean_aspr_list = [np.median(c) for c in aspr_list]
    n_bins = 20
    bins = 1.0 + .25 * np.arange(n_bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    ax = fig_a.add_subplot(111)
    ax.grid(b=True, which='major', linestyle='-')
    histo = plt.hist(mean_aspr_list, logbins, edgecolor='black', linewidth=1.2)
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

