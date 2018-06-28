""" Creates v(D) plot of all fall track (Holography) data ('streak_data.dat') from all folders listed in folder list."""

import os
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pickle
from scipy import optimize
from Speed.Holography.generate_fallstreaks import ParticleStreak, FallParticle, refine_streaks
from itertools import cycle, compress, chain
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
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/Meas22May/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/Meas23May/M2/')
    # folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/CalibrationBeads07Jun/')
    folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/2905/ps/seq1/')

    # # Properties for filtering streaks

    angle_leniency_deg = 5
    length_leniency_pct = 3
    min_streak_length = 3  # for separation between short and long streaks

    hist_plots = False
    calc_means = True

    list_of_folder_dim_lists = list()
    list_of_folder_streak_lists = list()
    list_of_folder_v_lists = list()
    info_list = list()

    for folder in folder_list:
        tmp = pickle.load(open(os.path.join(folder, 'streak_data.dat'), 'rb'))
        this_folders_streak_list = tmp['streaks']
        print('Added {} streaks from {}.'.format(len(this_folders_streak_list), folder))
        this_folders_streak_list = refine_streaks(this_folders_streak_list, angle_leniency_deg, length_leniency_pct)
        this_folders_streak_list = [a for a in this_folders_streak_list if (len(a.particle_streak) >= min_streak_length)]
        this_folders_mean_angle = np.mean([s.mean_angle for s in this_folders_streak_list])
        this_folders_dim_list = list()
        this_folders_v_list = list()
        for i, s in enumerate(this_folders_streak_list):
            this_folders_dim_list.append([p.majsiz * 1e6 for p in s.particle_streak])
            this_folders_v_list.append(s.get_projected_velocity(this_folders_mean_angle))
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

    # Plotting

    for hab in different_habits:
        selector_index_dict[hab] = [1 if s.streak_habit == hab else 0 for s in full_streak_list]
        streaks_by_habit = list(compress(full_streak_list, selector_index_dict[hab]))
        dim_median_by_habit = list(compress(full_dim_median_list, selector_index_dict[hab]))
        info_by_habit = list(compress(info_list, selector_index_dict[hab]))
        aspr_by_habit = list(compress(full_aspr_median_list, selector_index_dict[hab]))

        if hist_plots:
            plot_hists_by_habit(hab, streaks_by_habit, dim_median_by_habit, aspr_by_habit, info_by_habit)

    ax = v_dim_scatter(selector_index_dict, full_dim_list, full_dim_median_list, full_v_median_list, full_v_list, different_habits, full_im_list,
                       full_streakid_list, info_list, full_pos_list)
    if calc_means:
        hab = different_habits[0]
        plot_mean_in_scatter(ax, list(compress(full_dim_list, selector_index_dict[hab])),
                             list(compress(full_v_list, selector_index_dict[hab])))

    plt.show()


def plot_mean_in_scatter(ax, dim_list, v_list):
    mean_dim = np.mean(list(chain.from_iterable(dim_list)))
    mean_v = np.mean(list(chain.from_iterable(v_list)))
    std_dim = np.std(list(chain.from_iterable(dim_list)))
    std_v = np.std(list(chain.from_iterable(v_list)))

    ax.errorbar(mean_dim, mean_v, std_v, std_dim, marker='s', markersize=12, label='Means', color='k')
    ax.text(5, 175, 'v={0:.2f}+-{1:.2f}, D={2:.2f}+-{3:.2f}'.format(mean_v, std_v, mean_dim, std_dim),
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

    return amp, index


def v_dim_scatter(selector_list, dim_list, dim_median_list, v_median_list,
                  full_v_list, different_habits, im_list, streakid_list, info_list, pos_list):

    max_dim = 0
    for hab in different_habits:
        max_dim = np.max([max_dim, np.max(list(compress(dim_median_list, selector_list[hab])))])

    dims_spaced = np.arange(np.ceil(1.1 * max_dim / 10) * 10)
    locatelli_hobbs = 0.69*(dims_spaced*1e-3)**0.41*1e3
    maximum_vel = (2.22*2592-200)/2*60/1000
    almost_black = '#262626'
    fig = plt.figure()
    ax = fig.add_subplot(111)

    marker_dict = dict()
    marker_dict['Column         '] = 's'
    marker_dict['Aggregate      '] = ','
    marker_dict['Particle_nubbly'] = 'v'
    marker_dict['Particle_round '] = 'o'
    marker_dict['Plate          '] = (6, 0, 0)
    marker_dict['Needle         '] = 'D'

    # for this_dim_list, this_v_list in zip(full_dim_list, full_v_list):
    #     line = ax.scatter(this_dim_list, this_v_list, alpha=1,
    #                edgecolors=almost_black, linewidth=1, zorder=0, picker=5)
    #     # ax.errorbar([(i+j)/2 for i, j in zip(bin_edges[:-1], bin_edges[1:])], avg_masses, yerr=mass_std, fmt='o')

    # print('\nMean diameter: {:.2f} +- {:.2f}'.format(np.mean(dim_list), np.std(dim_list)))
    # print('\nMean fall velocity: {:.2f} +- {:.2f}'.format(np.mean(v_list), np.std(v_list)))

    streakids_in_habits = dict()
    lines = list()

    for i, hab in enumerate(different_habits):
        lines.append(ax.scatter(list(compress(dim_median_list, selector_list[hab])), list(compress(v_median_list, selector_list[hab])), alpha=1,
                                edgecolors=almost_black, linewidth=1, zorder=0, picker=i,
                                label='{} (N={})'.format(hab, sum(selector_list[hab])), marker=marker_dict[hab]))
        streakids_in_habits[hab] = list(compress(streakid_list, selector_list[hab]))

    ax.grid()
    # ax.plot(dims_spaced[1:], powerlaw(dims_spaced, amp_full, index_full)[1:], label='Power Law Full', linewidth=3, zorder=1)
    # ax.plot(dims_spaced, f, linewidth=3, label='Linear Capacitance Fit, v=aC, a={}]'.format(p1))
    ax.set_xlim([0, np.max(dims_spaced)])
    # ax.set_xlim([0, 200])
    ax.set_xlabel('Maximum diameter in µm', fontsize=20)
    # ax.set_ylim([0, 1.1 * np.max(v_median_list)])
    ax.set_ylim([0, 1.1*maximum_vel])
    # ax.set_ylim([0, 115])
    ax.set_ylabel('Fall speed in mm/s', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.plot(dims_spaced, locatelli_hobbs, label='Locatelli+Hobbs 74', linewidth=2, color='b')
    ax.axhline(maximum_vel, linewidth=2, color='k', label='Maximum measurable velocity')
    # ax.set_title('Fall speed vs. dimension for {}'.format(habit))
    ax.legend()

    fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, dim_list, dim_median_list,
                                                              im_list, streakids_in_habits, info_list, pos_list, v_median_list, full_v_list, lines, different_habits))
    return ax


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
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_xlabel('Index', fontsize=20)
        ax1.set_ylabel('Particle Max Diameter ($\mu$m)', fontsize=20)

        ax2 = ax1.twinx()
        fvl = full_v_list[global_streakid]
        xrange = list(np.arange(1, len(fvl)+1)-0.5)
        ax2.plot(xrange, fvl, c='g', lw=3)
        ax2.axhline(y=v_median_list[global_streakid], ls='--', c='g')
        ax2.set_ylim(0, 1.1*np.max(fvl))
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
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

