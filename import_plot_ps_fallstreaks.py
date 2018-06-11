""" Creates v(D) plot of all fall track (Holography) data ('streak_data.dat') from all folders listed in folder list."""

import os
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pickle
from scipy import optimize
from Speed.Holography.generate_fallstreaks import ParticleStreak, FallParticle, refine_streaks
# from matplotlib import style
# style.use('dark_background')


# Loading data ############################
folder_list = (
    # '/ipa2/holo/mweitzel/HIVIS_Holograms/Prev23Feb/',  # Columnar, Irregular
    # '/ipa2/holo/mweitzel/HIVIS_Holograms/Meas28Feb/M2/',  # Dendritic
    # '/ipa2/holo/mweitzel/HIVIS_Holograms/Meas01Mar/',  # Dendritic
    # '/ipa2/holo/mweitzel/HIVIS_Holograms/Meas22May/',   # Columnar
    # '/ipa2/holo/mweitzel/HIVIS_Holograms/Meas23May/M2/',   # Columnar
)

habit_list = (
    # 'Columnar,'
    'Dendritic',
    'Dendritic',
    'Columnar',
    'Columnar'
)

folder_list = list()
folder_list.append('/ipa2/holo/mweitzel/HIVIS_Holograms/CalibrationBeads07Jun/')
# Properties for filtering streaks
angle_leniency_deg = 10
length_leniency_pct = 5
# starting_streak = 250
# num_streaks_processed = 15000
min_streak_length = 3  # for separation between short and long streaks
xlims = [-2.3, 2.3]
ylims = [-2.9, 2.9]

plot_angle_hist = True
plot_aspr_hist = True
plot_z_histogram = True

list_of_dim_lists = list()
list_of_streak_lists = list()
info_list = list()

for folder, habit in zip(folder_list, habit_list):
    tmp = pickle.load(open(os.path.join(folder, 'streak_data.dat'), 'rb'))
    streak_list = tmp['streaks']
    streak_list = refine_streaks(streak_list, angle_leniency_deg, length_leniency_pct)
    streak_list = [a for a in streak_list if (len(a.particle_streak) >= min_streak_length)]
    this_dim_list = list()
    for i, s in enumerate(streak_list):
        this_dim_list.append([p.majsiz * 1e6 for p in s.particle_streak])
        info_list.append({'folder': folder, 'habit': habit, 'local_index': i, 'holonum': s.particle_streak[0].holonum})
    list_of_dim_lists.append(this_dim_list)
    list_of_streak_lists.append(streak_list)


    print('Added {} streaks from {}.'.format(len(streak_list), folder))

full_dim_median_list = list()
full_streak_list = list()

for d, s in zip(list_of_dim_lists, list_of_streak_lists):
    for d1, s1 in zip(d, s):
        full_dim_median_list.append(np.median(d1))
        full_streak_list.append(s1)

indexes = list(range(len(full_dim_median_list)))
indexes.sort(key=full_dim_median_list.__getitem__)
full_dim_median_list = list(map(full_dim_median_list.__getitem__, indexes))
full_streak_list = list(map(full_streak_list.__getitem__, indexes))
info_list = list(map(info_list.__getitem__, indexes))

full_dim_list = list()
full_pos_list = list()
full_aspr_list = list()
full_angle_list = list()
full_v_list = list()
full_v_median_list = list()
full_cap_list = list()
full_streakid_list = list()
full_im_list = list()

streak_id = 0

for s in full_streak_list:
    pos = sorted([p.spatial_position for p in s.particle_streak], key=lambda pos_entry: pos_entry[1])
    pos = [p.spatial_position for p in s.particle_streak]
    this_gaps = [q - p for (p, q) in zip(pos[:-1], pos[1:])]
    s_majsiz = [t.majsiz * 1e6 for t in s.particle_streak]
    s_minsiz = [t.minsiz * 1e6 for t in s.particle_streak]

    full_dim_list.append([p.majsiz * 1e6 for p in s.particle_streak])
    # full_dim_list.append([0.58*p.minsiz/2*(1+0.95*(p.majsiz/p.minsiz)**0.75) for p in s.particle_streak])
    full_pos_list.append([p.spatial_position for p in s.particle_streak])
    full_aspr_list.append([p.majsiz / p.minsiz for p in s.particle_streak])
    full_angle_list.append([np.arctan(-g[0]/g[1]) for g in this_gaps])
    full_v_list.append([np.sqrt(g[1] ** 2 + g[0] ** 2) * 100 for g in this_gaps])
    # full_v_list.append([np.sqrt(g[1] ** 2) * 100 for g in this_gaps])
    # full_v_list.append([-g[1] * 100 for g in this_gaps])

    full_cap_list.append(
        ([0.134 * (0.58 * p.minsiz / 2 * (1 + 0.95 * (p.majsiz / p.minsiz) ** 0.75)) for p in s.particle_streak]))
    # full_streakid_list.append([streak_id] * len(s.particle_streak))
    full_streakid_list.append(streak_id)
    full_im_list.append([np.transpose(p.partimg) for p in s.particle_streak])
    streak_id += 1


def fall_speed_projection(v_list, angle_list):
    mean_mean_angle = np.mean([np.mean(c) for c in full_angle_list])
    new_angles = [a-mean_mean_angle for a in angle_list]
    v_list = [v*np.cos(beta) for v, beta in zip(v_list, new_angles)]

    return v_list, mean_mean_angle


full_v_list, mean_angle = fall_speed_projection(full_v_list, full_angle_list)
full_v_median_list = [np.median(v) for v in full_v_list]
full_aspr_median_list = [np.median(c) for c in full_aspr_list]

# Fitting power law ############################
powerlaw = lambda x, amp, index: amp * (x**index)


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


amp_full, index_full = fit_powerlaw(full_dim_median_list, full_v_median_list)

# dims_t = np.transpose(full_dim_median_list)
# fitfunc = lambda params, dims_t: params[0]*dims_t
# errfunc = lambda p, x, y: fitfunc(p, x) - y
# init_m = 0.1
# init_p = np.array((init_m))
# p1, success = optimize.leastsq(errfunc, init_p.copy(), args=(dims_t, full_v_median_list))
# f = fitfunc(p1, dims_t)

# Plotting things ############################

if plot_angle_hist:
    fig_h = plt.figure()
    mean_angle_list = [np.median(c) for c in full_angle_list]
    n_bins = 50
    ax = fig_h.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    angle_range = np.arange(-0.5*np.pi, 0.5*np.pi, np.pi / n_bins)
    angle_range = np.append(angle_range, 2 * np.pi)
    a = plt.hist(mean_angle_list, angle_range)
    ax.set_theta_zero_location("S")
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.bar(angle_range, a[1])

if plot_aspr_hist:
    fig_a = plt.figure()
    mean_aspr_list = [np.median(c) for c in full_aspr_list]
    n_bins = 20
    ax = fig_a.add_subplot(111)
    a = plt.hist(mean_aspr_list, 1.0+.25*np.arange(20), edgecolor='black', linewidth=1.2)

if plot_z_histogram:
    is_in_folder = [[] for f in folder_list]
    for pl, t_info in zip(full_pos_list, info_list):
        zs = [p[2] for p in pl]
        z_mean = np.mean(zs)
        z_var = np.std(zs)
        t_folder = t_info['folder']
        is_in_folder[folder_list.index(t_folder)].append(z_mean)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = 0.005*np.arange(12)

    ax.hist(is_in_folder, bins, weights=[np.zeros_like(np.asarray(fol))+1./len(np.asarray(fol)) for fol in is_in_folder]
            , alpha=0.5, label=folder_list, edgecolor='black', linewidth=1.3, align='left')
    ax.legend(loc='upper right')
    ax.grid()

dims_spaced = np.arange(np.ceil(1.1 * np.max(full_dim_median_list) / 10) * 10)
almost_black = '#262626'
fig = plt.figure()
ax = fig.add_subplot(111)

# for this_dim_list, this_v_list in zip(full_dim_list, full_v_list):
#     line = ax.scatter(this_dim_list, this_v_list, alpha=1,
#                edgecolors=almost_black, linewidth=1, zorder=0, picker=5)
#     # ax.errorbar([(i+j)/2 for i, j in zip(bin_edges[:-1], bin_edges[1:])], avg_masses, yerr=mass_std, fmt='o')

line = ax.scatter(full_dim_median_list, full_v_median_list, alpha=1,
                  edgecolors=almost_black, linewidth=1, zorder=0, picker=5, c=np.asarray(['r' if c['habit'] == 'Columnar' else 'b' for c in info_list]))

ax.grid()
ax.plot(dims_spaced[1:], powerlaw(dims_spaced, amp_full, index_full)[1:], label='Power Law Full', linewidth=3, zorder=1)
# ax.plot(dims_spaced, f, linewidth=3, label='Linear Capacitance Fit, v=aC, a={}]'.format(p1))
ax.set_xlim([0, np.max(dims_spaced)])
# ax.set_xlim([0, 200])
ax.set_xlabel('Maximum diameter in µm', fontsize=20)
ax.set_ylim([0, 1.1 * np.max(full_v_median_list)])
# ax.set_ylim([0, 115])
ax.set_ylabel('Fall speed in mm/s', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend()


def onpick(event):

    if event.artist != line:
        return True

    # n = len(event.ind)

    # if not n: return True

    cmap = plt.get_cmap('bone')
    for subplotnum, dataind in enumerate(event.ind):
        fig_i = plt.figure()
        n = len(full_im_list[full_streakid_list[dataind]])
        fig_fullshape = (2, n+2)
        fig_i.suptitle('Particle {} in folder {}, local index {}'.format(full_streakid_list[dataind], info_list[dataind]['folder'][36:46], info_list[dataind]['local_index']), fontsize=14)

        iml = full_im_list[full_streakid_list[dataind]]
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
        fdl = full_dim_list[full_streakid_list[dataind]]
        ax1.plot(fdl, c='b', lw=3)
        ax1.axhline(y=full_dim_median_list[full_streakid_list[dataind]], ls='--', c='b')
        ax1.set_ylim(0, 1.1*np.max(fdl))
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.set_xlabel('Index', fontsize=20)
        ax1.set_ylabel('Particle Max Diameter ($\mu$m)', fontsize=20)

        ax2 = ax1.twinx()
        fvl = full_v_list[full_streakid_list[dataind]]
        xrange = list(np.arange(1, len(fvl)+1)-0.5)
        ax2.plot(xrange, fvl, c='g', lw=3)
        ax2.axhline(y=full_v_median_list[full_streakid_list[dataind]], ls='--', c='g')
        ax2.set_ylim(0, 1.1*np.max(fvl))
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_ylabel('Fall Speed (mm/s)', fontsize=20)

        fpl = full_pos_list[full_streakid_list[dataind]]
        ax3 = plt.subplot2grid(fig_fullshape, (0, n), rowspan=2, colspan=2)
        ax3.scatter([p[0] for p in fpl], [p[1] for p in fpl])
        ax3.grid()
        ax3.set_xlim(xlims)
        ax3.set_ylim(ylims)
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position('right')
        ax3.set_xlabel('Particle x position', fontsize=20)
        ax3.set_ylabel('Particle y (vertical) position', fontsize=20)

        fig_i.show()
    return True


# print('Mean Fall Angle: {}°'.format(mean_angle))
fig.canvas.mpl_connect('pick_event', onpick)

plt.show()
