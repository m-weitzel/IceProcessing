from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy import optimize
from Speed.generate_fallstreaks import ParticleStreak, FallParticle, refine_streaks
# from matplotlib import style
# style.use('dark_background')


# Loading data ############################
folder_list = (
    # '/ipa2/holo/mweitzel/HIVIS_Holograms/Prev23Feb/',  # Columnar, Irregular
    # '/ipa2/holo/mweitzel/HIVIS_Holograms/Meas28Feb/M2/',  # Dendritic
    '/ipa2/holo/mweitzel/HIVIS_Holograms/Meas01Mar/',  # Dendritic
)

# Properties for filtering streaks
angle_leniency_deg = 10
length_leniency_pct = 10
starting_streak = 250
num_streaks_processed = 15000
min_streak_length = 5   # for separation between short and long streaks


list_of_dim_lists = list()
list_of_v_lists = list()
list_of_cap_lists = list()
list_of_aspr_lists = list()
list_of_streakid_lists = list()

streak_id = 0

for folder in folder_list:
    tmp = pickle.load(open(folder + 'streak_data.dat', 'rb'))
    streak_list = tmp['streaks']
    streak_list = refine_streaks(streak_list, angle_leniency_deg, length_leniency_pct)
    streak_list = [a for a in streak_list if (len(a.particle_streak) >= min_streak_length)]

    this_dim_list = list()
    this_v_list = list()
    this_aspr_list = list()
    this_cap_list = list()
    this_streakid_list = list()

    for s in streak_list:
        pos = sorted([p.spatial_position for p in s.particle_streak], key=lambda pos_entry: pos_entry[1])
        this_gaps = [q - p for (p, q) in zip(pos[:-2], pos[1:])]
        s_majsiz = [t.majsiz * 1e6 for t in s.particle_streak]
        s_minsiz = [t.minsiz*1e6 for t in s.particle_streak]

        this_dim_list.append([p.majsiz * 1e6 for p in s.particle_streak])
        this_v_list.append([np.sqrt(g[1] ** 2 + g[0] ** 2) * 100 for g in this_gaps])
        this_aspr_list.append([p.majsiz / p.minsiz for p in s.particle_streak])
        this_cap_list.append(([0.134 *(0.58 * p.minsiz / 2 * (1 + 0.95 * (p.majsiz / p.minsiz) ** 0.75)) for p in s.particle_streak]))
        this_streakid_list.append([streak_id]*len(s.particle_streak))
        streak_id += 1

    list_of_dim_lists.append(this_dim_list)
    list_of_v_lists.append(this_v_list)
    list_of_cap_lists.append(this_cap_list)
    list_of_aspr_lists.append(this_aspr_list)
    list_of_streakid_lists.append(this_streakid_list)

    print('Added {} streaks from {}.'.format(len(streak_list), folder))

full_dim_list = list()
full_v_list = list()
full_cap_list = list()
full_streakid_list = list()

for d, v, c, s in zip(list_of_dim_lists, list_of_v_lists, list_of_cap_lists, list_of_streakid_lists):
    for d1, v1, c1, s1 in zip(d, v, c, s):
        full_dim_list.append(np.median(d1))
        full_v_list.append(np.median(v1))
        full_cap_list.append(np.median(c1))
        full_streakid_list.append(s1[0])

full_dim_list, full_v_list, full_cap_list, full_streakid_list = zip(*sorted(zip(full_dim_list, full_v_list, full_cap_list, full_streakid_list)))
full_dim_list = list(full_dim_list)
full_v_list = list(full_v_list)
full_cap_list = list(full_cap_list)
full_streakid_list = list(full_streakid_list)
list_of_aspr_lists = list_of_aspr_lists[0]
list_of_dim_lists = list_of_dim_lists[0]

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


amp_full, index_full = fit_powerlaw(full_dim_list, full_v_list)

# Plotting things ############################

dims_spaced = np.arange(np.ceil(1.1 * np.max(full_dim_list) / 10) * 10)
almost_black = '#262626'
fig = plt.figure()
ax = fig.add_subplot(111)

# for this_dim_list, this_v_list in zip(full_dim_list, full_v_list):
#     line = ax.scatter(this_dim_list, this_v_list, alpha=1,
#                edgecolors=almost_black, linewidth=1, zorder=0, picker=5)
#     # ax.errorbar([(i+j)/2 for i, j in zip(bin_edges[:-1], bin_edges[1:])], avg_masses, yerr=mass_std, fmt='o')

line = ax.scatter(full_dim_list, full_v_list, alpha=1,
                  edgecolors=almost_black, linewidth=1, zorder=0, picker=5)

ax.grid()
ax.plot(dims_spaced, powerlaw(dims_spaced, amp_full, index_full), label='Power Law Full', linewidth=3, zorder=1)
ax.set_xlim([0, np.max(dims_spaced)])
ax.set_xlabel('Maximum diameter in µm', fontsize=20)
ax.set_ylim([0, 1.1 * np.max(full_v_list)])
ax.set_ylabel('Fall speed in mm/s', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)


def onpick(event):

    if event.artist != line: return True

    N = len(event.ind)
    if not N: return True

    fig_i = plt.figure()
    for subplotnum, dataind in enumerate(event.ind):
        ax = fig_i.add_subplot(N,1,subplotnum+1)
        ax.plot(list_of_dim_lists[full_streakid_list[dataind]])
        # ax.set_ylim(1, 3)
    fig_i.show()
    return True

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()
