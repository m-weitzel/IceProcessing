from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy import optimize
from Speed.generate_fallstreaks import ParticleStreak, FallParticle, refine_streaks
import random
from matplotlib import style
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


full_dim_list = list()
full_v_list = list()
full_cap_list = list()
full_aspr_list = list()

for folder in folder_list:
    # tmp = pickle.load(open(folder + 'vs_dim_data.dat', 'rb'))
    tmp = pickle.load(open(folder + 'streak_data.dat', 'rb'))
    streak_list = tmp['streaks']
    streak_list = refine_streaks(streak_list, angle_leniency_deg, length_leniency_pct)
    streak_list = [a for a in streak_list if (len(a.particle_streak) >= min_streak_length)]

    full_dim_list += [np.median([p.majsiz*1e6 for p in s.particle_streak]) for s in streak_list]
    full_aspr_list += [np.median([p.majsiz/p.minsiz for p in s.particle_streak]) for s in streak_list]
    full_cap_list += [0.134*np.median([0.58*p.minsiz/2*(1+0.95*(p.majsiz/p.minsiz)**0.75) for p in s.particle_streak]) for s in streak_list]

    this_v_list = list()
    this_cap_list = list()

    for s in streak_list:
        pos = sorted([p.spatial_position for p in s.particle_streak], key=lambda pos_entry: pos_entry[1])
        this_gaps = [q - p for (p, q) in zip(pos[:-2], pos[1:])]

        s_majsiz = [t.majsiz * 1e6 for t in s.particle_streak]
        s_minsiz = [t.minsiz*1e6 for t in s.particle_streak]

        vs = [np.sqrt(g[1]**2+g[0]**2)*100 for g in this_gaps]
        this_v_list.append(np.median(vs))
        # this_cap_list.append(0.134*np.median([0.58*this_min/2*(1+0.95*(this_maj/this_min)**0.75) for this_min, this_maj in zip(s_minsiz, s_majsiz)]))

    full_v_list += this_v_list
    full_cap_list += this_cap_list

    print('Added {} streaks from {}.'.format(len(streak_list), folder))


full_dim_list, full_v_list, full_cap_list, full_aspr_list = zip(*sorted(zip(full_dim_list, full_v_list, full_cap_list, full_aspr_list)))

random_sample = random.sample(range(len(streak_list)), 25)

# Fitting power law ############################
powerlaw = lambda x, amp, index: amp * (x**index)


def fit_powerlaw(x, y):

    x = [this_x for this_x, this_y in zip(x, y) if not(np.isnan(this_y))]
    y = [this_y for this_y in y if not(np.isnan(this_y))]

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

dims_spaced = np.arange(np.ceil(1.1*np.max(full_dim_list)/10)*10)
almost_black = '#262626'
fig, ax = plt.subplots(1)
for this_dim_list, this_v_list in zip(full_dim_list, full_v_list):
    ax.scatter(this_dim_list, this_v_list, alpha=1,
               edgecolors=almost_black, linewidth=1, zorder=0)
    # ax.errorbar([(i+j)/2 for i, j in zip(bin_edges[:-1], bin_edges[1:])], avg_masses, yerr=mass_std, fmt='o')

ax.grid()
ax.plot(dims_spaced, powerlaw(dims_spaced, amp_full, index_full), label='Power Law Full', linewidth=3, zorder=1)
ax.set_xlim([0, np.max(dims_spaced)])
ax.set_xlabel('Maximum diameter in µm', fontsize=20)
ax.set_ylim([0, 1.1*np.max(full_v_list)])
ax.set_ylabel('Fall speed in mm/s', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)

fig, axarr = plt.subplots(5, 5)
for i in range(5):
    for j in range(5):
        # streak_dims = [p.majsiz for p in streak_list[random_sample[i*5+j]].particle_streak]
        streak_dims = [p.majsiz/p.minsiz for p in streak_list[random_sample[i * 5 + j]].particle_streak]
        mean_dim = np.mean(streak_dims)
        # streak_dims = [(s-mean_dim)/mean_dim*100 for s in streak_dims]

        axarr[i, j].plot(streak_dims)
        axarr[i, j].set_ylim([-20, 20])
        axarr[i, j].set_ylim([1, 3])
        axarr[i, j].grid()
        axarr[i, j].axhline(y=0, c='k')
        axarr[i, j].set_xticks(range(len(streak_dims)))

plt.show()
