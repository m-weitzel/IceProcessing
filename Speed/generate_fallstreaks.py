import scipy.io as sio
from scipy import optimize
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os


def main():
    path = '/ipa2/holo/mweitzel/HIVIS_Holograms/Meas01Mar/'
    filename_ps = 'ps_bypredict.mat'

    a = sio.loadmat(path+filename_ps)

    # Adjust to how the camera was tilted to have true spatial directions in fall images, multiply by 1000 to get mm
    tmp = a['xp']*1000
    a['xp'] = a['yp']*1000
    a['yp'] = tmp

    # General parameters
    pxl_size = 1

    # Properties for finding streaks
    max_size_diff = 0.1
    max_dist_from_predict = 1
    base_velocity_guess = [0, -0.08, 0]
    min_streak_length = 5   # for separation between short and long streaks

    # Properties for filtering streaks
    angle_leniency_deg = 10
    length_leniency_pct = 10
    starting_streak = 250
    num_streaks_processed = 15000

    # End of properties

    p_list = list()
    for k in range(0, len(a['times'])):
        p_list.append(FallParticle(a['times'][k][0], 0, a['xp'][k][0]*pxl_size,
                                   a['yp'][k][0]*pxl_size, a['zp'][k][0]*pxl_size,
                                   a['majsiz'][k][0], a['minsiz'][k][0]))

    last_holonum = p_list[-1].holonum

    holonums = range(0, last_holonum+1)

    ps_in_holos = [[] for _ in holonums]

    for p in p_list:
        ps_in_holos[p.holonum].append(p)

    streak_list = list()

    while p_list:
        this_particle = p_list[0]
        velocity_guess = base_velocity_guess
        new_streak = ParticleStreak(this_particle, velocity_guess)
        p_list.remove(this_particle)
        extended = True
        while extended:
            try:
                # If no particles in hologram, end the streak
                if len(ps_in_holos[new_streak.particle_streak[-1].holonum+1]) < 1:
                    extended = False
                    ext_by = []
                else:
                    # Else find streak particles in next hologram
                    new_streak, extended, ext_by = find_streak_particles(new_streak, ps_in_holos[new_streak.particle_streak[-1].holonum+1],
                                                                         velocity_guess, max_dist=max_dist_from_predict, max_size_diff=max_size_diff)
            except IndexError:
                # If last hologram is reached, end the streak
                extended = False
                ext_by = []

            if extended:
                p_list.remove(ext_by)                                                 # remove the recently added particle from particle list
                ps_in_holos[new_streak.particle_streak[-2].holonum+1].remove(ext_by)  # remove particle from list of particles in current (next) hologram
                if len(new_streak.particle_streak) > 2:
                    # Starting with the third particle, velocity guess for finding new particles is derived from the last two found particles
                    velocity_guess = new_streak.particle_streak[-1].spatial_position-new_streak.particle_streak[-2].spatial_position
        streak_list.append(new_streak)

    streak_list = refine_streaks(streak_list, angle_leniency_deg, length_leniency_pct)
    only_long_streaks = [a for a in streak_list if (len(a.particle_streak) >= min_streak_length)]
    if len(only_long_streaks) < (starting_streak+np.max([num_streaks_processed, len(only_long_streaks)])):
        first_processed_streak = len(only_long_streaks)-np.min([num_streaks_processed, len(only_long_streaks)])
        last_processed_streak = len(only_long_streaks)
        print('Processing last {} out of {} streaks'.format(last_processed_streak-first_processed_streak, len(only_long_streaks)))
    else:
        first_processed_streak = starting_streak
        last_processed_streak = starting_streak+num_streaks_processed
    short_streaks = [a for a in streak_list if a not in only_long_streaks]

    v_list = list()
    dim_list = list()
    cap_list = list()

    v_std_list = list()
    dim_std_list = list()

    cmap = get_cmap(len(only_long_streaks))

    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    for i, streak in enumerate(only_long_streaks[first_processed_streak:last_processed_streak]):
        pos = sorted([p.spatial_position for p in streak.particle_streak], key=lambda pos_entry: pos_entry[1])
        ax.scatter([p[0] for p in pos], [p[1] for p in pos], c=cmap(i), s=50)
        ax.plot([p[0] for p in pos], [p[1] for p in pos], c=cmap(i))  # , linewidth = streak.mean_diam*1e5)
        this_gaps = [q-p for (p, q) in zip(pos[:-2], pos[1:])]

        vs = [np.sqrt(g[1]**2+g[0]**2)*100 for g in this_gaps]
        v_list.append(np.median(vs))
        v_std_list.append(np.std(vs))

        s_majsiz = [s.majsiz*1e6 for s in streak.particle_streak]
        s_minsiz = [s.minsiz*1e6 for s in streak.particle_streak]
        dim_list.append(np.median(s_majsiz))
        dim_std_list.append(np.std(s_majsiz))
        cap_list.append(0.134*np.median([0.58*this_min/2*(1+0.95*(this_maj/this_min)**0.75) for this_min, this_maj in zip(s_minsiz, s_majsiz)]))

        # streak.plot_props()

    # for streak in short_streaks:
    #     pos = [p.spatial_position for p in streak.particle_streak]
    #     ax.scatter([p[0] for p in pos], [p[1] for p in pos], c='b', marker='<', alpha=0.3)

    ax.set_title('Sample fall tracks from holograms measurement 1, Feb 23 2018 (unfiltered)', fontsize=20)
    ax.set_xlabel('x in mm', fontsize=20)
    ax.set_ylabel('y in mm', fontsize=20)

    ax.grid('on')
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-3, 3])

    # fig, ax = plt.subplots(1)
    # for streak in short_streaks:
    #     pos = [p.spatial_position for p in streak.particle_streak]
    #     ax.scatter([p[0] for p in pos], [p[1] for p in pos], c='b', marker='<')

    amp_full, index_full = fit_powerlaw(dim_list, v_list)
    powerlaw = lambda x, amp, index: amp * (x ** index)

    # dims_spaced = np.arange(maxsize)
    dims_spaced = np.arange(np.ceil(np.max(dim_list)/10)*10)

    fig, ax = plt.subplots(1)
    # ax.errorbar(dim_list, v_list, xerr=dim_std_list, yerr=v_std_list, fmt='o')

    ax.scatter(dim_list, v_list)
    # ax.scatter(cap_list, v_list)

    ax.grid('on')
    # ax.plot(dims_spaced, powerlaw(dims_spaced, amp_full, index_full), label='Power Law Full', linewidth=3,
    #         zorder=1)

    # ax.set_xlim(0, max(dim_list)*1.1, ax.set_ylim(0, max(v_list)*1.1))

    save_flag = input('Save data to file?')
    if save_flag == 'Yes' or save_flag == 'yes':

        plotdir = path+'plots'
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)

        save_dict = {"folder": path, "dims": dim_list, "vs": v_list}
        pickle.dump(save_dict, open(path + 'vs_dim_data.dat', 'wb'))
        print('Data saved in '+path+'vs_dim_data.dat.')

        plt.savefig(plotdir+'v_graph.png')
        print('Graph saved in ' + plotdir + 'v_graph.png.')

    plt.show()


class FallParticle:
    def __init__(self, holonum, index_in_hologram, xpos, ypos, zpos, majsiz, minsiz):
        self.holonum = holonum
        self.index_in_hologram = index_in_hologram
        self.xpos = xpos
        self.ypos = ypos
        self.zpos = zpos
        self.spatial_position = np.asarray([self.xpos, self.ypos, self.zpos])
        self.majsiz = majsiz
        self.minsiz = minsiz
        self.aspr = majsiz/minsiz


class ParticleStreak:
    def __init__(self, initial_particle, first_guess_velocity):
        self.initial_particle = initial_particle
        self.particle_streak = [initial_particle]
        self.vel_vector = first_guess_velocity
        self.streak_length = self.get_streak_length()
        self.mean_diam = self.get_mean_diam()

    def add_particle(self, particle):
        self.particle_streak.append(particle)

    def get_streak_length(self):
        streak_covered_distance = np.sqrt((self.particle_streak[0].xpos - self.particle_streak[-1].xpos) ** 2 + (self.particle_streak[0].ypos - self.particle_streak[-1].ypos) ** 2 + (
            self.particle_streak[0].zpos - self.particle_streak[-1].zpos) ** 2)
        return streak_covered_distance

    def get_mean_diam(self):
        diameters = [c.majsiz for c in self.particle_streak]
        return np.mean(diameters)

    def plot_props(self):
        diameters = [c.majsiz for c in self.particle_streak]
        aspect_ratios = [c.aspr for c in self.particle_streak]

        f, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(diameters)
        ax2.plot(aspect_ratios)


def find_streak_particles(this_streak, particle_list, velocity, max_dist=1, max_ind=10, max_size_diff=0.05):
    initial_particle = this_streak.particle_streak[-1]

    predicted_position = np.asarray(initial_particle.spatial_position)+np.asarray(velocity)
    predicted_particle = initial_particle
    predicted_particle.xpos = predicted_position[0]
    predicted_particle.ypos = predicted_position[1]
    predicted_particle.zpos = predicted_position[2]
    dists = [dist(predicted_particle, pb) for pb in particle_list]
    size_diffs = [abs(pb.majsiz-predicted_particle.majsiz)/predicted_particle.majsiz for pb in particle_list]
    dists_s, sizes_s = (list(x) for x in zip(*sorted(zip(dists, size_diffs), key=lambda pair: pair[0])))
    try:
        dist_ind = 0
        extended = False
        while (dists_s[dist_ind] < max_dist) & (dist_ind < max_ind):
            closest = dists_s[dist_ind]
            index_closest = dists.index(closest)
            closest_particle = particle_list[index_closest]
            if sizes_s[dist_ind] < max_size_diff:
                this_streak.add_particle(closest_particle)
                extended = True
                ext_by = closest_particle
                break
            else:
                dist_ind += 1
        if not extended:
            ext_by = []

        return this_streak, extended, ext_by
    except IndexError:
        return this_streak, [], []


def dist(particle_a, particle_b):
    distance = np.sqrt((particle_a.xpos-particle_b.xpos)**2+(particle_a.ypos-particle_b.ypos)**2+(particle_a.zpos-particle_b.zpos)**2)
    return distance


def get_cmap(n, name='hsv'):
    # Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    # RGB color; the keyword argument name must be a standard mpl colormap name.
    return plt.cm.get_cmap(name, n)


def fit_powerlaw(x, y):

    x = [this_x for this_x, this_y in zip(x, y) if not(np.isnan(this_y))]
    y = [this_y for this_y in y if not(np.isnan(this_y))]

    logx = np.log10(x)
    logy = np.log10(y)

    fitfunc = lambda p, t_x: p[0]+p[1]*t_x
    errfunc = lambda p, t_x, t_y: (t_y-fitfunc(p, t_x))

    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit, args=(logx, logy), full_output=1)

    pfinal = out[0]
    # covar = out[1]

    index = pfinal[1]
    amp = 10.0**pfinal[0]
    # pl = amp * (x**index)

    return amp, index


def refine_streaks(streak_list, angle_leniency, length_leniency):
    for streak in streak_list:
        if len(streak.particle_streak) > 1:
            angles = [np.angle(s1.xpos - s2.xpos + 1j * (s1.ypos - s2.ypos), deg=True) for s1, s2 in
                      zip(streak.particle_streak[:-1], streak.particle_streak[1:])]
            median_ang = np.median(angles)
            outlier = np.where(np.abs(np.asarray(angles)-median_ang) > angle_leniency)
            for o in outlier[0].tolist()[::-1]:
                if o == 0:
                    del streak.particle_streak[0]
                else:
                    del streak.particle_streak[o+1]
    return streak_list


if __name__ == "__main__":
    main()