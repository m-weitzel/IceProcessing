import scipy.io as sio
from scipy import optimize
from matplotlib import pyplot as plt
import numpy as np
import time
import pickle
import os


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


class particleStreak:
    def __init__(self, initial_particle, first_guess_velocity):
        self.initial_particle = initial_particle
        self.particle_streak = [initial_particle]
        self.vel_vector = first_guess_velocity
        self.streak_length = self.get_streak_length()
        self.mean_diam = self.get_mean_diam()

    def add_particle(self, particle):
        self.particle_streak.append(particle)

    def get_streak_length(self):
        first_pos = self.particle_streak[0].spatial_position
        last_pos = self.particle_streak[-1].spatial_position
        dist = np.sqrt((self.particle_streak[0].xpos - self.particle_streak[-1].xpos) ** 2 + (self.particle_streak[0].ypos - self.particle_streak[-1].ypos) ** 2 + (
            self.particle_streak[0].zpos - self.particle_streak[-1].zpos) ** 2)
        return dist

    def get_mean_diam(self):
        diams = [c.majsiz for c in self.particle_streak]
        return np.mean(diams)

    def plot_props(self):
        diams = [c.majsiz for c in self.particle_streak]
        asprs = [c.aspr for c in self.particle_streak]

        f, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(diams)
        ax2.plot(asprs)


def find_streak_particles(this_streak, particle_list, velocity, max_dist=1, max_ind=10, max_sizdiff=0.05):
    initial_particle = this_streak.particle_streak[-1]

    predicted_position = np.asarray(initial_particle.spatial_position)+np.asarray(velocity)
    predicted_particle = initial_particle
    predicted_particle.xpos = predicted_position[0]
    predicted_particle.ypos = predicted_position[1]
    predicted_particle.zpos = predicted_position[2]
    dists = [dist(predicted_particle, pb) for pb in particle_list]
    sizdiffs = [abs(pb.majsiz-predicted_particle.majsiz)/predicted_particle.majsiz for pb in particle_list]
    if (len(dists) < 1) | (len(sizdiffs) < 1):
        print('empty')
    dists_s, sizs_s = (list(x) for x in zip(*sorted(zip(dists, sizdiffs), key=lambda pair: pair[0])))

    try:
        dist_ind = 0
        extended = False
        while (dists_s[dist_ind] < max_dist) & (dist_ind < max_ind):
            closest = dists_s[dist_ind]
            index_closest = dists.index(closest)
            closest_particle = particle_list[index_closest]
            if sizs_s[dist_ind] < max_sizdiff:
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


def plot_stuff(a, pxl_size, xp, yp):

    # inversion
    tmp = xp
    xp = max(yp) - yp
    yp = max(tmp) - tmp

    times = a['times']
    area = a['area'] * pxl_size ** 2
    majsiz = a['majsiz'] * pxl_size
    # minsiz = a['minsiz'] * pxl_size

    plt.scatter(xp, yp, c=times)

    plt.xlim([min(xp), max(xp)])
    plt.ylim([min(yp), max(yp)])

    plt.figure()
    plt.scatter(majsiz, area)
    plt.xlim([min(majsiz), max(majsiz)])
    plt.ylim([min(area), max(area)])

    plt.show()


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def streak_video(ps):
    ts = ps['times']
    pos = list(zip(list(ps['xp']), list(ps['yp'])))

    ax, fig = plt.subplots()
    current_crystal_ind = 0
    for i in range(len(ts)):
        plt.title('Frame %d' % i)
        while i >= ts[current_crystal_ind]:
            plt.scatter(pos[current_crystal_ind][0], pos[current_crystal_ind][1])
            if current_crystal_ind < len(ts):
                current_crystal_ind += 1
            else:
                break
        plt.show()
        time.sleep(5)


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


def main():
    # path = '/uni-mainz.de/homes/maweitze/FallSpeedHolograms/2510/whole/whole/'
    # path = '/ipa2/holo/mweitzel/HIVIS_Holograms/Tests_21Feb/'
    # path = '/ipa2/holo/mweitzel/HIVIS_Holograms/Prev23Feb/'
    path = '/ipa2/holo/mweitzel/HIVIS_Holograms/Meas28Feb/M2/'
    filename_ps = 'ps_bypredict.mat'


    try:
        tmp = pickle.load(open(path+'vs_dim_data.dat', 'rb'))
        dim_list = tmp['dims']
        v_list = tmp['vs']
    except (FileNotFoundError, EOFError):

        a = sio.loadmat(path+filename_ps)

        tmp = a['xp']*1000
        a['xp'] = a['yp']*1000
        a['yp'] = tmp

        pxl_size = 1
        min_streak_length = 5
        maxsize = 200
        max_sizdiff = 0.05
        max_dist_from_predict = 0.5
        base_velocity_guess = [0, 0, 0]

        p_list = list()
        for k in range(0, len(a['times'])):
            p_list.append(FallParticle(a['times'][k][0], 0, a['xp'][k][0]*pxl_size, a['yp'][k][0]*pxl_size, a['zp'][k][0]*pxl_size, a['majsiz'][k][0], a['minsiz'][k][0]))

        last_holonum = p_list[-1].holonum

        holonums = range(0, last_holonum+1)

        ps_in_holos = [[] for _ in holonums]

        for p in p_list:
            ps_in_holos[p.holonum].append(p)

        streak_list = list()

        while p_list:
            this_particle = p_list[0]
            velocity_guess = base_velocity_guess
            new_streak = particleStreak(this_particle, velocity_guess)
            p_list.remove(this_particle)
            extended = True
            count_test = 0
            while extended:
                try:
                    if len(ps_in_holos[new_streak.particle_streak[-1].holonum+1])<1:
                        extended=False
                        ext_by = []
                        print('Empty hologram, ending streaks.')
                    else:
                        new_streak, extended, ext_by = find_streak_particles(new_streak, ps_in_holos[new_streak.particle_streak[-1].holonum+1],
                                                                             velocity_guess, max_dist=max_dist_from_predict, max_sizdiff=max_sizdiff)
                        count_test += 1
                except IndexError:
                    print('Last hologram reached, quitting...')
                    extended = False
                    ext_by = []

                if extended:
                    p_list.remove(ext_by)
                    ps_in_holos[new_streak.particle_streak[-2].holonum+1].remove(ext_by)
                    if len(new_streak.particle_streak) > 2:
                        velocity_guess = new_streak.particle_streak[-1].spatial_position-new_streak.particle_streak[-2].spatial_position
                        print('Applying self-determined velocity guess. {}'.format(tuple(velocity_guess)))
            streak_list.append(new_streak)
            if not extended:
                print('Completed streak with '+str(len(new_streak.particle_streak))+' elements.')

        only_long_streaks = [a for a in streak_list if (len(a.particle_streak) >= min_streak_length)]#&(a.get_streak_length() > min_streak_length)]
        # short_streaks = [a for a in streak_list if len(a.particle_streak) < min_streak_length]
        short_streaks = [a for a in streak_list if a not in only_long_streaks]

        v_list = list()
        dim_list = list()

        v_std_list = list()
        dim_std_list = list()

        cmap = get_cmap(len(only_long_streaks))

        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        for i, streak in enumerate(only_long_streaks[:50]):
            pos = sorted([p.spatial_position for p in streak.particle_streak], key=lambda pos_entry: pos_entry[1])
            ax.scatter([p[0] for p in pos], [p[1] for p in pos], c=cmap(i), s=50)
            ax.plot([p[0] for p in pos], [p[1] for p in pos], c=cmap(i))#, linewidth=streak.mean_diam*1e5)
            this_gaps = [q-p for (p,q) in zip(pos[:-2], pos[1:])]

            vs = [np.sqrt(g[1]**2+g[0]**2)*100 for g in this_gaps]
            v_list.append(np.median(vs))
            v_std_list.append(np.std(vs))

            s_majsiz = [s.majsiz*1e6 for s in streak.particle_streak]
            dim_list.append(np.median(s_majsiz))
            dim_std_list.append(np.std(s_majsiz))
            # streak.plot_props()

        # for streak in short_streaks:
        #     pos = [p.spatial_position for p in streak.particle_streak]
        #     ax.scatter([p[0] for p in pos], [p[1] for p in pos], c='b', marker='<', alpha=0.3)


        ax.set_title('Sample fall tracks from holograms measurement 1, Feb 23 2018 (unfiltered)', fontsize=20)
        ax.set_xlabel('x in mm', fontsize=20)
        ax.set_ylabel('y in mm', fontsize=20)
        # ax.set_title('Streaks in Exp. 1, detected with Auto Threshold = 1.0, 60 fps')
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
    ax.grid('on')
    ax.plot(dims_spaced, powerlaw(dims_spaced, amp_full, index_full), label='Power Law Full', linewidth=3,
            zorder=1)
    ax.set_xlim(0, max(dim_list)*1.1, ax.set_ylim(0, max(v_list)*1.1))

    save_flag = input('Save data to file?')
    if save_flag == 'Yes' or save_flag == 'yes':

        plotdir=path+'plots'
        # plotdir = '/uni-mainz.de/homes/maweitze/CCR/test/'
        if not os.path.exists(plotdir):
            os.makedirs(plotdir)

        save_dict = {"folder": path, "dims": dim_list, "vs": v_list}
        pickle.dump(save_dict, open(path + 'vs_dim_data.dat', 'wb'))
        print('Data saved in '+path+'vs_dim_data.dat.')

        plt.savefig(plotdir+'v_graph.png')
        print('Graph saved in ' + plotdir + 'v_graph.png.')

    plt.show()

if __name__ == "__main__":
    main()