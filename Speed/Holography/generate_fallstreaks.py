""" Contains the classes FallParticle and ParticleStreak.
Loads a "ps_bypredict.mat" file which contains a matlab array of particle objects with descriptive properties. From those,
 find_streak_particles tries to find objects in (temporally) consecutive holograms with a predetermined vague spatial
 relationship to each other. Generally, this relationship is determined from an estimated velocity as a certain distance
 away following gravity. Objects within a certain distance of that estimated location are then accepted.
 Groups of thusly connected FallParticles are considered a ParticleStreak.
 A dictionary of ParticleStreaks is eventually saved as "streak_data.dat" if the user confirms."""

import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os
from datetime import datetime
from utilities.fit_powerlaw import fit_powerlaw


def main():
    path = '/ipa/holo/mweitzel/HIVIS_Holograms/Prev23FebOrient/'
    temperature = -12

    streak_list = generate_streaks(path, temperature)

    save_flag = input('Save {} streaks to file?'.format(len(streak_list)))
    if save_flag == 'Yes' or save_flag == 'yes':

        plot_dir = os.path.join(path, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        print('Saving {} streaks in '.format(len(streak_list))+os.path.join(path, 'streak_data.dat.'))
        save_dict = {"folder": path, "streaks": streak_list, "temperature": temperature}
        pickle.dump(save_dict, open(os.path.join(path, 'streak_data_orient.dat'), 'wb'), -1)

    plt.show()


class FallParticle:
    def __init__(self, holonum, index_in_hologram, xpos, ypos, zpos, habit, majsiz, minsiz, area, orient, *args):
        self.holonum = holonum
        self.index_in_hologram = index_in_hologram
        self.xpos = xpos
        self.ypos = ypos
        self.zpos = zpos
        self.habit = 'Unclassified' if habit == 'Particle_nubbly' else habit
        self.spatial_position = np.asarray([self.xpos, self.ypos, self.zpos])
        self.majsiz = majsiz
        self.minsiz = minsiz
        self.aspr = majsiz/minsiz
        self.area = area
        self.orient = orient
        self.is_streak_particle = False
        if len(args) > 0:
            self.partimg = args[0]


class ParticleStreak:
    def __init__(self, initial_particle):
        self.initial_particle = initial_particle
        self.particle_streak = [initial_particle]
        self.streak_habit = initial_particle.habit
        self.streak_length = self.get_streak_length()
        self.angles = ()
        self.mean_angle = self.set_mean_angle()

    def add_particle(self, particle):
        self.particle_streak.append(particle)
        self.angles = self.get_angles()
        self.mean_angle = self.set_mean_angle()

    def set_mean_angle(self):
        return np.mean(self.angles)

    def get_streak_length(self):
        streak_covered_distance = np.sqrt((self.particle_streak[0].xpos - self.particle_streak[-1].xpos) ** 2 + (self.particle_streak[0].ypos - self.particle_streak[-1].ypos) ** 2 + (
            self.particle_streak[0].zpos - self.particle_streak[-1].zpos) ** 2)
        return streak_covered_distance

    def get_angles(self):
        pos = sorted([p.spatial_position for p in self.particle_streak], key=lambda pos_entry: pos_entry[1])
        this_gaps = [q - p for (p, q) in zip(pos[:-1], pos[1:])]
        angles = [np.arctan(-g[0] / g[1]) for g in this_gaps]
        return angles

    def get_projected_velocity(self, mean_angle, framerate):
        pos = sorted([p.spatial_position for p in self.particle_streak], key=lambda pos_entry: pos_entry[1], reverse=True)
        this_gaps = [q - p for (p, q) in zip(pos[:-1], pos[1:])]

        abs_v_list = [np.sqrt(g[1] ** 2 + g[0] ** 2) * framerate for g in this_gaps]  # absolute velocity
        # abs_v_list = [np.sqrt(g[1] ** 2 + g[0] ** 2) * 56 for g in this_gaps]
        new_angles = [a - mean_angle for a in self.angles]

        v_list = [v * np.cos(beta) for v, beta in zip(abs_v_list, new_angles)]

        return v_list


def generate_streaks(folder, temperature):
    filename_ps = 'ps_orient_bypredict.mat'

    a = sio.loadmat(os.path.join(folder, filename_ps))

    # Adjust to how the camera was tilted (90°) to have true spatial directions in fall images, multiply by 1000 to get mm
    tmp = a['xp']*1000
    a['xp'] = a['yp']*1000
    a['yp'] = tmp
    a['zp'] = a['zp']*1000

    # General parameters
    pxl_size = 1
    fr = get_folder_framerate(folder)

    # Properties for finding streaks
    min_length = 3                  # minimum number of consecutive particles to be considered a streak
    max_size_diff = 0.5             # relative size difference - 0.1 corresponds to 10%
    max_dist_from_predict = 1       # total, mostly in mm
    static_velocity = True          # True: Velocity fixed at vel_in_mm between two frames, False: automatically calculated (from Stokes)

    if static_velocity:
        vel_guess = lambda x, y: -x/y
        vel_in_mm_p_s = 60              # guess for velocity (in mm/s)
        base_velocity_guess = [0, vel_guess(vel_in_mm_p_s, fr), 0]
    else:
        # y_vel = lambda x: (-0.69*(x*1e3)**0.41)/60*1e3      # Locatelli&Hobbs Agg.s of unrimed assemblages of plates, side planes, ...
                                                            # v = 0.69*D^0.41, v in m/s, D in mm, 60 fps, y in mm
        # y_vel = lambda x: (-0.1 * (x * 1e-3) ** 0.05)/60 * 1e3
        rho_o = 1000
        y_vel = lambda x: -2*(x/2)**2*9.81*(rho_o-1.34)/(9*eta(temperature))/fr*1e3   # Stokes

    # End of properties

    initial_p_list = list()
    for k in range(0, len(a['times'])):
        try:
            initial_p_list.append(FallParticle(a['times'][k][0], 0, a['xp'][k][0]*pxl_size,
                                       a['yp'][k][0]*pxl_size, a['zp'][k][0]*pxl_size,
                                       a['prediction'][k].rstrip(), a['majsiz'][k][0],
                                       a['minsiz'][k][0], a['area'][k][0],
                                       a['angle'][k][0], a['ims'][k][0]))
        except KeyError:
            # print('No angle data in {}, neglecting.'.format(folder))
            initial_p_list.append(FallParticle(a['times'][k][0], 0, a['xp'][k][0]*pxl_size,
                                       a['yp'][k][0]*pxl_size, a['zp'][k][0]*pxl_size,
                                       a['prediction'][k].rstrip(), a['majsiz'][k][0],
                                       a['minsiz'][k][0], a['area'][k][0],
                                       0, a['ims'][k][0]))

    print('{} particles loaded.'.format(len(initial_p_list)))

    p_list = initial_p_list.copy()

    last_holonum = p_list[-1].holonum
    holonums = range(0, last_holonum+1)

    habit_list = [p.habit for p in p_list]
    different_habits = list(set(habit_list))

    p_by_habits = dict()
    streak_list = list()

    for hab in different_habits:
        p_by_habits[hab] = [p for p in p_list if p.habit == hab]

        processing_p_list = p_by_habits[hab]
        ps_in_holos = [[] for _ in holonums]
        for p in processing_p_list:
            ps_in_holos[p.holonum].append(p)

        while processing_p_list:
            this_particle = processing_p_list[0]
            if static_velocity:
                velocity_guess = base_velocity_guess
            else:
                y_velocity = y_vel(this_particle.majsiz)
                velocity_guess = [0, y_velocity, 0]

            new_streak = ParticleStreak(this_particle)
            processing_p_list.remove(this_particle)
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
                                                                             velocity_guess, max_dist=max_dist_from_predict, max_size_diff=max_size_diff, prevent_upwards=False)
                except IndexError:
                    # If last hologram is reached, end the streak
                    extended = False
                    ext_by = []

                if extended:
                    p_list.remove(ext_by)                                                 # remove the recently added particle from particle list
                    processing_p_list.remove(ext_by)                                      # remove the recently added particle from particle list
                    ps_in_holos[new_streak.particle_streak[-2].holonum+1].remove(ext_by)  # remove particle from list of particles in current (next) hologram
                    if len(new_streak.particle_streak) > 2:
                        # Starting with the third particle, velocity guess for finding new particles is derived from the last two found particles
                        old_guess = velocity_guess
                        velocity_guess = new_streak.particle_streak[-1].spatial_position-new_streak.particle_streak[-2].spatial_position
                        # print('Old guess: {}, new guess: {}'.format(old_guess, velocity_guess))
            if len(new_streak.particle_streak) >= min_length:
                streak_list.append(new_streak)
                for part in new_streak.particle_streak:
                    part.is_streak_particle = True
    return streak_list


def get_folder_framerate(folder):
    try:
        filelist = os.listdir(os.path.join(folder, 'holos'))
        filelist = [f for f in filelist if f.endswith('.png')]
        filelist.sort()
        gettime = lambda t: datetime.strptime(filelist[t][-19:-4], '%H-%M-%S-%f')
    except FileNotFoundError:
        filelist = os.listdir(os.path.join(folder, 'Fall'))
        filelist = [f for f in filelist if f.endswith('.png')]
        filelist.sort()
        gettime = lambda t: datetime.strptime(filelist[t][18:33], '%H-%M-%S-%f')

    start = gettime(0)
    end = gettime(-1)
    quart = gettime(np.int(len(filelist)/4))
    half = gettime(np.int(len(filelist)/2))
    threequart = gettime(np.int(3*len(filelist)/4))
    first_q = quart-start
    secnd_q = half-quart
    third_q = threequart-half
    fourt_q = end-threequart
    total_time = lambda t: t.seconds+t.microseconds*1e-6

    fr = np.int(len(filelist)/4)/np.median([total_time(first_q), total_time(secnd_q), total_time(third_q), total_time(fourt_q)])
    # fr = 54
    print('Frame rate for {0}: {1:.1f} fps'.format(folder, fr))
    return fr


def find_streak_particles(this_streak, particle_list, velocity, max_dist=1, max_ind=10, max_size_diff=0.05, prevent_upwards=True):
    initial_particle = this_streak.particle_streak[-1]

    predicted_position = np.asarray(initial_particle.spatial_position)+np.asarray(velocity)
    predicted_particle = initial_particle
    predicted_particle.xpos = predicted_position[0]
    predicted_particle.ypos = predicted_position[1]
    predicted_particle.zpos = predicted_position[2]

    dists = [dist(predicted_particle, pb, ignore_zpos=False) for pb in particle_list]

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
                if (closest_particle.spatial_position[1] > initial_particle.spatial_position[1]) & prevent_upwards:
                    print('Upwards fall!')
                    dist_ind += 1
                else:
                    this_streak.add_particle(closest_particle)
                    extended = True
                    ext_by = closest_particle
                    break
            else:
                dist_ind += 1
        if not extended:
            ext_by = []
        # print(dist_ind)
        return this_streak, extended, ext_by
    except IndexError:
        return this_streak, [], []


def dist(particle_a, particle_b, ignore_zpos=False):
    if ignore_zpos:
        distance = np.sqrt((particle_a.xpos-particle_b.xpos)**2+(particle_a.ypos-particle_b.ypos)**2)
    else:
        distance = np.sqrt((particle_a.xpos - particle_b.xpos)**2+(particle_a.ypos - particle_b.ypos)**2
                           + (particle_a.zpos - particle_b.zpos) ** 2)
    return distance


def eta(t):

    # ts = [c+273.15 for c in (-25, -10, 0, 25)]
    # e = [c*1e-6 for c in (15.88, 16.65, 17.15, 18.32)]
    # e = [c*1e-6 for c in (16.05, 16.85, 17.36, 18.62)]

    # pl = fit_powerlaw(ts, e)

    # eta = pl[0]*(t+273.15)**pl[1]

    eta = 18.6*1e-6*((t+273.15)/300)**0.76

    return eta


def refine_streaks(streak_list, angle_leniency):
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
                    del streak.particle_streak[o+1:]
    return streak_list


if __name__ == "__main__":
    main()