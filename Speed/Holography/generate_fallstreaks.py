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


def main():
    path = '/ipa2/holo/mweitzel/HIVIS_Holograms/Meas23May/M2/'
    filename_ps = 'ps_bypredict.mat'

    a = sio.loadmat(os.path.join(path, filename_ps))

    # Adjust to how the camera was tilted (90°) to have true spatial directions in fall images, multiply by 1000 to get mm
    tmp = a['xp']*1000
    a['xp'] = a['yp']*1000
    a['yp'] = tmp
    a['zp'] = a['zp']*1000

    # General parameters
    pxl_size = 1

    # Properties for finding streaks
    max_size_diff = 0.1
    max_dist_from_predict = 0.5
    static_velocity = False

    if static_velocity:
        base_velocity_guess = [0, -1.6, 0]
    else:
        y_vel = lambda x: -0.08*x


        # End of properties

    p_list = list()
    for k in range(0, len(a['times'])):
        p_list.append(FallParticle(a['times'][k][0], 0, a['xp'][k][0]*pxl_size,
                                   a['yp'][k][0]*pxl_size, a['zp'][k][0]*pxl_size,
                                   a['prediction'][k], a['majsiz'][k][0],
                                   a['minsiz'][k][0], a['ims'][k][0]))

    last_holonum = p_list[-1].holonum
    holonums = range(0, last_holonum+1)
    ps_in_holos = [[] for _ in holonums]

    for p in p_list:
        ps_in_holos[p.holonum].append(p)

    habit_list = [p.habit for p in p_list]
    different_habits = list(set(habit_list))

    p_by_habits = dict()
    streak_list = list()

    for hab in different_habits:
        p_by_habits[hab] = [p for p in p_list if p.habit == hab]

    for hab in different_habits:
        processing_p_list = p_by_habits[hab]
        while processing_p_list:
            this_particle = processing_p_list[0]
            if static_velocity:
                velocity_guess = base_velocity_guess
            else:
                y_velocity = y_vel(this_particle.majsiz)
                velocity_guess = [0, y_velocity, 0]

            new_streak = ParticleStreak(this_particle, velocity_guess)
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
                                                                             velocity_guess, max_dist=max_dist_from_predict, max_size_diff=max_size_diff)
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
                        velocity_guess = new_streak.particle_streak[-1].spatial_position-new_streak.particle_streak[-2].spatial_position
            if len(new_streak.particle_streak) >= min_length:
                streak_list.append(new_streak)

    save_flag = input('Save data to file?')
    if save_flag == 'Yes' or save_flag == 'yes':

        plot_dir = os.path.join(path, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        save_dict = {"folder": path, "streaks": streak_list}
        pickle.dump(save_dict, open(os.path.join(path, 'streak_data.dat'), 'wb'), -1)
        print('{} streaks saved in '.format(len(streak_list))+os.path.join(path, 'streak_data.dat.'))

    plt.show()


class FallParticle:
    def __init__(self, holonum, index_in_hologram, xpos, ypos, zpos, habit, majsiz, minsiz, *args):
        self.holonum = holonum
        self.index_in_hologram = index_in_hologram
        self.xpos = xpos
        self.ypos = ypos
        self.zpos = zpos
        self.habit = habit
        self.spatial_position = np.asarray([self.xpos, self.ypos, self.zpos])
        self.majsiz = majsiz
        self.minsiz = minsiz
        self.aspr = majsiz/minsiz
        if len(args) > 0:
            self.partimg = args[0]


class ParticleStreak:
    def __init__(self, initial_particle, first_guess_velocity):
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

    def get_projected_velocity(self, mean_angle):
        pos = sorted([p.spatial_position for p in self.particle_streak], key=lambda pos_entry: pos_entry[1])
        this_gaps = [q - p for (p, q) in zip(pos[:-1], pos[1:])]
        abs_v_list = [np.sqrt(g[1] ** 2 + g[0] ** 2) * 100 for g in this_gaps]  # absolute velocity
        new_angles = [a - mean_angle for a in self.angles]
        v_list = [v * np.cos(beta) for v, beta in zip(abs_v_list, new_angles)]
        return v_list


def find_streak_particles(this_streak, particle_list, velocity, max_dist=1, max_ind=10, max_size_diff=0.05):
    initial_particle = this_streak.particle_streak[-1]

    predicted_position = np.asarray(initial_particle.spatial_position)+np.asarray(velocity)
    predicted_particle = initial_particle
    predicted_particle.xpos = predicted_position[0]
    predicted_particle.ypos = predicted_position[1]
    predicted_particle.zpos = predicted_position[2]
    dists = [dist(predicted_particle, pb, ignore_zpos=True) for pb in particle_list]

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


def dist(particle_a, particle_b, ignore_zpos=False):
    if ignore_zpos:
        distance = np.sqrt((particle_a.xpos-particle_b.xpos)**2+(particle_a.ypos-particle_b.ypos)**2)
    else:
        distance = np.sqrt((particle_a.xpos - particle_b.xpos)**2+(particle_a.ypos - particle_b.ypos)**2
                   +(particle_a.zpos - particle_b.zpos) ** 2)
    return distance


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
                    del streak.particle_streak[o+1:]
    return streak_list


if __name__ == "__main__":
    main()