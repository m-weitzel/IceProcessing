import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np

class FallPrtcl:
    def __init__(self, holonum, index_in_hologram, xpos, ypos, zpos, majsiz, minsiz):
        self.holonum = holonum
        self.index_in_hologram = index_in_hologram
        self.xpos = xpos
        self.ypos = ypos
        self.zpos = zpos
        self.spatial_position = np.asarray([self.xpos, self.ypos, self.zpos])
        self.majsiz = majsiz
        self.minsiz = minsiz


class PrtclStreak:
    def __init__(self, initial_prtcl, first_guess_velocity):
        self.initial_prtcl = initial_prtcl
        self.prtcl_streak = [initial_prtcl]
        self.vel_vector = first_guess_velocity

    def add_particle(self, prtcl):
        self.prtcl_streak.append(prtcl)


def find_streak_particles(this_streak, prtcl_list, velocity, max_dist=0.01):
    initial_prtcl = this_streak.prtcl_streak[-1]

    predicted_position = np.asarray(initial_prtcl.spatial_position)+np.asarray(velocity)
    predicted_prtcl = initial_prtcl
    predicted_prtcl.xpos = predicted_position[0]
    predicted_prtcl.ypos = predicted_position[1]
    predicted_prtcl.zpos = predicted_position[2]
    dists = [dist(predicted_prtcl, pb) for pb in prtcl_list]
    dists_s = sorted(dists)
    try:
        closest = dists_s[0]
        index_closest = dists.index(closest)
        closest_prtcl = prtcl_list[index_closest]

        if dists_s[0] < max_dist:
            this_streak.add_particle(closest_prtcl)
            extended = True
            ext_by = closest_prtcl
        else:
            extended = False
            ext_by = []
        return this_streak, extended, ext_by
    except IndexError:
        return this_streak, [], []


def dist(prtcl_a, prtcl_b):
    distance = np.sqrt((prtcl_a.xpos-prtcl_b.xpos)**2+(prtcl_a.ypos-prtcl_b.ypos)**2+(prtcl_a.zpos-prtcl_b.zpos)**2)
    return distance

def plot_stuff(a, pxl_size, xp, yp):

    # inversion
    tmp = xp
    xp = max(yp) - yp
    yp = max(tmp) - tmp

    times = a['times']
    area = a['area'] * pxl_size ** 2
    majsiz = a['majsiz'] * pxl_size
    minsiz = a['minsiz'] * pxl_size

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

def main():
    a=sio.loadmat('/uni-mainz.de/homes/maweitze/FallSpeedHolograms/2510_2/ps.mat')

    pxl_size = 4.8

    p_list = list()
    for k in range(0, len(a['times'])):
        p_list.append(FallPrtcl(a['times'][k][0], 0, a['xp'][k][0]*pxl_size, a['yp'][k][0]*pxl_size, a['zp'][k][0]*pxl_size, a['majsiz'][k][0], a['minsiz'][k][0]))

    last_holonum = p_list[-1].holonum

    holonums = range(0, last_holonum+1)

    ps_in_holos = [[] for i in holonums]

    for p in p_list:
        ps_in_holos[p.holonum].append(p)

    streak_list = list()

    while p_list:
        this_prtcl = p_list[0]
        velocity_guess= [-0.005, 0, 0]  # cm in 1/60 s
        new_streak = PrtclStreak(this_prtcl, velocity_guess)
        p_list.remove(this_prtcl)
        extended = True
        while extended:
            try:
                new_streak, extended, ext_by = find_streak_particles(new_streak, ps_in_holos[new_streak.prtcl_streak[-1].holonum+1], velocity_guess)
            except IndexError:
                print('Last hologram reached, quitting...')
                extended = False
                ext_by = []

            if extended:
                p_list.remove(ext_by)
                ps_in_holos[new_streak.prtcl_streak[-2].holonum+1].remove(ext_by)
                if len(new_streak.prtcl_streak) > 3:
                    velocity_guess = new_streak.prtcl_streak[-1].spatial_position-new_streak.prtcl_streak[-2].spatial_position
                    print('Applying self-determined velocity guess.')
        streak_list.append(new_streak)
        if not(extended):
            print('Completed streak with '+str(len(new_streak.prtcl_streak))+' elements.')

    lens = [len(a.prtcl_streak) for a in streak_list]

    only_long_streaks = [a for a in streak_list if len(a.prtcl_streak)>=5]
    short_streaks = [a for a in streak_list if len(a.prtcl_streak)<5]

    cmap = get_cmap(len(only_long_streaks))
    for i, streak in enumerate(only_long_streaks):
        pos = [p.spatial_position for p in streak.prtcl_streak]
        plt.scatter([p[0] for p in pos], [p[1] for p in pos], c=cmap(i))

    plt.figure()
    for streak in short_streaks:
        pos = [p.spatial_position for p in streak.prtcl_streak]
        plt.scatter([p[0] for p in pos], [p[1] for p in pos], c='b', marker='<')

    plt.show()


if __name__ == "__main__":
    main()