from utilities.IceSizing import MicroImg
from utilities.plot_size_distribution import plot_size_dist
from Speed.Imaging.fallspeed_analysis import load_v_data, load_mass_data
from Speed.Holography.generate_fallstreaks import ParticleStreak, FallParticle, refine_streaks, get_folder_framerate, eta
from matplotlib import pyplot as plt
import pickle
import os
import numpy as np

folder_list_holo = list()

folder_list_holo.append('/ipa/holo/mweitzel/HIVIS_Holograms/Meas22May/')   # Columnar
folder_list_holo.append('/ipa/holo/mweitzel/HIVIS_Holograms/Meas23May/M2/')   # Columnar
folder_list_holo.append('/ipa/holo/mweitzel/HIVIS_Holograms/2905M1/')
folder_list_holo.append('/ipa/holo/mweitzel/HIVIS_Holograms/26Sep/')

folder_list_img = list()
folder_list_img.append('/ipa/holo/mweitzel/Windkanal/Ice/CCR/Y2017/1107/M1')
folder_list_img.append('/ipa/holo/mweitzel/Windkanal/Ice/CCR/Y2017/0808/M1')
folder_list_img.append('/ipa/holo/mweitzel/Windkanal/Ice/CCR/Y2017/0908/M1')

v_list = list()
dim_list = list()

for folder in folder_list_holo:
    tmp = pickle.load(open(os.path.join(folder, 'streak_data.dat'), 'rb'))
    framerate = get_folder_framerate(tmp['folder'])
    this_folders_streak_list = tmp['streaks']
    print('Added {} streaks from {}.'.format(len(this_folders_streak_list), folder))
    this_folders_streak_list = refine_streaks(this_folders_streak_list, 5)
    this_folders_streak_list = [a for a in this_folders_streak_list if (len(a.particle_streak) >= 3)]
    this_folders_mean_angle = np.mean([s.mean_angle for s in this_folders_streak_list])
    for i, s in enumerate(this_folders_streak_list):
            dim_list.extend([p.majsiz * 1e6 for p in s.particle_streak])
            v_list.extend(s.get_projected_velocity(this_folders_mean_angle, framerate))

fall_dist_all = list()
diams = list()
for fldr in folder_list_img:
    fall_dist, _, _, _ = load_v_data(fldr)
    _, dim, _, _ = load_mass_data(fldr)
    fall_dist_all.extend(fall_dist)
    diams.extend(dim)

max_v = np.max((np.max(v_list), np.max(fall_dist_all)))

fig = plt.figure(figsize=(18, 10), dpi=100)

ax = plt.subplot2grid((2, 2), (0, 0))
ax.hist(v_list, 25, histtype='step', fill=True, linewidth=3, density=True, log=False, edgecolor='k', range=(0, 120))
ax.grid()
ax = plt.subplot2grid((2, 2), (0, 1))
ax.hist(dim_list, 25, histtype='step', fill=True, linewidth=3, density=True, log=False, edgecolor='k')
ax.grid()
ax = plt.subplot2grid((2, 2), (1, 0))
ax.hist(fall_dist_all, 25, histtype='step', fill=True, linewidth=3, density=True, log=False, edgecolor='k', range=(0, 120))
ax.grid()
ax = plt.subplot2grid((2, 2), (1, 1))
ax.hist(diams, 25, histtype='step', fill=True, linewidth=3, density=True, log=False, edgecolor='k')
ax.grid()

_, ax_v_holo = plot_size_dist(v_list, 25, lims=[(0, 120)], xlabel='Fall velocity in mm/s')
# _, ax_d_holo = plot_size_dist(dim_list, 25, xlabel='Diameter in um')
# _, ax_v_strk = plot_size_dist(fall_dist_all, 25, lims=[(0, 120)], xlabel='Fall velocity in mm/s')
# _, ax_d_strk = plot_size_dist(diams, 25, xlabel='Diameter in um')

# fig.show()
plt.show()