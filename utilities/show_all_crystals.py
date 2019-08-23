from matplotlib import pyplot as plt
from Speed.Holography.generate_fallstreaks import ParticleStreak, FallParticle, refine_streaks, get_folder_framerate, eta
from utilities.make_pretty_figure import savefig_ipa, imshow_in_figure, density_plot
import numpy as np
# from utilities.plot_size_distribution import plot_size_dist
from utilities.tempsave_extracted_streaks import *
# import pandas as pd
import matplotlib.gridspec as gridspec
from import_plot_ps_fallstreaks import p3d, load_streaks


folder_list = list()
# folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/Prev23Feb/')  # Columnar, Irregular
# folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/Meas28Feb/M2/')  # Dendritic
# folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/Meas01Mar/')  # Dendritic
folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/Meas22May/')   # Columnar
# folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/Meas23May/M2/')   # Columnar
# folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/2905M1/')
# folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/2905/ps/seq2')
# folder_list.append('/ipa/holo/mweitzel/HIVIS_Holograms/26Sep/')

angle_leniency_deg = 20
min_streak_length = 3  # for separation between short and long streaks

info_list = list()

full_streak_list = list()
only_orient_list = list()

list_of_folder_streak_lists = load_streaks(folder_list, 'allcrystal_streak_save.dat')
for l in list_of_folder_streak_lists:
    full_streak_list.extend(l)
    try:
        a = l[0].initial_particle.orient
        only_orient_list.extend(l)
    except AttributeError:
        print('No orientation in folder.')
    except IndexError:
        pass

ips = [s.initial_particle for s in full_streak_list]
ims = [p.partimg for p in ips]
sizs = [i.shape for i in ims]
total_s = [s[0]*s[1] for s in sizs]

indexes = list(range(len(full_streak_list)))
indexes.sort(key=total_s.__getitem__, reverse=True)
this_folders_streak_list = list(map(full_streak_list.__getitem__, indexes))
ims = list(map(ims.__getitem__, indexes))
sizs = list(map(sizs.__getitem__, indexes))

delta_an = list()
gt_thresh_counter = 0
angle_thresh = 20

for prtstrk in [s.particle_streak for s in only_orient_list]:
    # an_max = np.max([p.orient for p in prtstrk])
    # an_min = np.min([p.orient for p in prtstrk])
    # delta = an_max-an_min

    an_b4 = prtstrk[0].orient
    an_af = prtstrk[-1].orient
    delta = an_af-an_b4
    if delta > 90:
        delta = 180-delta
    if prtstrk[0].aspr > 2:
        delta_an.append(delta)
    if delta > angle_thresh:
        gt_thresh_counter += 1
        gs1 = gridspec.GridSpec(len(prtstrk), 1)
        for m, p in enumerate(prtstrk):
            ax = plt.subplot(gs1[m, 0])
            ax.imshow(np.abs(p.partimg), cmap='bone')
    # _, ax = imshow_in_figure()
    # p3d(ax, [p.spatial_position for p in prtstrk])

print('{0:.1f}% of particles changed angle more than {1:.0f}°.'.format(gt_thresh_counter/len(only_orient_list)*100, angle_thresh))
fig, ax = imshow_in_figure()
ax.hist(delta_an)
plt.show()

# mean_ang = [np.rad2deg(s.mean_angle) for s in full_streak_list]
# pos = [s.particle_streak[np.int(np.ceil(len(s.particle_streak)/2))].spatial_position for s in full_streak_list]
# density_plot(mean_ang, pos, 2.22, (2048, 2592))
#
# orient_ips = [s.initial_particle for s in only_orient_list]
# fig2, ax = plot_size_dist([np.abs(p.orient) for p in orient_ips], 20, normed=True)
# ax.grid(which='major', alpha=0.5)
# savefig_ipa(fig2, 'OrientationHistogram')
# plt.show()


def column_gallery(ips, ims):
    # [~, ord] = sort(total_s, 'descend');

    m = 0
    n = 0

    N_row = 20
    M_image = 240

    while m < len(ips)-1:
        n = 0
        while (n < M_image) and ((m+n) < (len(ips)-1)):
            batch = np.arange(m+n, np.min([m+n+N_row, len(ips)-1]))
            t_ims = [ims[i] for i in batch]
            t_a = [s[0] for s in [sizs[j] for j in batch]]
            t_b = [s[1] for s in [sizs[j] for j in batch]]

            max_a = np.max(t_a)
            t_ims[0] = np.pad(t_ims[0], ((max_a-t_a[0], 0), (0, 0)), 'constant', constant_values=np.nan)
            this_row = t_ims[0]
            for i in np.arange(1, len(batch)):
               t_ims[i] = np.pad(t_ims[i], ((np.max(max_a-t_a[i], 0), 0), (0, 0)), 'constant', constant_values=np.nan)
               this_row = np.concatenate((this_row, t_ims[i]), axis=1)

            if n == 0:
                print('Starting new row, n={0:.0f}, m={1:.0f}.'.format(n, m))
                full_im = this_row
            else:
                print('Continuing row, n={0:.0f}, m={1:.0f}'.format(n, m))
                if this_row.shape[1] < full_im.shape[1]:
                    this_row = np.pad(this_row, ((0, 0), (0, np.max(full_im.shape[1]-this_row.shape[1]))), 'constant', constant_values=np.nan)
                else:
                    full_im = np.pad(full_im, ((0, 0), (0, this_row.shape[1]-full_im.shape[1])), 'constant', constant_values=np.nan)

                tmp = np.concatenate((full_im, this_row))
                full_im = tmp
            n += 30
        print('Row complete, n={0:.0f}, m={1:.0f}'.format(n, m))
        fig, ax = imshow_in_figure(figspan=(18, 10), dpi=200)
        ax.imshow(np.abs(full_im), cmap='bone')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if m == 0:
            savefig_ipa(fig, 'FallingColumns')
        # ax.grid(True)
        m += M_image
