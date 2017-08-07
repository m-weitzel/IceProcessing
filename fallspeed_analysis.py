import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from IceSizing import MicroImg
import os
import numpy as np
import pickle
from scipy.stats import norm
import cv2

folder = '/uni-mainz.de/homes/maweitze/CCR/1107/M1/'

fall_folder = folder+'Fall'
folder_list = os.listdir(fall_folder)
cont_real = list()
fall_dist = list()

try:
    tmp = pickle.load(open(folder+'fall_speed_data.dat', 'rb'))
    cont_real = tmp[0]
    fall_dist = tmp[1]

except FileNotFoundError:
    print('No old data file found, starting from scratch.')
    for filename in folder_list:
        if '_cropped' in filename:
            img = MicroImg('Streak', fall_folder, filename,
                           thresh_type=('Bin', -225), minsize=75, maxsize=1000, dilation=10)

            dims = img.dimensions
            conts = img.contours

            for dim, cont in zip(dims, conts):
                if dim['Short Axis'] < 4:
                    cont_real.append(cont)
                    fall_dist.append(dim['Long Axis'])

            img.contours = cont_real
            print('Processed '+filename)
            plt.imshow(img.processed_image)
            plt.savefig(fall_folder+'/'+filename+'_processed.png')

    pickle.dump((cont_real, fall_dist), open(folder+'fall_speed_data.dat','wb'))

tmp = pickle.load(open(folder+'mass_dim_data.dat','rb'))

area_eq_diam_list = list()
max_diam_list = list()
mass_list = list()
dropdiam_list = list()

for obj in tmp['crystal']:
    area_eq_diam_list.append(2*np.sqrt(obj['Area']/np.pi))
    max_diam_list.append(obj['Long Axis'])
    mass_list.append(np.pi/6*obj['Drop Diameter']**3)
    dropdiam_list.append(obj['Drop Diameter'])

pixel_size    = 23.03 # in µm
exposure_time = 85000 # in µs

vs = np.asarray(fall_dist)*pixel_size/exposure_time*100 # in cm/s
# plt.imshow(img.processed_image)

n_bins = 25

v_max = 3
ae_max = 75
mxdim_max = 105
mass_max = 87500
dropdiam_max = 75

def plot_param_hist(ax, list_of_vals, max_val, n_bins, unit):
    bins = max_val/n_bins*np.arange(n_bins)
    (mu, sigma) = norm.fit(list_of_vals)
    n, bins, _ = ax.hist(list_of_vals, bins=bins, normed=1)
    y = mlab.normpdf(bins, mu, sigma)
    ax.plot(bins, y, 'g--', linewidth=2)
    ax.axvline(x=mu, ymax=max(y)/ax.get_ylim()[1], color='r', linewidth=2)
    ax.axvline(x=mu-sigma, ymax=max(y)/ax.get_ylim()[1], color='r', linewidth=2)
    ax.axvline(x=mu+sigma, ymax=max(y)/ax.get_ylim()[1], color='r', linewidth=2)
    ax.axhline(y=max(y), xmin=(mu-sigma)/ax.get_xlim()[1], xmax=(mu+sigma)/ax.get_xlim()[1], color='r', linewidth=2)

    ax.text(0.95*float(ax.get_xlim()[1]), 0.95*float(ax.get_ylim()[1]), 'Mean:${0:.3f} \pm {1:.3f}$ {2:s}'.format(mu, sigma, unit),
             bbox=dict(facecolor='red', alpha=0.2), horizontalalignment='right', verticalalignment='top')

fig, axs = plt.subplots(2, 2)
ax = axs[0][0]
plot_param_hist(ax, vs, v_max, n_bins, 'cm/s')
ax.set_title('Terminal Velocity in cm/s')
ax.set_xlabel('Terminal Velocity in cm/s')
ax.set_ylabel('Count')

ax = axs[0][1]
plot_param_hist(ax, area_eq_diam_list, ae_max, n_bins, '$\mu m$')
ax.set_title('Area equivalent diameter in um')
ax.set_xlabel('Area equivalent diameter in um')
ax.set_ylabel('Count')

ax = axs[1][0]
plot_param_hist(ax, max_diam_list, mxdim_max, n_bins, '$\mu m$')
ax.set_title('Maximum diameter in um')
ax.set_xlabel('Maximum diameter in um')
ax.set_ylabel('Count')

ax = axs[1][1]
plot_param_hist(ax, dropdiam_list, dropdiam_max, n_bins, '$\mu m$')
ax.set_title('Drop diameter in um')
ax.set_xlabel('Drop diameter in um')
ax.set_ylabel('Count')

plt.suptitle('Histogram Overview for '+folder[-8:], fontsize=12)

plt.savefig(folder + 'histogram.png')
plt.show()

def get_angles(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurd = cv2.GaussianBlur(img, (15, 15), 0)
    grad = cv2.Laplacian(blurd, -1)
    # grad = cv2.Sobel(blurd, -1, 1, 0, ksize=15)

    edges = cv2.Canny(grad, 3, 9, apertureSize=3)
    lines = cv2.HoughLines(grad, 1, np.pi / 180, 200)

    return lines[0][1]