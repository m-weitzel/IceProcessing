import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.ticker as tck
from IceSizing import MicroImg
import os
import numpy as np
import pickle
from scipy.stats import norm
import cv2

folder = '/uni-mainz.de/homes/maweitze/CCR/0908/M1/'

pixel_size = 23.03      # in µm
exposure_time = 85000   # in µs

fall_folder = folder+'Fall'
folder_list = sorted(os.listdir(fall_folder))
folder_list = [f for f in folder_list if '.png' in f]
cont_real = list()
fall_dist = list()
orientation = list()
centerpt = list()
time_list = list()

try:
    tmp = pickle.load(open(folder+'fall_speed_data.dat', 'rb'))
    cont_real = tmp[0]
    fall_dist = tmp[1]
    orientation = tmp[2]
    centerpt = tmp[3]
    time_list = tmp[4]

except (FileNotFoundError, IndexError):
    print('No old data file found, starting from scratch.')
    try:
        os.mkdir(fall_folder+'/processed')
    except FileExistsError:
        pass
    for i, filename in enumerate(folder_list):
        if '_cropped' in filename:
            img = MicroImg('Streak', fall_folder, filename,
                           thresh_type=('Bin', -180), minsize=75, maxsize=10000, dilation=10)

            dims = img.data
            conts = img.contours

            for dim, cont in zip(dims, conts):
                if dim['Short Axis'] < 8:
                    cont_real.append(cont)
                    fall_dist.append(dim['Long Axis'])
                    orientation.append(dim['Orientation'])
                    centerpt.append(dim['Center Points'])
                    time_list.append([i])

            img.contours = cont_real
            print('Processed '+filename)
            # plt.imshow(img.processed_image)
            cv2.imwrite(fall_folder+'/processed/'+filename+'_processed.png', img.processed_image)

    pickle.dump((cont_real, fall_dist, orientation, centerpt, time_list), open(folder+'fall_speed_data.dat', 'wb'))

tmp = pickle.load(open(folder+'mass_dim_data.dat', 'rb'))

area_eq_diam_list = list()
max_diam_list = list()
mass_list = list()
dropdiam_list = list()

for obj in tmp['crystal']:
    area_eq_diam_list.append(2*np.sqrt(obj['Area']/np.pi))
    # area_eq_diam_list.append(0.58*obj['Short Axis']/2*(1+0.95*(obj['Long Axis']/obj['Short Axis'])**0.75))
    max_diam_list.append(obj['Long Axis'])
    mass_list.append(np.pi/6*obj['Drop Diameter']**3)
    dropdiam_list.append(obj['Drop Diameter'])

vs = np.asarray(fall_dist)*pixel_size/exposure_time*100  # in cm/s

projected_vs = [v*np.cos(o) for (v, o) in zip(vs, orientation)]

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
    n, bins, _ = ax.hist(list_of_vals, bins=bins)
    dx = bins[1] - bins[0]
    scale = len(list_of_vals)*dx
    y = mlab.normpdf(bins, mu, sigma) * scale
    ax.plot(bins, y, 'g--', linewidth=2)
    ax.axvline(x=mu, ymax=max(y)/ax.get_ylim()[1], color='r', linewidth=2)
    ax.axvline(x=mu-sigma, ymax=max(y)/ax.get_ylim()[1], color='r', linewidth=2)
    ax.axvline(x=mu+sigma, ymax=max(y)/ax.get_ylim()[1], color='r', linewidth=2)
    ax.axhline(y=max(y), xmin=(mu-sigma)/ax.get_xlim()[1], xmax=(mu+sigma)/ax.get_xlim()[1], color='r', linewidth=2)

    ax.text(0.95*float(ax.get_xlim()[1]), 0.95*float(ax.get_ylim()[1]), 'Mean:${0:.3f} \pm {1:.3f}$ {2:s}'.format(mu, sigma, unit),
             bbox=dict(facecolor='red', alpha=0.2), horizontalalignment='right', verticalalignment='top')

_, axs = plt.subplots(2, 2, figsize=(20, 12.5))
ax = axs[0][0]
plot_param_hist(ax, projected_vs, v_max, n_bins, 'cm/s')
ax.set_title('Terminal Velocity in cm/s')
ax.set_xlabel('Terminal Velocity in cm/s')
ax.set_ylabel('Count')

ax = axs[0][1]
plot_param_hist(ax, area_eq_diam_list, ae_max, n_bins, '$\mu m$')
ax.set_title('Area equivalent diameter in um')
ax.set_xlabel('Area equivalent diameter in um')
ax.set_ylabel('Count')

ax = axs[1][1]
plot_param_hist(ax, max_diam_list, mxdim_max, n_bins, '$\mu m$')
ax.set_title('Maximum diameter in um')
ax.set_xlabel('Maximum diameter in um')
ax.set_ylabel('Count')

ax = axs[1][0]
plot_param_hist(ax, dropdiam_list, dropdiam_max, n_bins, '$\mu m$')
ax.set_title('Drop diameter in um')
ax.set_xlabel('Drop diameter in um')
ax.set_ylabel('Count')

plt.suptitle('Histogram Overview for '+folder[-8:], fontsize=12)

plt.savefig(folder + 'histogram.png')

fig = plt.figure(figsize=(8, 8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
ax.set_theta_zero_location("S")
N = 100
width = (2*np.pi)/N
theta = np.linspace(-np.pi+np.pi/N, np.pi+np.pi/N, N, endpoint=False)
max_height = 8
# ori = [np.deg2rad(a) for a in np.sort(orientation) if ~np.isnan(np.deg2rad(a))]
# ori = orientation
radii = np.histogram(orientation, theta)
bars = ax.bar(radii[1][:-1], radii[0], width=width, bottom=20)

for r, bar in zip(orientation, bars):
    bar.set_facecolor(plt.cm.jet(r/10.))
    bar.set_alpha(0.5)

temp_vals = list()
mean_vals = np.zeros(len(folder_list))
std_vals = np.zeros(len(folder_list))
curr_val = 0

for time, v in zip(time_list, projected_vs):
    if time == [curr_val]:
        temp_vals.append(v)
    if time != [curr_val]:
        if len(temp_vals)>0:
            mean_vals[curr_val] = np.mean(temp_vals)
            std_vals[curr_val] = np.std(temp_vals)
            curr_val += 1
        else:
            mean_vals[curr_val] = np.nan
            std_vals[curr_val] = np.nan
            curr_val += 1
        while time != [curr_val]:
            mean_vals[curr_val] = np.nan
            std_vals[curr_val] = np.nan
            curr_val += 1
        temp_vals = list()
        temp_vals.append(v)

N=101

sum_run = np.nancumsum(np.insert(mean_vals, 0, 0))
sum_std = np.nancumsum(np.insert(std_vals, 0, 0))

rm_vs = np.zeros(len(mean_vals))
# rm_std = np.zeros(len(folder_list))

for n in np.arange(np.min([N-1, len(mean_vals)])):
    rm_vs[n] = np.nanmean(mean_vals[:n])
    # rm_std[n] = np.nanstd(mean_vals[:n])
rm_vs[N-1:] = (sum_run[N:]-sum_run[:-N])/N
rm_std = (sum_std[N:]-sum_std[:-N])/N

f_start = folder_list[0]
start_time = np.float16(f_start[21:23])*60+np.float16(f_start[24:26])+np.float16(f_start[27:30])/1000
time_fl = [np.float16(f[21:23])*60+np.float16(f[24:26])+np.float16(f[27:30])/1000-start_time for f in folder_list if '.png' in f]

plt.figure()
plt.plot(time_fl, rm_vs, color='b')
plt.plot(time_fl[np.int((N-1)/2):np.int(-(N-1)/2)], rm_vs[np.int((N-1)/2):np.int(-(N-1)/2)]+rm_std, color='b', linestyle='--')
plt.plot(time_fl[np.int((N-1)/2):np.int(-(N-1)/2)], rm_vs[np.int((N-1)/2):np.int(-(N-1)/2)]-rm_std, color='b', linestyle='--')
plt.title('Running Mean of Fall Speed over Time')
plt.xlabel('Time in seconds')
plt.ylabel('Running mean of fall velocity in cm/s')

center_x = [c[0] for c in centerpt]

f, ax = plt.subplots(1)
ax.scatter(center_x, [o/np.pi for o in orientation])
ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
ax.yaxis.set_major_locator(tck.MultipleLocator(base=0.5))


# plt.show()

os = sorted(zip(orientation, centerpt), key=lambda tup: tup[1][0])
orientation_s = [o[0] for o in os]
centerpt_s = [o[1] for o in os]

bins_orientation = list()

bin_size = 40
max_x_bin = np.ceil(830/bin_size)
max_y_bin = np.ceil(2048/bin_size)
x_range = np.arange(max_x_bin+1)[1:]
y_range = np.arange(max_y_bin+1)[1:]
xs = x_range*bin_size
ys = y_range*bin_size

for i in x_range-1:
    bins_orientation.append(list())
    for j in y_range-1:
        bins_orientation[int(i)].append(list())

for oc in zip(orientation_s, centerpt_s):
    set_flag=0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if (oc[1][0] < x) & (oc[1][1] < y):
                bins_orientation[i][j].append(oc[0])
                set_flag = 1
                break
        if set_flag:
            break

binned_o = np.zeros([len(xs), len(ys)])
binned_n = np.zeros([len(xs), len(ys)])

for i in x_range-1:
    for j in y_range-1:
        binned_o[int(i)][int(j)] = np.mean(bins_orientation[int(i)][int(j)])
        binned_n[int(i)][int(j)] = len(bins_orientation[int(i)][int(j)])

plt.figure()
plt.pcolormesh(xs*pixel_size, ys*pixel_size, np.flip(np.transpose(binned_n),0)/np.max(binned_n))
plt.colorbar()
plt.xlabel('x in $\mu m$')
plt.ylabel('y in $\mu m$')
plt.title('Relative occurence of fall streak center points')

plt.show()


def get_angles(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurd = cv2.GaussianBlur(img, (15, 15), 0)
    grad = cv2.Laplacian(blurd, -1)
    # grad = cv2.Sobel(blurd, -1, 1, 0, ksize=15)

    # edges = cv2.Canny(grad, 3, 9, apertureSize=3)
    lines = cv2.HoughLines(grad, 1, np.pi / 180, 200)

    return lines[0][1]