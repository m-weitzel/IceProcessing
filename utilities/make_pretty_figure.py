import matplotlib.pyplot as plt
import numpy as np
import os
from time import strftime


def imshow_in_figure(img=None, ax=None, xlabel=None, ylabel=None, title=None, hide_axes=False, figspan=(20.48, 20.48), dpi=100, grid=True):

    fig = plt.figure(figsize=figspan, dpi=dpi)
    if ax is None:
        ax = fig.add_subplot(111)
    else:
        fig.axes.append(ax)
        ax.figure = fig
        fig.add_axes(ax)

    fig.canvas.set_window_title(title)

    if img is None:
        pass
    else:
        ax.imshow(img, cmap='bone')

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=20)

    ax.set_title(title, fontsize=28)
    if grid:
        ax.grid(b=grid, which='major', linestyle='-')

    ax.tick_params(axis='both', which='major', labelsize=24)

    if hide_axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    return fig, ax


def create_hist(vals, ax=None, minval=0, maxval=None, bins=None, step=1, label='', **kwargs):
    from scipy.stats import skew, kurtosis
    if maxval is None:
        maxval = np.max(vals)

    if bins is None:
        bins = np.arange(minval, maxval, step)
    if ax is None:
        # fig, ax = imshow_in_figure(grid=True, figspan=(10, 6), ax=ax, **kwargs)
        fig, ax = imshow_in_figure(grid=True, figspan=(10, 6), **kwargs)
    else:
        fig = None

    if label == '':
        label = 'N = {}'.format(len(vals))

    _, b, _ = ax.hist(vals, bins, edgecolor='black', linewidth=1.2, density=True, weights=np.ones(len(vals))/len(vals), label=label, zorder=3)
    s = skew(vals)
    k = kurtosis(vals)
    ax.set_xlim(minval, 2*b[-1]-b[-2])
    ax.set_ylabel('PDF', fontsize=20)

    # ax.set_xlabel(r'Drop diameter in $\mu m$', fontsize=20)
    # ax.set_xlabel(r'Fall orientation in $\degree$', fontsize=20)
    # ax.get_legend().set_visible(False)

    ax.grid(which='major', linestyle='-', zorder=0)
    ax.legend(fontsize=20, loc='upper right')

    ax.tick_params(axis='both', which='major', labelsize=20)
    print('Skewness of distribution: {}'.format(s))
    print('Kurtosis of distribution: {}'.format(k))
    print('N = {}'.format(len(vals)))

    return fig, ax


def savefig_ipa(fig, fn, dpi=100):
    date = strftime("%d%b%y")
    time = strftime("%-H.%M.%S")

    path = '/ipa/holo/mweitzel/Ergebnisse/figdump'

    datepath = os.path.join(path, date)

    try:
        os.mkdir(datepath)
    except FileExistsError:
        pass
    # fig.savefig(os.path.join(datepath, fn+time+'.png'), bbox_inches='tight',)
    fig.savefig(os.path.join(datepath, fn+time+'.png'), dpi=dpi, bbox_inches='tight')


def density_plot(values, position, pxl_size, imsize, relative=True, nan_val=0):

    class ValueClass:
        def __init__(self, val, position):
            self.val = val
            self.xpos = position[0]
            self.ypos = position[1]

    summary_values = list()
    for v, p, in zip(values, position):
        summary_values.append(ValueClass(v, p))

    xs = np.arange(-imsize[0]*pxl_size/1000/2, imsize[0]*pxl_size/1000/2, 0.35)
    ys = np.arange(-imsize[1]*pxl_size/1000/2, imsize[1]*pxl_size/1000/2, 0.35)

    sorted_stuff = sorted(summary_values, key=lambda v: v.xpos)

    mean_val_in_bins = list()

    for i, _ in enumerate(xs):
        mean_val_in_bins.append(list())
        for j, _ in enumerate(ys):
            mean_val_in_bins[i].append(list())

    for val_class in sorted_stuff:
        set_flag = 0
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if (val_class.xpos < x) & (val_class.ypos < y):
                    mean_val_in_bins[i][j].append(val_class.val)
                    set_flag = 1
                    break
            if set_flag:
                break

    binned_vals = np.zeros([len(xs), len(ys)])

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            elements_in_this_bin = mean_val_in_bins[int(i)][int(j)]
            if len(elements_in_this_bin) > 3:
                this_mean_val = np.median(elements_in_this_bin)
            else:
                this_mean_val = nan_val
            # if np.isnan(this_val) or (len(bins_vals[int(i)][int(j)]) < 4):
            #     binned_vals[int(i)][int(j)] = nan_val
            # else:
            #     binned_vals[int(i)][int(j)] = this_val
            binned_vals[i][j] = this_mean_val
            mean_val_in_bins[i][j] = len(elements_in_this_bin)

    if relative:
        sum_all_n = np.sum(mean_val_in_bins)
        binned_n = [b/sum_all_n for b in [c for c in mean_val_in_bins]]

    # f, ax = plt.subplots(figsize=(8, 14))
    f, ax = imshow_in_figure(figspan=(8, 14))
    ax.set_aspect('equal')
    ax.set_xlim([xs[0]*pxl_size/1000, xs[-1]*pxl_size/1000])
    ax.set_ylim([ys[0]*pxl_size/1000, ys[-1]*pxl_size/1000])
    f.canvas.draw()
    cmap = plt.cm.YlOrRd
    # cmap = plt.cm.Spectral_r
    cmap.set_under(color='white')
    im = ax.pcolor(xs * pxl_size/1000, ys * pxl_size/1000, np.flip(np.transpose(binned_n), 0), cmap=cmap, vmin=0.0001)
    cbar = f.colorbar(im)
    cbar.ax.tick_params(labelsize=20)
    # cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel('x in $mm$', fontsize=20)
    ax.set_ylabel('y in $mm$', fontsize=20)
    ax.set_title('PDF of number in x/y bin', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # f, ax = plt.subplots(figsize=(8, 14))
    f, ax = imshow_in_figure(figspan=(8, 14))
    ax.set_aspect('equal')
    ax.set_aspect('equal')
    ax.set_title('Mean of value in x/y bin', fontsize=20)
    # ax.set_xlim(xs[0], xs[-1])
    # ax.set_ylim(ys[0], ys[-1])
    f.canvas.draw()
    # X, Y = np.meshgrid(xs, ys)
    cmap = plt.cm.RdBu
    cmap.set_under(color='white')
    im_pc = ax.pcolor(xs * pxl_size/1000, ys * pxl_size/1000, np.flip(np.transpose(binned_vals), 0), cmap=cmap, vmin=-10, vmax=10)

    cbar = f.colorbar(im_pc)#, ticks=np.linspace(0, 50, 6))
    cbar.set_label('v in $mm/s$', fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('x in $mm$', fontsize=20)
    ax.set_ylabel('y in $mm$', fontsize=20)
    # ax.set_title('Quiver plot of mean orientation and fall speed', fontsize=20)

        # ax.axis([0, 1000, 0, 1000])
    # f, ax = plt.subplots(figsize=(8, 14))
    # ax.set_aspect('equal')
    # ax.set_xlim(xs[0]*pxl_size/1000, xs[-1]*pxl_size/1000)
    # ax.set_ylim(ys[0]*pxl_size/1000, ys[-1]*pxl_size/1000)
    # f.canvas.draw()
    # cmap = plt.cm.Spectral_r
    # cmap.set_under(color='white')
    # im = ax.pcolor(xs * pxl_size/1000, ys * pxl_size/1000, np.flip(np.transpose(np.rad2deg(binned_o)), 0), cmap=cmap, vmin=-30, vmax=30)
    # ax.set_xlabel('x in $mm$', fontsize=20)
    # ax.set_ylabel('y in $mm$', fontsize=20)
    # ax.set_title('Median fall streak orientation relative to verticality', fontsize=20)
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # cbar = f.colorbar(im, ticks=np.linspace(-45, 45, 7))
    # cbar.set_label('$\phi$ in $\degree$', fontsize=20)
    # ticklabels = list(np.linspace(-1, 1, 9))
    # ticklabels = [str(t)+'$/4\cdot\pi$' for t in ticklabels]
    # cbar.ax.set_yticklabels([str(t)+'$/4\cdot\pi$' for t in list(np.linspace(-1, 1, 9))])