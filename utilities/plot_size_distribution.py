from utilities.make_pretty_figure import *
from matplotlib import ticker


def plot_size_dist(sizes, n_bins, normed=True, log=False, lims=False, xlabel='', **kwargs):
    fig_h, ax = imshow_in_figure(grid=False, figspan=(18, 10))
    # ax = fig_h.add_subplot(111)

    if lims:
        h = ax.hist(sizes, n_bins, histtype='bar', label='N = {}'.format(len(sizes)), linewidth=3, density=normed, log=log, edgecolor='k', range=lims[0])
        # ax.hist(sizes, n_bins, histtype='stepfilled', density=normed, log=log, range=lims[0], fc=(0, 0, 1, 0.001))
        ax.set_xlim(lims[0])
        try:
            ax.set_ylim(lims[1])
        except IndexError:
            pass
    else:
        h = ax.hist(sizes, n_bins, histtype='bar', label='N = {}'.format(len(sizes)), linewidth=3, density=normed, log=log, edgecolor='k')
        # ax.hist(sizes, n_bins, histtype='stepfilled', density=normed, log=log, fc=(0, 0, 1, 0.001))

    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel('PDF', fontsize=24)
    oom = 10**np.floor(np.log10(np.max(h[0])))
    if (np.max(h[0])/oom) < 3:
        ys = np.arange(0, np.max(h[0]), oom/2)
    else:
        ys = np.arange(0, np.max(h[0]), oom/2)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    ax.set_yticks(ys)
    # ax.grid()

    return fig_h, ax