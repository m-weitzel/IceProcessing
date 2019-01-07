import matplotlib.pyplot as plt


def plot_size_dist(sizes, n_bins, normed=True, log=False, lims=False, xlabel=''):
    fig_h = plt.figure()
    ax = fig_h.add_subplot(111)

    if lims:
        ax.hist(sizes, n_bins, histtype='step', fill=True, label='N = {}'.format(len(sizes)), linewidth=3, density=normed, log=log, edgecolor='k', range=lims[0])
        ax.set_xlim(lims[0])
        try:
            ax.set_ylim(lims[1])
        except IndexError:
            pass
    else:
        ax.hist(sizes, n_bins, histtype='step', fill=True, label='N = {}'.format(len(sizes)), linewidth=3, density=normed, log=log, edgecolor='k')

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel('Relative occurence', fontsize=20)
    ax.set_ylabel('Relative occurence', fontsize=20)
    ax.grid()

    return fig_h, ax