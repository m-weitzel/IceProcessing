import matplotlib.pyplot as plt


def plot_size_dist(sizes, n_bins):
    fig_h = plt.figure()
    ax = fig_h.add_subplot(111)

    ax.hist(sizes, n_bins, histtype='step', fill=False, label='N = {}'.format(len(sizes)), linewidth=3)
    ax.set_xlabel('Particle Diameter', fontsize=20)
    ax.set_ylabel('Count', fontsize=20)
    ax.grid()

    return fig_h, ax