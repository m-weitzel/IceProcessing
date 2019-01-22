import matplotlib.pyplot as plt
import numpy as np
import os
from time import strftime


def imshow_in_figure(img=None, ax=None, xlabel=None, ylabel=None, title=None, hide_axes=False, figspan=(10, 10), dpi=100, grid=True):

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
        ax.imshow(img)

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=20)

    ax.set_title(title, fontsize=28)
    if grid:
        ax.grid(b=grid, which='major', linestyle='-')

    ax.tick_params(axis='both', which='major', labelsize=20)

    if hide_axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    return fig, ax


def create_hist(vals, ax=None, minval=0, maxval=None, bins=None, step=1, **kwargs):

    if maxval is None:
        maxval = np.max(vals)

    if bins is None:
        bins = np.arange(minval, maxval, step)
    if ax is None:
        fig, ax = imshow_in_figure(grid=True, figspan=(10, 5), ax=ax, **kwargs)
    else:
        fig = None

    _, b, _ = ax.hist(vals, bins, edgecolor='black', linewidth=1.2, density=False, weights=np.ones(len(vals))/len(vals), label='N = {}'.format(len(vals)))
    ax.set_xlim(0, b[-1])
    ax.set_ylabel('PDF', fontsize=20)
    ax.grid(which='major', linestyle='-')
    # ax.legend(fontsize=20, loc='upper right')

    ax.tick_params(axis='both', which='major', labelsize=20)

    return fig, ax


def savefig_ipa(fig, fn):
    date = strftime("%d%b%y")
    time = strftime("%-H.%M.%S")

    path = '/ipa/holo/mweitzel/Ergebnisse/figdump'

    datepath = os.path.join(path, date)

    try:
        os.mkdir(datepath)
    except FileExistsError:
        pass
    fig.savefig(os.path.join(datepath, fn+time+'.png'), bbox_inches='tight',)

