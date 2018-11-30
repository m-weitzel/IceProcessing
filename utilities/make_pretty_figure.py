import matplotlib.pyplot as plt
import numpy as np


def imshow_in_figure(img=None, ax=None, xlabel=None, ylabel=None, title=None, hide_axes=False, figspan=(10, 10), dpi=100, grid=False):

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

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    ax.set_title(title, fontsize=28)
    ax.grid(b=grid, which='major', linestyle='-')

    ax.tick_params(axis='both', which='major', labelsize=20)

    if hide_axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    return fig, ax


def create_hist(vals, minval, maxval, step=1, **kwargs):

    bins = np.arange(minval, maxval, step)

    fig, ax = imshow_in_figure(grid=True, **kwargs)
    ax.hist(vals, bins, edgecolor='black', linewidth=1.2)

