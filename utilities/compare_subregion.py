import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import RectangleSelector
from utilities.IceSizing import MicroImg
from utilities.make_pretty_figure import savefig_ipa
import time


def main():
    # path = '/ipa/holo/mweitzel/Windkanal/Ice/CCR/Y2017/3103/M1/Ice-5.png'
    path = '/ipa/holo/mweitzel/Windkanal/Ice/CCR/0103/Ice-6.png'
    analyze_subregion(path)


def analyze_subregion(path):

    base_img = plt.imread(path)
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111)
    ax.imshow(base_img, cmap='bone')
    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()

    img_list = list()
    label_list = list()
    img_list.append(base_img)
    label_list.append('Base image')
    filters = (
               ('Otsu', 0),
               ('Bin', 0),
               ('Adaptive', 0),
               # ('Canny', 3),
               ('Canny', 3),
               ('kmeans', 2, 2)
               )
    dilations = (
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                )

    pixel_size = 0.33

    for f, d in zip(filters, dilations):
        folder, file = tuple(path.rsplit('/', 1))
        starttime = time.time()
        t_microimg = MicroImg('Ice', folder, file, pixel_size, f, maxsize=np.inf, dilation=d, fill_flag=True, min_dist_to_edge=1)
        print('-- {0:.2f} seconds for {1} --'.format(time.time()-starttime, f))
        img_list.append(t_microimg.thresh_img)
        label_list.append(f[0])

    f = compare_subregion(img_list, (x1, y1, x2, y2), label_list)

    savefig_ipa(f, 'compare_column')
    plt.show()


def compare_subregion(img_list, region, label_list):

    # fig = plt.figure(figsize=(10, 10), dpi=100)
    fig = plt.figure(figsize=(8, 10), dpi=100)

    n = np.int(np.sqrt(len(img_list)))+1

    axes = fig.subplots(3, 2)

    for i, (img, a, l) in enumerate(zip(img_list, axes.flatten(), label_list)):
        subimg = img[region[1]:region[3], region[0]:region[2]]
        if l == 'Base image':
            a.imshow(subimg, cmap='bone')
        else:
            a.imshow(subimg)
        a.get_xaxis().set_visible(False)
        a.yaxis.set_major_locator(plt.NullLocator())
        # a.get_yaxis().set_visible(False)
        # a.set_title(l, fontsize=20)
        a.set_ylabel(l, fontsize=18)
        if i & 0x1:
            a.yaxis.set_label_position("right")

    fig.subplots_adjust(hspace=.05, wspace=0)

    return fig


def line_select_callback(eclick, erelease):
    print(eclick.xdata)
    'eclick and erelease are the press and release events'
    global x1, y1, x2, y2
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    print(" The button you used was: %s %s" % (eclick.button,
          erelease.button))


def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


if __name__ == "__main__":
    main()