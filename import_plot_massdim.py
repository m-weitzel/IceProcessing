""" Plotting script for m(D) relationship and parameterization. Loads mass_dim_data.dat from all listed folders,
puts all mass and dimension values in a list and scatter plots them. Then defines a parameterization that is also plotted,
along with parameterizations from literature."""

from matplotlib import ticker
import pickle
from scipy import stats
from itertools import cycle
from utilities.find_ccr_folder import find_ccr
from utilities.make_pretty_figure import *
from utilities.plot_size_distribution import plot_size_dist
from utilities.fit_powerlaw import fit_powerlaw


def main():

    basedir = find_ccr()
    folder_list = (

        # Clean Measurements

        os.path.join(basedir, 'Y2017/2203/M1/'),  # Columnar Irregular
        os.path.join(basedir, 'Y2017/2203/M2/'),  # Columnar, Bullet Rosettes
        os.path.join(basedir, 'Y2017/3103/M1/'),  # Columnar
        os.path.join(basedir, 'Y2017/1107/M1/'),  # Dense Irregular
        os.path.join(basedir, 'Y2017/0808/M1/'),  # Columnar

        # Medium measurements

        os.path.join(basedir, 'Y2017/0604/M1/'),    # Aggregates
        os.path.join(basedir, 'Y2017/0208/M1/'),    # Dendritic, Aggregates
        os.path.join(basedir, 'Y2017/0208/M2/'),    # Irregular, Aggregates
        os.path.join(basedir, 'Y2017/0908/M1/'),    # Dendritic, Irregular, Dense
        os.path.join(basedir, '0103/'),      # Dendritic (aggregates)
        os.path.join(basedir, '26Sep'),
        #
        #
        # # Unclean measurements
        #
        # os.path.join(basedir, '2804/M1/'),      # Irregular, Columnar
        # os.path.join(basedir, 'Y2017/1503/M1/'),    # Irregular Dendritic, Aggregates
        # os.path.join(basedir, 'Y2017/1907/M1/'),    # Dendritic
        # os.path.join(basedir, 'Y2017/1907/M2/'),    # Dendritic
        # os.path.join(basedir, 'Y2017/1907/M3/'),    # Dendritic
    )

    compare = False
    if compare:
        compare_list_folder = os.path.join(basedir, '26Sep')

    minsize = 0
    maxsize = 150
    plot_scatter = True
    plot_massdim = True
    logscale = False
    plot_binned = True
    fontsize_base = 20

    # folder_list = (os.path.join(basedir, '26Sep')),
    #                '/uni-mainz.de/homes/maweitze/Dropbox/Dissertation/Ergebnisse/EisMainz/2203/M2/')

    # (x_shift, y_shift, dim_list, mass_list)

    folders_dim_list = list()
    folders_mass_list = list()
    folders_aspr_list = list()

    full_dim_list = list()
    full_mass_list = list()

    index_list = []

    for folder, i in zip(folder_list, np.arange(1, len(folder_list)+1)):
        tmp = pickle.load(open(os.path.join(folder, 'mass_dim_data.dat'), 'rb'))
        crystal_list = tmp['crystal']

        this_dim_list = dim_list(crystal_list, 'majsiz')
        this_mass_list = dim_list(crystal_list, 'mass')
        this_aspr_list = dim_list(crystal_list, 'aspr')

        filtered_lists = [(a, b, c) for (a, b, c) in zip(this_dim_list, this_mass_list, this_aspr_list) if ((float(a) > minsize) & (float(a) < maxsize))]
        this_dim_list = [a[0] for a in filtered_lists]
        this_mass_list = [a[1] for a in filtered_lists]
        this_aspr_list = [a[2] for a in filtered_lists]

        folders_dim_list.append(this_dim_list)
        folders_mass_list.append(this_mass_list)
        folders_aspr_list.append(this_aspr_list)
        print('Added '+str(len(this_dim_list))+' from '+folder+'.')
        full_dim_list += this_dim_list
        full_mass_list += this_mass_list
        index_list += [i]*len(this_dim_list)

    if compare:
        tmp = pickle.load(open(os.path.join(compare_list_folder, 'mass_dim_data.dat'), 'rb'))
        compare_list = tmp['crystal']

        comp_dim_list = dim_list(compare_list, 'majsiz')
        comp_mass_list = dim_list(compare_list, 'mass')

    full_dim_list, full_mass_list, index_list = zip(*sorted(zip(full_dim_list, full_mass_list, index_list)))

    # bins = np.linspace(7.5, 157.5, 31)
    # bins = np.linspace(5, 155, 16)
    bins = np.linspace(15, 105, 11)

    avg_masses, bin_edges, binnumber = stats.binned_statistic(full_dim_list, full_mass_list,
                                                              statistic='mean', bins=bins)
    num_in_bin, _, _ = stats.binned_statistic(full_dim_list, full_mass_list,
                                              statistic='count', bins=bins)
    # mass_std, _, _= stats.binned_statistic(full_dim_list, full_mass_list
    #                                                          , statistic='std', bins=bins)

    # symbols = ["o", "8", "s", "p", "h", "H", "D"]
    symbols = ["o"]
    # symbols = ["o", "s", "+", "x", "D"]
    symbolcycler = cycle(symbols)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    colorcycler = cycle(colors)

    almost_black = '#262626'

    max_aspr = max([max(a) for a in folders_aspr_list])

    # fig, ax = plt.subplots(1)
    fig, ax = imshow_in_figure(figspan=(18, 10))

    if logscale:
        xlim = 10
        ydist_factor = 2
        ydist0 = 1/2
        plt.xlim(xlim, 1.5 * np.max(full_dim_list))
        plt.ylim(1e-9, 500*1e-9)
        ax.set_xscale('log')
        ax.set_yscale('log')

        # ax.grid(which='minor', alpha=0.3)
        # ax.grid(which='major', alpha=0.5)
        xloc = 1.2*xlim
        yloc = [ydist0, ydist0/ydist_factor, ydist0/(2*ydist_factor)]
        legloc = 4
        ymax = 300e-9

    else:
        xmin = 15
        # xmax = 1.1*np.max(full_dim_list)
        xmax = 145
        ymin = 0
        ymax = 1.1*np.max(full_mass_list)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        xloc = 1
        yloc_base = 0.8
        yloc_factor = 1.1
        yloc = [yloc_base, yloc_base-0.05*yloc_factor, yloc_base-0.1*yloc_factor, yloc_base-0.15*yloc_factor]
        legloc = 2
        # ax.grid()

    dim_bins = [(i+j)/2 for i, j in zip(bin_edges[:-1], bin_edges[1:])]

    powerlaw = lambda x, amp, index: amp * (x**index)

    dims_spaced = np.arange(maxsize)
    if plot_binned:
        amp_bins, index_bins = fit_powerlaw([d*1e-6 for d in dim_bins], avg_masses)
    amp_full, index_full = fit_powerlaw([d*1e-6 for d in full_dim_list], full_mass_list)
    mass_bulk = np.pi / 6*(dims_spaced*1e-6) ** 3 * 916.7            # D in m, m in kg
    brown_franc = 0.0185 * (dims_spaced*1e-6) ** 1.9                 # D in m, m in kg, equal to LH1974
    # brown_franc_cgs = 0.00294*(dims_spaced*1e-4)**1.9/1000           # cgs, from Heymsfield 2010
    mitchell_90 = 0.022*(dims_spaced*1e-3)**2/1e6                    # D in mm, m in mg
    mitchell_2010 = 0.08274*(dims_spaced*1e-4) ** 2.814/1000         # cgs
    heymsfield2010 = 0.007*(dims_spaced*1e-4)**2.2/1000              # cgs

    # bakerlawson = 0.115*(dims_spaced*1e-6)**1.218/1000               # A in mm², m in mg
    # D in mm, m in mg
    # heymsfield = 0.176*dims_spaced**2.2*1000

    rmse_of_fit = rmse([amp_full*d**index_full for d in full_dim_list], full_mass_list)

    loc_count = 0

    # if plot_massdim:
    #     plt.text(xloc, ymax*yloc[loc_count], 'Brown&Francis: $m=0.0185\cdot D^{1.9}$\n Mitchell: $m=0.022\cdot D^{2.0}$', bbox=dict(facecolor='red', alpha=0.2), fontsize=fontsize_base*0.7)
    # else:
    #     plt.text(xloc, ymax * yloc[loc_count], 'Baker&Lawson: $m=0.115\cdot D^{1.218}$',
    #              bbox=dict(facecolor='red', alpha=0.2), fontsize=fontsize_base * 0.7)
    #
    # loc_count += 2
    #
    # if plot_binned:
    #     plt.text(xloc, ymax*yloc[loc_count], 'Power Law Fit with Bins: $m = {0:.4f}\cdot D^{{{1:.3f}}}$\nn = {2}'.format(amp_bins/1000, index_bins, n),
    #          bbox=dict(facecolor='blue', alpha=0.2), fontsize=fontsize_base*0.7)
    #     loc_count += 1
    #
    # plt.text(xloc, ymax*yloc[loc_count], 'Power Law Fit full data: $m = {0:.4f}\cdot D^{{{1:.3f}}}$\nn = {2}, RMSE = {3:4.2f} ng'.format(amp_full/1000, index_full, n, rmse_of_fit*1e-12*1e9),
    #          bbox=dict(facecolor='blue', alpha=0.2), fontsize=fontsize_base*0.7)
    # loc_count += 1
    #
    # if not plot_binned:
    #     plt.text(xloc, ymax*yloc[loc_count],
    #              'Maximum aspect ratio: $AR_{{max}} = {0:.2f}$\n Blue: AR < 1.5, Red: 1.5 < AR < 2.5, Green: 2.5 < AR'.format(max_aspr),
    #              # 'Maximum aspect ratio: $AR_{{max}} = {0:.2f}$'.format(max_aspr),
    #              bbox=dict(facecolor='green', alpha=0.2), fontsize=fontsize_base*0.7)
    #     loc_count += 1
    # loc_count += 2

    # n = len(full_mass_list)

    # if plot_massdim & plot_binned:
    #     plt.text(35, 100e-9, 'Power Law Fit full data: m$ = {2:.4f}\cdot $D$^{{{3:.3f}}}$'.format(1.9, 2.0, amp_full/1000, index_full)+
    #          '\nPower Law Fit binned data: m$ = {0:.4f}\cdot $D$^{{{1:.3f}}}$'.format(amp_bins/1000, index_bins), bbox=dict(facecolor='blue', alpha=0.1), fontsize=fontsize_base*0.9, ha='right')
    # else:
    #     # plt.text(xloc + 45.5, ymax * 0.95 * yloc[loc_count],
    #     plt.text(
    #              xloc, ymax * 0.98 * yloc[loc_count],
    #             # 'Brown&Francis: m$=0.0185\cdot $D$^{1.9}$\n' + ' Mitchell: m$=0.022\cdot $D$^{2.0}$\n' +
    #              'Power Law Fit full data: m$ = {0:.4f}\cdot $D$^{{{1:.3f}}}$\nn = ${2}$, RMSE = ${3:4.2f}$ ng'.format(amp_full / 1000, index_full, n, rmse_of_fit * 1e-12 * 1e9),
    #             bbox=dict(facecolor='blue', alpha=0.1), fontsize=fontsize_base * 1, ha='left'
    #              )

    if plot_binned:
        edge_indices = np.insert(np.cumsum(num_in_bin), 0, 0)
        binned_masses = [full_mass_list[int(c):int(d)] for c, d in zip(edge_indices[:-1], edge_indices[1:])]       # list of all dim values in bins, length = number of bins

        mass_stds = [np.std(vm) for vm in binned_masses]

        error_oor = mass_stds > avg_masses
        sparses = num_in_bin < 5
        exclude = np.invert(sparses+error_oor)
        dim_bins = np.asarray(dim_bins)[exclude]
        avg_masses = np.asarray(avg_masses)[exclude]
        mass_stds = np.asarray(mass_stds)[exclude]

        ax.plot(dims_spaced, powerlaw(dims_spaced*1e-6, amp_bins, index_bins), label=r'm={0:.5f}D^{1:.2f}'.format(amp_bins, index_bins), linewidth=3, zorder=1)
        print(amp_bins)
        ax.errorbar(dim_bins, avg_masses, yerr=mass_stds, linestyle='none', fmt='none', color='k', capsize=5)

        ax.scatter(90, 20e-12, s=3*10**0.8, c='g')
        ax.scatter(110, 20e-12, s=3*100**0.8, c='g')
        ax.scatter(130, 20e-12, s=3*1000**0.8, c='g')

        ax.text(90, 21e-12, '10')
        ax.text(110, 21e-12, '100')
        ax.text(130, 21e-12, '1000')
    else:
        ax.plot(dims_spaced, powerlaw(dims_spaced, amp_full, index_full), label='Power Law Fit', linewidth=3, zorder=1, c='orange')

    if plot_massdim:
        ax.plot(dims_spaced, mass_bulk, label='Solid Ice Spheres', linestyle='-.', linewidth=3, zorder=1)
        ax.plot(dims_spaced, brown_franc, label='Brown&Francis 95', linestyle='--', zorder=1)
        # ax.plot(dims_spaced, brown_franc_cgs, label='Brown&Francis 1995', linestyle='--', zorder=1)
        ax.plot(dims_spaced, mitchell_90, label='Mitchell 1990', linestyle='--', zorder=1)
        ax.plot(dims_spaced, mitchell_2010, label='Mitchell 2010', linestyle='--', zorder=1)
        ax.plot(dims_spaced, heymsfield2010, label='Heymsfield 2010', linestyle='--', zorder=1)
    # else:
    #     ax.plot(dims_spaced, bakerlawson, label='Baker&Lawson 06', linestyle='--', zorder=1)
    # ax.plot(dims_spaced, heymsfield, label='Heymsfield 2011', linestyle='--')
    # plt.plot(dims_spaced, dims_spaced)

    if plot_scatter:
        for this_dim_list, this_mass_list, this_aspr_list, this_folder in zip(folders_dim_list, folders_mass_list, folders_aspr_list, folder_list):
            # ax.scatter(this_dim_list, this_mass_list, label=this_folder[-8:], c=next(colorcycler), marker=next(symbolcycler), alpha=1, edgecolor=almost_black, linewidth=0.15)
            # col_list = [[1 / (max_aspr - 1) * (c - 1), 1 / (max_aspr - 1) * (max_aspr - c), 0] for c in this_aspr_list]
            col_list = [[1, 0, 0] if 1.5 < c < 2.5 else [0, 0, 1] if c < 1.5 else [0, 1, 0] for c in this_aspr_list]
            # ax.scatter(this_dim_list, this_mass_list, label=this_folder[-8:], c=col_list, marker=next(symbolcycler), alpha=1,
            #            edgecolors=almost_black, linewidth=1)
            # ax.scatter(this_dim_list, this_mass_list, c=col_list, marker=next(symbolcycler), alpha=1,
            #            edgecolors=almost_black, linewidth=1)
            if plot_binned:
                ax.scatter(dim_bins, avg_masses, alpha=1, edgecolors=almost_black, linewidth=1, s=3*num_in_bin**0.8, zorder=2)
            else:
                ax.scatter(full_dim_list, mass_list, alpha=1, edgecolors=almost_black, linewidth=1, zorder=0)#, c=col_list)
            # ax.errorbar([(i+j)/2 for i, j in zip(bin_edges[:-1], bin_edges[1:])], avg_masses, yerr=mass_std, fmt='o')

        if compare:
            ax.scatter(comp_dim_list, comp_mass_list, alpha=1, edgecolor=almost_black, linewidth=1, zorder=0, c='y')

    plt.xlabel('Maximum dimension in $\mathrm{\mu m}$', fontsize=fontsize_base)
    plt.ylabel('Mass in ng', fontsize=fontsize_base)
    ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y*1e12))
    ax.yaxis.set_major_formatter(ticks_y)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_base)
    # ax.set_xticks([25,50,75,100])
    # ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    # ttl = plt.title('Mass Dimension Relation', fontsize=fontsize_base*1.5)
    # ttl.set_position([0.5, 1.025])

    legend = ax.legend(frameon=True, scatterpoints=1, loc=legloc, fontsize=fontsize_base)
    # for k in np.arange(4, len(legend.legendHandles)):
    #     legend.legendHandles[k].set_color('blue')
    light_grey = np.array([float(248)/float(255)]*3)
    rect = legend.get_frame()
    rect.set_facecolor(light_grey)
    rect.set_linewidth(0.0)

    savefig_ipa(fig, 'MassDimScatter')

    # fig_hist, _ = create_hist(full_dim_list)
    fig_hist, _ = plot_size_dist(full_dim_list, 25, xlabel='Size in micrometers')
    savefig_ipa(fig_hist, 'MassDimHist')

    plt.show()


def dim_list(c_list, type):
    if type == 'areaeq':
        return [2*np.sqrt(float(a['Area'])/np.pi) for a in c_list]
    elif type == 'area':
        return [a['Area'] for a in c_list]
    elif type == 'majminmean':
        return [(float(a['Long Axis'])+float(a['Short Axis']))/2 for a in c_list]                         # Mean of Maximum and Minimum dimension
    elif type == 'aspr':
        return [(float(a['Long Axis'])/float(a['Short Axis'])) for a in c_list]                            # Aspect Ratio
    elif type == 'cap':
        return [0.134 * (0.58 * a['Short Axis'] / 2 * (1 + 0.95 * (a['Long Axis'] / a['Short Axis']) ** 0.75)) for a in c_list]      # Capacitance
    elif type == 'mass':
        return [np.pi/6*(a['Drop Diameter']*1e-6)**3*1000 for a in c_list]
    elif type == 'minsiz':
        return [float(a['Short Axis']) for a in c_list]
    elif type == 'dropdiam':
        return [float(a['Drop Diameter']) for a in c_list]
    else:
        return [a['Long Axis'] for a in c_list]


if __name__ == "__main__":
    main()

