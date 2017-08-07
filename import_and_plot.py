from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy import optimize
from itertools import cycle
# from pylab import *

folder_list = (
    # Clean Measurements

    '/uni-mainz.de/homes/maweitze/CCR/2203/M1/',  # Columnar
    '/uni-mainz.de/homes/maweitze/CCR/2203/M2/',  # Columnar, Bullet Rosettes
    '/uni-mainz.de/homes/maweitze/CCR/3103/M1/',  # Columnar
    '/uni-mainz.de/homes/maweitze/CCR/1107/M1/',  # Dense Irregular

    # Medium measurements

    '/uni-mainz.de/homes/maweitze/CCR/0604/M1/',    # Aggregates
    '/uni-mainz.de/homes/maweitze/CCR/0208/M1/',    # Dendritic, Aggregates
    '/uni-mainz.de/homes/maweitze/CCR/0208/M2/',    # Irregular, Aggregates

    # Unclean measurements

    # '/uni-mainz.de/homes/maweitze/CCR/1503/M1/',    # Irregular Dendritic, Aggregates
    # '/uni-mainz.de/homes/maweitze/CCR/1907/M1/',    # Dendritic
    # '/uni-mainz.de/homes/maweitze/CCR/1907/M2/',    # Dendritic
    # '/uni-mainz.de/homes/maweitze/CCR/1907/M3/',    # Dendritic
)

minsize=5
maxsize=200
logscale = False

# folder_list = ('/uni-mainz.de/homes/maweitze/Dropbox/Dissertation/Ergebnisse/EisMainz/1907/M3/',
#                '/uni-mainz.de/homes/maweitze/Dropbox/Dissertation/Ergebnisse/EisMainz/2203/M2/')

# (x_shift, y_shift, dim_list, mass_list)

folders_dim_list = list()
folders_mass_list = list()
folders_aspr_list = list()

full_dim_list = list()
full_mass_list = list()

index_list = []

for folder, i in zip(folder_list, np.arange(1,len(folder_list)+1)):
    tmp = pickle.load(open(folder+'mass_dim_data.dat', 'rb'))
    crystal_list = tmp['crystal']

    this_dim_list = [
                     # float(a['Long Axis'])                                                                                  # Maximum dimension
                     # (float(a['Long Axis'])+float(a['Short Axis']))/2                                                      # Mean of Maximum and Minimum dimension
                     # (float(a['Long Axis'])/float(a['Short Axis']))                                                        # Aspect Ratio
                     # a['Area']                                                                                             # Area
                     2*np.sqrt(float(a['Area'])/np.pi)                                                                     # Area-equivalent diameter
                     for a in crystal_list if (float(a['Long Axis']) > minsize) & (float(a['Long Axis']) < maxsize)]

    this_mass_list = [
                      np.pi/6*a['Drop Diameter']**3                                                                       # Mass
                      # float(a['Short Axis'])                                                                              # Short Axis
                      # float(a['Drop Diameter'])                                                                           # Drop Diameter
                      for a in crystal_list if (float(a['Long Axis']) > minsize) & (float(a['Long Axis']) < maxsize)]

    this_aspr_list = [float(a['Long Axis']) / float(a['Short Axis'])
                      for a in crystal_list if (float(a['Long Axis']) > minsize) & (float(a['Long Axis']) < maxsize)]

    folders_dim_list.append(this_dim_list)
    folders_mass_list.append(this_mass_list)
    folders_aspr_list.append(this_aspr_list)

    full_dim_list += this_dim_list
    full_mass_list += this_mass_list
    index_list += [i]*len(this_dim_list)

full_dim_list, full_mass_list, index_list = zip(*sorted(zip(full_dim_list, full_mass_list, index_list)))

symbols = ["o", "8", "s", "p", "*", "h", "H", "+", "x", "X", "D"]
# symbols = ["o", "s", "+", "x", "D"]
symbolcycler = cycle(symbols)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
colorcycler = cycle(colors)

almost_black = '#262626'

max_aspr = max([max(a) for a in folders_aspr_list])

fig, ax = plt.subplots(1)

for this_dim_list, this_mass_list, this_aspr_list, this_folder in zip(folders_dim_list, folders_mass_list, folders_aspr_list, folder_list):
    # ax.scatter(this_dim_list, this_mass_list, label=this_folder[-8:], c=next(colorcycler), marker=next(symbolcycler), alpha=1, edgecolor=almost_black, linewidth=0.15)
    col_list = [[1 / (max_aspr - 1) * (c - 1), 1 / (max_aspr - 1) * (max_aspr - c), 0] for c in this_aspr_list]
    ax.scatter(this_dim_list, this_mass_list, label=this_folder[-8:], c=col_list, marker=next(symbolcycler), alpha=1,
               edgecolors=almost_black, linewidth=1)
if logscale:
    xlim = 10
    ydist_factor = 2
    ydist0 = 1/2
    plt.xlim(xlim, 1.5 * np.max(full_dim_list))
    plt.ylim(300, 1.5 * np.max(full_mass_list))
    ax.set_xscale('log')
    ax.set_yscale('log')
    xloc = 1.2*xlim
    yloc = [ydist0, ydist0/ydist_factor, ydist0/(2*ydist_factor)]
    legloc = 4

else:

    plt.xlim(0, 1.1*np.max(full_dim_list))
    plt.ylim(0, 1.1*np.max(full_mass_list))
    xloc = 10
    yloc = [0.9, 0.85, 0.8]
    legloc = 1


powerlaw = lambda x, amp, index: amp * (x**index)

logx = np.log10(full_dim_list)
logy = np.log10(full_mass_list)

fitfunc = lambda p, x: p[0]+p[1]*x
errfunc = lambda p, x, y: (y-fitfunc(p,x))

pinit = [1.0, -1.0]
out = optimize.leastsq(errfunc, pinit, args=(logx, logy), full_output=1)

pfinal = out[0]
covar = out[1]

index = pfinal[1]
amp = 10.0**pfinal[0]

n = len(full_dim_list)
nu = len(np.unique(full_mass_list))

plt.text(xloc, np.max(full_mass_list)*yloc[0], 'Brown&Francis: $m=0.0185\cdot D^{1.9}$, Mitchell: $m=0.022\cdot D^{2.0}$', bbox=dict(facecolor='red', alpha=0.2))
plt.text(xloc, np.max(full_mass_list)*yloc[1], 'Power Law Fit: $m = {0:.4f}\cdot D^{{{1:.3f}}}$\nn = {2}, unique: {3}'.format(amp/1000, index, n, nu),
         bbox=dict(facecolor='blue', alpha=0.2))

plt.text(xloc, np.max(full_mass_list)*yloc[2],
         'Maximum aspect ratio: $AR_{{max}} = {0:.2f}$\n Red for high AR, Green for low AR'.format(max_aspr),
         bbox=dict(facecolor='green', alpha=0.2))

dims_spaced = np.arange(maxsize)
mass_bulk = np.pi / 6 * (dims_spaced) ** 3 * 0.9167
brown_franc = 0.0185 * dims_spaced ** 1.9 * 1000
mitchell = 0.022*dims_spaced**2*1000

ax.plot(dims_spaced, powerlaw(dims_spaced, amp, index), label='Power Law')
ax.plot(dims_spaced, mass_bulk, label='Bulk Ice', linestyle='--')
ax.plot(dims_spaced, brown_franc, label='Brown&Francis', linestyle='--')
ax.plot(dims_spaced, mitchell, label='Mitchell 90', linestyle='--')
# plt.plot(dims_spaced, dims_spaced)
plt.xlabel('Long Axis in um')
plt.ylabel('Mass in some g')
plt.title('Mass Dimension Relation')

legend = ax.legend(frameon=True, scatterpoints=1, loc=legloc)
light_grey = np.array([float(248)/float(255)]*3)
rect = legend.get_frame()
rect.set_facecolor(light_grey)
rect.set_linewidth(0.0)

plt.show()
