from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy import optimize
from itertools import cycle
# from pylab import *

folder_list = (
              # '/uni-mainz.de/homes/maweitze/CCR/1503/M1/',    # Irregular Dendritic, Aggregates
              '/uni-mainz.de/homes/maweitze/CCR/2203/M1/',    # Columnar
              '/uni-mainz.de/homes/maweitze/CCR/2203/M2/',    # Columnar, Bullet Rosettes
              '/uni-mainz.de/homes/maweitze/CCR/3103/M1/',    # Columnar
              # '/uni-mainz.de/homes/maweitze/CCR/0604/M1/',    # Aggregates
              '/uni-mainz.de/homes/maweitze/CCR/1107/M1/',    # Dense Irregular
              # '/uni-mainz.de/homes/maweitze/CCR/1907/M1/',    # Dendritic
              # '/uni-mainz.de/homes/maweitze/CCR/1907/M2/',    # Dendritic
              # '/uni-mainz.de/homes/maweitze/CCR/1907/M3/',    # Dendritic
               )


minsize=5
maxsize=200

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

    # this_dim_list = [float(a['Long Axis']) for a in crystal_list if (float(a['Long Axis']) > minsize) & (float(a['Long Axis']) < maxsize)]                                # Maximum dimension
    this_dim_list = [(float(a['Long Axis'])+float(a['Short Axis']))/2 for a in crystal_list if (float(a['Long Axis']) > minsize) & (float(a['Long Axis']) < maxsize)] # Mean of Maximum and Minimum dimension
    # this_dim_list = [(float(a['Long Axis'])/float(a['Short Axis'])) for a in crystal_list if (float(a['Long Axis']) > minsize) & (float(a['Long Axis']) < maxsize)]       # Aspect Ratio
    # this_dim_list = [a['Area'] for a in crystal_list if ((float(a['Long Axis']) > minsize) & (float(a['Long Axis']) < maxsize))]                                          # Area
    # this_mass_list = [float(a['Short Axis']) for a in crystal_list if (float(a['Long Axis']) > minsize) & (float(a['Long Axis']) < maxsize)]                              # Short Axis
    this_mass_list = [np.pi/6*a['Drop Diameter']**3 for a in crystal_list if ((float(a['Long Axis']) > minsize) & (float(a['Long Axis']) < maxsize))]                     # Mass
    # this_mass_list = [float(a['Drop Diameter']) for a in crystal_list if ((float(a['Long Axis']) > minsize) & (float(a['Long Axis']) < maxsize))]
    this_aspr_list = [(float(a['Short Axis'])/float(a['Long Axis'])) for a in crystal_list if (float(a['Long Axis']) > minsize) & (float(a['Long Axis']) < maxsize)]

    folders_dim_list.append(this_dim_list)
    folders_mass_list.append(this_mass_list)
    folders_aspr_list.append(this_aspr_list)

    full_dim_list += this_dim_list
    full_mass_list += this_mass_list
    index_list += [i]*len(this_dim_list)

full_dim_list, full_mass_list, index_list = zip(*sorted(zip(full_dim_list, full_mass_list, index_list)))

#
# full_dim_list = list(np.asarray(full_dim_list)*np.asarray(filter_kernel))
# full_mass_list = list(np.asarray(full_mass_list)*np.asarray(filter_kernel))

# colorlist = [[1-(0.8/c)**0.5, (0.8/c)**0.5, (0.8/c)**0.5] for c in index_list]



symbols=["o", "8", "s", "p", "*", "h", "H", "+", "x", "X", "D"]
symbolcycler = cycle(symbols)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
colorcycler = cycle(colors)

almost_black = '#262626'

min_aspr = min([min(a) for a in folders_aspr_list])

fig, ax = plt.subplots(1)

for this_dim_list, this_mass_list, this_aspr_list, this_folder in zip(folders_dim_list, folders_mass_list, folders_aspr_list, folder_list):
    # ax.scatter(this_dim_list, this_mass_list, label=this_folder[-8:], c=next(colorcycler), marker=next(symbolcycler), alpha=1, edgecolor=almost_black, linewidth=0.15)
    col_list = [[1/(1-min_aspr)*c-min_aspr/(1-min_aspr), 1/(1+min_aspr)*(1-c), 0] for c in this_aspr_list]
    ax.scatter(this_dim_list, this_mass_list, label=this_folder[-8:], c=col_list, marker=next(symbolcycler), alpha=1, edgecolor=almost_black, linewidth=0.5)

plt.xlim(0, 1.1*np.max(full_dim_list))
plt.ylim(0, 1.1*np.max(full_mass_list))

legend = ax.legend(frameon=True, scatterpoints=1)

light_grey = np.array([float(248)/float(255)]*3)
rect = legend.get_frame()
rect.set_facecolor(light_grey)
rect.set_linewidth(0.0)

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

plt.text(25, np.max(full_mass_list)*0.9, 'Brown&Francis: $m=0.0185\cdot D^{1.9}$', bbox=dict(facecolor='red', alpha=0.2))
plt.text(25, np.max(full_mass_list)*0.95,  'Power Law Fit: $m = {0:.4f}\cdot D^{{{1:.3f}}}$\nn = {2}, unique: {3}'.format(amp/1000, index, n, nu), bbox=dict(facecolor='blue', alpha=0.2))

plt.text(25, np.max(full_mass_list)*0.8,  'Minimum aspect ratio: $AR_{{min}} = {0:.2f}$\n Red for high AR, Green for low AR'.format(min_aspr), bbox=dict(facecolor='green', alpha=0.2))

plt.plot(full_dim_list, powerlaw(full_dim_list, amp, index))

dims_spaced = np.arange(150)
mass_bulk = np.pi/6*(dims_spaced)**3*0.9167
brown_franc = 0.0185*dims_spaced**1.9*1000


plt.plot(dims_spaced, mass_bulk)
plt.plot(dims_spaced, brown_franc)
# plt.plot(dims_spaced, dims_spaced)
plt.xlabel('Long Axis in um')
plt.ylabel('Mass in some g')
plt.title('Mass Dimension Relation')

plt.show()

