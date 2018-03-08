from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import pickle
from scipy import optimize
from scipy import stats
from itertools import cycle
# from pylab import *
from matplotlib import style
# style.use('dark_background')


# Loading data ############################

folder_list = (

    # Clean Measurements

    '/ipa2/holo/mweitzel/HIVIS_Holograms/Prev23Feb/',  # Columnar, Irregular
    '/ipa2/holo/mweitzel/HIVIS_Holograms/Meas28Feb/M2/',  # Dendritic

)

compare_list_folder = '/ipa2/holo/mweitzel/HIVIS_Holograms/Meas28Feb/M2/'  # Dendritic

full_dim_list = list()
full_v_list = list()

for folder in folder_list:
    tmp = pickle.load(open(folder + 'vs_dim_data.dat', 'rb'))
    dim_list = tmp['dims']
    v_list = tmp['vs']

    full_dim_list += dim_list
    full_v_list += v_list


full_dim_list, full_v_list = zip(*sorted(zip(full_dim_list, full_v_list)))

tmp = pickle.load(open(compare_list_folder+'vs_dim_data.dat', 'rb'))

comp_dim_list = tmp['dims']
comp_v_list = tmp['vs']

# Fitting power law ############################

powerlaw = lambda x, amp, index: amp * (x**index)


def fit_powerlaw(x, y):

    x = [this_x for this_x, this_y in zip(x, y) if not(np.isnan(this_y))]
    y = [this_y for this_y in y if not(np.isnan(this_y))]

    logx = np.log10(x)
    logy = np.log10(y)

    fitfunc = lambda p, x: p[0]+p[1]*x
    errfunc = lambda p, x, y: (y-fitfunc(p,x))

    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit, args=(logx, logy), full_output=1)

    pfinal = out[0]
    covar = out[1]

    index = pfinal[1]
    amp = 10.0**pfinal[0]
    pl = amp * (x**index)

    return amp, index


amp_full, index_full = fit_powerlaw(full_dim_list, full_v_list)

# Plotting things ############################

dims_spaced = np.arange(np.ceil(np.max(dim_list)/10)*10)
almost_black = '#262626'
fig, ax = plt.subplots(1)
for this_dim_list, this_v_list in zip(full_dim_list, full_v_list):
        ax.scatter(full_dim_list, full_v_list, alpha=1,
                   edgecolors=almost_black, linewidth=1, zorder=0)
    # ax.errorbar([(i+j)/2 for i, j in zip(bin_edges[:-1], bin_edges[1:])], avg_masses, yerr=mass_std, fmt='o')

ax.scatter(comp_dim_list, comp_v_list, alpha=1, edgecolor=almost_black, linewidth=1, zorder=0, c='y')

ax.plot(dims_spaced, powerlaw(dims_spaced, amp_full, index_full), label='Power Law Full', linewidth=3, zorder=1)

plt.show()
