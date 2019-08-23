from utilities.make_pretty_figure import *
import numpy as np


class param:
    def __init__(self, index, exponent, name):
        self.index = index
        self.exponent = exponent
        self.name = name

    def apply_param(self, diameters):
        return [self.index*d**self.exponent for d in diameters]


density_plot = True

diams = np.arange(200)*1e-6

all = param(0.03097, 2.13, r'Data fit for $D_{sec}$')
all_binned = param(0.01037, 2.01, 'all binned')
all_area = param(0.49723, 2.36, r'Data fit for $D_{ae}$')
all_area_binned = param(0.12439, 2.22, 'all area binned')

goodonly = param(0.00289, 1.88, r'Filtered data, $D_{sec}$')
goodonly_binned = param(0.12828, 2.28, 'good only binned')
goodonly_area = param(1.28184, 2.46, r'Filtered data, $D_{ae}$')
goodonly_area_binned = param(3.58492, 2.57, 'good only area binned')

bulk_mass = param(np.pi/6*916.7, 3, 'bulk ice')

# lit_m10 = param(0.007642, 1.802, 'Mitchell 2010')
lit_m10 = param(35.133, 2.814, 'Mitchell 2010')
lit_c13_lower = param(366.519, 3, 'Cotton 2013')
lit_c13 = param(0.026, 2, 'Cotton 2013')

fig, ax = imshow_in_figure(figspan=(16, 10))

# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'k', 'tab:red', 'y', 'c']

if density_plot:
    # for par in (all, goodonly):
    par = all_area
    ax.plot([par/bulk for par, bulk in zip(par.apply_param(diams), bulk_mass.apply_param(diams))], label=par.name, lw=4, c=colors[0])
    # for par in (all_area, goodonly_area):
    par = all
    ax.plot([par/bulk for par, bulk in zip(par.apply_param(diams), bulk_mass.apply_param(diams))], label=par.name, lw=4, c=colors[1])

    # for par in (lit_m10, lit_c13):
    #     ax.plot([par/bulk for par, bulk in zip(par.apply_param(diams), bulk_mass.apply_param(diams))], '.', label=par.name, markersize=2)
    ax.plot([par/bulk for par, bulk in zip(lit_m10.apply_param(diams), bulk_mass.apply_param(diams))], '--', label=lit_m10.name, markersize=5, c=colors[4], lw=4)
    c13 = [par/bulk for par, bulk in zip(lit_c13_lower.apply_param(diams[:71]), bulk_mass.apply_param(diams[:71]))]
    c13.extend([par/bulk for par, bulk in zip(lit_c13.apply_param(diams[71:]), bulk_mass.apply_param(diams[71:]))])
    ax.plot(c13, '--', label=lit_c13.name, markersize=5, c=colors[3], lw=4)

    ax.set_xlabel(r'Maximum dimension D (different definition for each study) in $\mathrm{\mu m}$', fontsize=20)
    ax.set_ylabel(r'Relative density $\mathrm{\rho(D)/\rho_{ice}}$', fontsize=20)

else:
    for par in (all, all_binned, all_area, all_area_binned, goodonly, goodonly_binned, goodonly_area, goodonly_area_binned):
        ax.plot(par.apply_param(diams), label=par.name)
    ax.plot(bulk_mass.apply_param(diams), '--', label=bulk_mass.name)
    ax.set_ylabel('Crystal mass in g', fontsize=20)

ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlim(10, 130)
ax.set_ylim(0.05, 10)
# ax.set_xscale('log')
ax.set_yscale('log')
b = ax.get_yticklabels()
c = list(ax.get_yticks())
# b.insert(3, plt.Text(0, 0.5, '$\\mathdefault{5\cdot10^{-1}}$'))
# c.insert(3, 0.5)
# ax.set_yticklabels(b)
# ax.set_yticks(np.asarray(c))

ax.legend(fontsize=18)
ax.grid(b=True, which='minor', linestyle='--', alpha=0.4)
savefig_ipa(fig, 'powerlaw_comp')
plt.show()
