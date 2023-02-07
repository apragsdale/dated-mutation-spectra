"""
- Three mutation spectrum histories.
- Recent mutation spectrum between three datasets, w/ and w/out singletons, compared
  to Iceland trios
- Correlations in allele ages between all mutations and a subset, comparing
  GEVA and Relate
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=5)
matplotlib.rc("ytick", labelsize=5)
matplotlib.rc("axes", labelsize=6)
matplotlib.rc("axes", titlesize=6)
matplotlib.rc("legend", fontsize=4)

import pandas as pd

from loess.loess_1d import loess_1d
import numpy as np

from util import *
from plot_spectra_history import get_relative_spectra_histories

from bokeh.palettes import Dark2
from bokeh.palettes import Colorblind as cb

D = Dark2[6]
colors = [D[1], D[2], D[5], D[4], D[3], D[0]]

bins = {}
spectra = {}

for dataset in ["geva", "relate", "tsdate"]:
    s, b = get_relative_spectra_histories(dataset, "ALL")
    bins[dataset] = b
    spectra[dataset] = s

classes = ["A>C", "A>G", "A>T", "C>A", "C>G", "C>T"]


def plot_spectra(dataset, ax):
    for i, c in enumerate(classes):
        y = (spectra[dataset][c] - spectra[dataset][c][0]) * 100
        x = bins[dataset]
        xout, yout, wout = loess_1d(x, y, frac=0.5, degree=2)
        ax.plot(xout, yout, c=colors[i], lw=1.0, label=c)
    ax.set_xscale("log")
    ax.set_ylabel("Percent change")
    ax.set_xlabel("Generations ago")


fig = plt.figure(1, figsize=(6.5, 5))
fig.clf()
grid = (3, 7)

ax1 = plt.subplot2grid(grid, (0, 0), colspan=3, rowspan=1)
dataset = "geva"
plot_spectra(dataset, ax1)
ax1.set_title("GEVA")

ax2 = plt.subplot2grid(grid, (1, 0), colspan=3, rowspan=1)
dataset = "relate"
plot_spectra(dataset, ax2)
ax2.set_title("Relate")

ax3 = plt.subplot2grid(grid, (2, 0), colspan=3, rowspan=1)
dataset = "tsdate"
plot_spectra(dataset, ax3)
ax3.set_title("tsdate")

ylim = max(
    [
        max(np.abs(ax1.get_ylim())),
        max(np.abs(ax2.get_ylim())),
        max(np.abs(ax3.get_ylim())),
    ]
)
ax1.set_ylim(-ylim, ylim)
ax2.set_ylim(-ylim, ylim)
ax3.set_ylim(-ylim, ylim)

ax1.legend(ncol=2, loc="upper left", frameon=False)

ax4 = plt.subplot2grid(grid, (0, 3), colspan=4, rowspan=1)
# recent spectra (last 100 gens) against Iceland trios
geva_recent = np.array([0.0946, 0.3600, 0.0886, 0.1201, 0.1057, 0.2310])
relate_recent = np.array([0.0989, 0.3598, 0.0908, 0.1168, 0.1062, 0.2275])
tsdate_recent = np.array([0.1002, 0.3590, 0.0921, 0.1164, 0.1060, 0.2263])
trios = np.array([0.0962, 0.3638, 0.0923, 0.0951, 0.1202, 0.2324])

width = 0.2
x = np.arange(6)

ax4.bar(x - 1.5 * width, geva_recent, width, color=cb[5][0], label="GEVA")
ax4.bar(x - 0.5 * width, relate_recent, width, color=cb[5][1], label="Relate")
ax4.bar(x + 0.5 * width, tsdate_recent, width, color=cb[5][2], label="tsdate")
ax4.bar(x + 1.5 * width, trios, width, color=cb[5][3], label="Pedigree")

ax4.legend(frameon=False, loc="upper left")
ax4.set_xticks(x)
ax4.set_xticklabels(classes)
ax4.set_ylabel("Relative proportion")
ax4.set_title("Mutation spectrum among young variants")

ages = pd.read_pickle("ages.shared.geva-relate.pkl.gz")

max_age = max(max(ages["AgeGEVA"]), max(ages["AgeRelate"]))
min_age = min(max(ages["AgeGEVA"]), min(ages["AgeRelate"]))

x = np.logspace(np.log10(min_age), np.log10(max_age), 200)


def age_heat(ax, mut=None):
    ax.axline([0, 0], [1, 1], color="k", lw=0.1)
    cmap = "cubehelix_r"
    # cmap = "gist_heat_r"
    if mut is None:
        _, _, _, image = ax.hist2d(
            ages["AgeGEVA"], ages["AgeRelate"], bins=(x, x), cmap=cmap
        )

    else:
        _, _, _, image = ax.hist2d(
            ages[ages["Mut"] == mut]["AgeGEVA"],
            ages[ages["Mut"] == mut]["AgeRelate"],
            bins=(x, x),
            cmap=cmap,
        )
    image.set_edgecolor("face")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e2, 1e5)
    ax.set_ylim(1e2, 1e5)
    ax.set_xlabel("Allele age (GEVA)")
    ax.set_ylabel("Allele age (Relate)")
    if mut is None:
        mut = "All"
    ax.set_title(mut + " mutations")


ax5 = plt.subplot2grid(grid, (1, 3), colspan=2, rowspan=1)
age_heat(ax5)

ax6 = plt.subplot2grid(grid, (1, 5), colspan=2, rowspan=1)
age_heat(ax6, mut="A>G")

def age_densities(shared, unique, ax, legend=False, dataset=""):
    v = np.concatenate((shared, unique))
    sns.kdeplot(
        [v, shared, unique],
        common_norm=True,
        fill=True,
        ax=ax,
        legend=False,
    )
    ax.set_xticks([2, 3, 4, 5])
    ax.set_xticklabels([100, 1000, 10000, 100000])
    ax.set_xlabel("Generations ago")
    ax.set_xlim(1.5, 5.5)
    ax.set_yticks([])
    ax.set_title(dataset + " A>G mutations")
    if legend:
        ax.legend(["All", "Shared", "Unique"])


ax7 = plt.subplot2grid(grid, (2, 3), colspan=2, rowspan=1)
ages_geva = pd.read_pickle("ages.unique.geva.pkl.gz")
shared_ages = np.log10(ages[ages["Mut"] == "A>G"]["AgeGEVA"])
unique_ages = np.log10(ages_geva[ages_geva["Mut"] == "A>G"]["Age"])

age_densities(shared_ages, unique_ages, ax7, legend=True, dataset="GEVA")

ax8 = plt.subplot2grid(grid, (2, 5), colspan=2, rowspan=1)
ages_relate = pd.read_pickle("ages.unique.relate.pkl.gz")
shared_ages = np.log10(ages[ages["Mut"] == "A>G"]["AgeRelate"])
unique_ages = np.log10(ages_relate[ages_relate["Mut"] == "A>G"]["Age"])

age_densities(shared_ages, unique_ages, ax8, legend=False, dataset="Relate")


fig.tight_layout()
plt.savefig("plots/fig1.pdf")

