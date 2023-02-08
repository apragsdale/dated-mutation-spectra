"""
Showing issues with mutation spectra.

Panels:
- Correlation between shared mutations, with r^2.
- Geva vs Relate shared and unique
- Recent mutation spectrum between three datasets, w/ and w/out singletons, compared 
  to Iceland trios

"""
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=5)
matplotlib.rc("ytick", labelsize=5)
matplotlib.rc("axes", labelsize=6)
matplotlib.rc("axes", titlesize=6)
matplotlib.rc("legend", fontsize=5)

from matplotlib.ticker import LogLocator

import pandas as pd
import numpy as np

from util import *
from bokeh.palettes import Colorblind as cb

classes = ["A>C", "A>G", "A>T", "C>A", "C>G", "C>T"]

def age_heat(ax, age1, age2, dataset1=None, dataset2=None):
    min_age = 1e2
    max_age = 1e5
    x = np.logspace(np.log10(min_age), np.log10(max_age), 200)
    ax.axline([0, 0], [1, 1], color="k", lw=0.1)
    cmap = "cubehelix_r"
    # cmap = "gist_heat_r"
    _, _, _, image = ax.hist2d(
        age1, age2, bins=(x, x), cmap=cmap
    )
    image.set_edgecolor("face")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e2, 1e5)
    ax.set_ylim(1e2, 1e5)
    ax.set_xlabel(f"Allele age ({dataset1})")
    ax.set_ylabel(f"Allele age ({dataset2})")
    ax.set_xticks([100, 1000, 10000, 100000])
    ax.set_yticks([100, 1000, 10000, 100000])
    ax.set_aspect('equal')
    r = np.corrcoef(ages1, ages2)[0][1]
    ax.text(20000, 200, rf"$r^2={r**2:0.2f}$", fontsize=5, va="center", ha="center")
    ax.xaxis.set_minor_locator(LogLocator(subs=(1.0,)))
    ax.yaxis.set_minor_locator(LogLocator(subs=(1.0,)))

grid = (6, 3)
fig = plt.figure(2, figsize=(6, 4.5))
fig.clf()

# heat plots

ax1 = plt.subplot2grid(grid, (0, 0), rowspan=2)
ages1 = np.array([])
ages2 = np.array([])
for chrom in range(22):
    ages = pd.read_pickle("data/ages.shared.chr{0}.geva-relate.pkl.gz".format(chrom))
    ages1 = np.concatenate((ages1, ages["AgeGEVA"]))
    ages2 = np.concatenate((ages2, ages["AgeRelate"]))

age_heat(ax1, ages1, ages2, dataset1="GEVA", dataset2="Relate")
print("Plotted heat map for GEVA and Relate")

ax2 = plt.subplot2grid(grid, (2, 0), rowspan=2)
ages1 = np.array([])
ages2 = np.array([])
for chrom in range(22):
    ages = pd.read_pickle("data/ages.shared.chr{0}.geva-tsdate.pkl.gz".format(chrom))
    ages1 = np.concatenate((ages1, ages["AgeGEVA"]))
    ages2 = np.concatenate((ages2, ages["Agetsdate"]))

age_heat(ax2, ages1, ages2, dataset1="GEVA", dataset2="tsdate")
print("Plotted heat map for GEVA and tsdate")

ax3 = plt.subplot2grid(grid, (4, 0), rowspan=2)
ages1 = np.array([])
ages2 = np.array([])
for chrom in range(22):
    ages = pd.read_pickle("data/ages.shared.chr{0}.relate-tsdate.pkl.gz".format(chrom))
    ages1 = np.concatenate((ages1, ages["AgeRelate"]))
    ages2 = np.concatenate((ages2, ages["Agetsdate"]))

age_heat(ax3, ages1, ages2, dataset1="Relate", dataset2="tsdate")
print("Plotted heat map for Relate and tsdate")

# bar plots (shared + unique, and young vs trios)
ax4 = plt.subplot2grid(grid, (0, 1), colspan=grid[1]-1, rowspan=3)
ax5 = plt.subplot2grid(grid, (3, 1), colspan=grid[1]-1, rowspan=3)

# shared and unique calls

mutation_counts = {"shared": defaultdict(int), "geva": defaultdict(int), "relate": defaultdict(int)}
for chrom in range(22):
    ages = pd.read_pickle("data/ages.shared.chr{0}.geva-relate.pkl.gz".format(chrom))
    ages_geva = pd.read_pickle("data/ages.unique.chr{0}.geva.pkl.gz".format(chrom))
    ages_relate = pd.read_pickle("data/ages.unique.chr{0}.relate.pkl.gz".format(chrom))
    for c in classes:
        mutation_counts["shared"][c] += len(ages[ages["Mut"] == c])
        mutation_counts["geva"][c] += len(ages_geva[ages_geva["Mut"] == c])
        mutation_counts["relate"][c] += len(ages_relate[ages_relate["Mut"] == c])

totals = totals = {k: sum(v.values()) for k, v in mutation_counts.items()}
props = {k: [v[c] / totals[k] for c in classes] for k, v in mutation_counts.items()}

width = 0.2
x = np.arange(6)

ax4.bar(x - width, props["shared"], 0.9*width, color=cb[5][0], label="Shared")
ax4.bar(x, props["geva"], 0.9*width, color=cb[5][1], label="GEVA-unique")
ax4.bar(x + width, props["relate"], 0.9*width, color=cb[5][2], label="Relate-unique")

ax4.legend(frameon=False, loc="upper left")
ax4.set_xticks(x)
ax4.set_xticklabels(classes)
ax4.set_ylabel("Relative proportion")
ax4.set_title("Mutation spectrum among overlapping and uniquely dated variants")
ax4.set_xlabel("Mutation classes")

print("Plotted overlapping and unique variants")

# recent spectra (last 100 gens) against Iceland trios
geva_recent = np.array([0.0946, 0.3600, 0.0886, 0.1201, 0.1057, 0.2310])
relate_recent = np.array([0.0989, 0.3598, 0.0908, 0.1168, 0.1062, 0.2275])
tsdate_recent = np.array([0.1002, 0.3590, 0.0921, 0.1164, 0.1060, 0.2263])
trios = np.array([0.0962, 0.3638, 0.0923, 0.0951, 0.1202, 0.2324])

ax5.bar(x - 1.5 * width, geva_recent, 0.9 * width, color=cb[5][0], label="GEVA")
ax5.bar(x - 0.5 * width, relate_recent, 0.9 * width, color=cb[5][1], label="Relate")
ax5.bar(x + 0.5 * width, tsdate_recent, 0.9 * width, color=cb[5][2], label="tsdate")
ax5.bar(x + 1.5 * width, trios, 0.9 * width, color=cb[5][3], label="Pedigree")

ax5.legend(frameon=False, loc="upper left")
ax5.set_xticks(x)
ax5.set_xticklabels(classes)
ax5.set_ylabel("Relative proportion")
ax5.set_title("Mutation spectrum among young variants")
ax5.set_xlabel("Mutation classes")

print("Plotted young variants")

fig.tight_layout()
fig.subplots_adjust(left=0)
fig.text(0.03, 0.97, "A", fontsize=8, va="center", ha="center")
fig.text(0.03, 0.65, "B", fontsize=8, va="center", ha="center")
fig.text(0.03, 0.33, "C", fontsize=8, va="center", ha="center")
fig.text(0.28, 0.97, "D", fontsize=8, va="center", ha="center")
fig.text(0.28, 0.49, "E", fontsize=8, va="center", ha="center")

plt.savefig("plots/fig2.pdf")

"""
### age densities
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
"""
