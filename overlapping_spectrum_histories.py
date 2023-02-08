import pandas as pd
import numpy as np
import pickle
import gzip

from bin_variant_ages import add_mut_class

# overlapping sites and load data
overlap = pickle.load(gzip.open("data/overlapping-pos.geva-relate.gz", "rb"))

max_age = 1e4

classes = ["A>C", "A>G", "A>T", "C>A", "C>G", "C>T"]

dfs = {"geva": {}, "relate": {}}

for i in range(22):
    for dataset in dfs.keys():
        chrom = i + 1
        df = pd.read_pickle(f"data/{dataset}/parsed_variants.chr{chrom}.gzip")
        df = df[df["Pos"].isin(overlap[i])]
        # df = df[df["Age"] < max_age]
        df = df[df["AC"] > 1]
        df = df[df["ALL"] <= 0.98]
        df = add_mut_class(df)
        df = df[df["Mut"].isin(classes)]
        dfs[dataset][chrom] = df
    o = set(dfs["geva"][chrom]["Pos"]).intersection(set(dfs["relate"][chrom]["Pos"]))
    for dataset in dfs.keys():
        dfs[dataset][chrom] = dfs[dataset][chrom][dfs[dataset][chrom]["Pos"].isin(o)]
        dfs[dataset][chrom] = dfs[dataset][chrom][dfs[dataset][chrom]["Age"] < max_age]
    print("finished", chrom)


# get ages and set up bins

ages_geva = np.concatenate([dfs["geva"][chrom]["Age"] for chrom in range(1, 23)])
ages_relate = np.concatenate([dfs["relate"][chrom]["Age"] for chrom in range(1, 23)])


ages_geva = np.sort(ages_geva)
ages_relate = np.sort(ages_relate)

num_bins = 100
bins_geva = []
bins_relate = []

bin_size_geva = int(len(ages_geva) / num_bins)
bin_size_relate = int(len(ages_relate) / num_bins)
for i in range(num_bins):
    bins_geva.append(ages_geva[bin_size_geva * i])
    bins_relate.append(ages_relate[bin_size_relate * i])

bins_geva.append(1e4)
bins_relate.append(1e4)


# count mutation types over bins

counts_geva = {i: [0, 0, 0, 0, 0, 0] for i in range(num_bins)}
counts_relate = {i: [0, 0, 0, 0, 0, 0] for i in range(num_bins)}

classes = ["A>C", "A>G", "A>T", "C>A", "C>G", "C>T"]

for chrom in range(1, 23):
    df_geva = dfs["geva"][chrom]
    df_relate = dfs["relate"][chrom]
    for i, (b0, b1) in enumerate(zip(bins_geva[:-1], bins_geva[1:])):
        df2 = df_geva[(df_geva["Age"] >= b0) & (df_geva["Age"] < b1)]
        for j, c in enumerate(classes):
            counts_geva[i][j] += len(df2[df2["Mut"] == c])
    for i, (b0, b1) in enumerate(zip(bins_relate[:-1], bins_relate[1:])):
        df2 = df_relate[(df_relate["Age"] >= b0) & (df_relate["Age"] < b1)]
        for j, c in enumerate(classes):
            counts_relate[i][j] += len(df2[df2["Mut"] == c])
    print(chrom)

for i in range(num_bins):
    counts_geva[i] = [_ / sum(counts_geva[i]) for _ in counts_geva[i]]
    counts_relate[i] = [_ / sum(counts_relate[i]) for _ in counts_relate[i]]


# plots

from bokeh.palettes import Dark2
import matplotlib
import matplotlib.pylab as plt
from loess.loess_1d import loess_1d

plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=5)
matplotlib.rc("ytick", labelsize=5)
matplotlib.rc("axes", labelsize=6)
matplotlib.rc("axes", titlesize=6)
matplotlib.rc("legend", fontsize=6)

D = Dark2[6]
colors = [D[1], D[2], D[5], D[4], D[3], D[0]]

fig = plt.figure(999, figsize=(6.5, 3))
fig.clf()
ax1 = plt.subplot(1, 2, 1)
x = (np.array(bins_geva[:-1]) + np.array(bins_geva[1:])) / 2
for j, c in enumerate(classes):
    y = np.array(
        [(counts_geva[i][j] - counts_geva[0][j]) * 100 for i in range(num_bins)]
    )
    ax1.plot(x, y, c=colors[j], lw=0.1, label=None)
for j, c in enumerate(classes):
    y = np.array(
        [(counts_geva[i][j] - counts_geva[0][j]) * 100 for i in range(num_bins)]
    )
    xout, yout, wout = loess_1d(x, y, frac=0.5, degree=2)
    ax1.plot(x, yout, c=colors[j], lw=2.0, label=c)
ax1.set_xscale("log")
ax1.set_title("GEVA")
ax1.set_ylabel("Percent change")
ax1.set_xlabel("Generations ago")
ax1.legend(ncol=2, fontsize=5)
ax1 = plt.subplot(1, 2, 2)
x = (np.array(bins_relate[:-1]) + np.array(bins_relate[1:])) / 2
for j, c in enumerate(classes):
    y = np.array(
        [(counts_relate[i][j] - counts_relate[0][j]) * 100 for i in range(num_bins)]
    )
    ax1.plot(x, y, c=colors[j], lw=0.1, label=None)
for j, c in enumerate(classes):
    y = np.array(
        [(counts_relate[i][j] - counts_relate[0][j]) * 100 for i in range(num_bins)]
    )
    xout, yout, wout = loess_1d(x, y, frac=0.5, degree=2)
    ax1.plot(x, yout, c=colors[j], lw=2.0, label=c)
ax1.set_xscale("log")
ax1.set_title("Relate")
ax1.set_xlabel("Generations ago")
plt.tight_layout()
plt.savefig("plots/overlapping.geva.relate.pdf")


