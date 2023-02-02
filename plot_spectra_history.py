import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from loess.loess_1d import loess_1d
import argparse
from bokeh.palettes import Dark2
import matplotlib
import sys

from util import *

plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=5)
matplotlib.rc("ytick", labelsize=5)
matplotlib.rc("axes", labelsize=6)
matplotlib.rc("axes", titlesize=6)
matplotlib.rc("legend", fontsize=6)

pops = ["ALL", "AFR", "EAS", "EUR", "SAS"]

classes = [
    "A>C",
    "A>G",
    "A>T",
    "C>A",
    "C>G",
    "C>T",
]

D = Dark2[6]
colors = [D[1], D[2], D[5], D[4], D[3], D[0]]


def make_parser():
    ADHF = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser("bin_variant_ages.py", formatter_class=ADHF)
    optional = parser.add_argument_group("Optional")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
    )
    optional.add_argument("--max_age", "-m", default=10000, type=float)
    optional.add_argument(
        "--singletons",
        "-s",
        action="store_true",
    )
    optional.add_argument(
        "--frequency",
        "-f",
        type=float,
        default=0.98,
    )
    return parser


def get_relative_spectra_histories(
    dataset, pop, max_age=10000, keep_singletons=False, max_frequency=0.98
):
    fname = f"./data/binned_ages.{dataset}.{pop}.max_age.{max_age}.singletons.{keep_singletons}.max_frequency.{max_frequency}.csv"
    df = pd.read_csv(fname, sep="\t")
    y = np.zeros((100, len(classes)))
    for i, c in enumerate(classes):
        y[:, i] = df[c]
    for j in range(len(y)):
        y[j] /= y[j].sum()
    spectra = {}
    for i, c in enumerate(classes):
        spectra[c] = y[:, i]
    bin_mids = np.array((df["Min"] + df["Max"]) / 2)
    return spectra, bin_mids


def plot_spectra(
    dataset, max_age, keep_singletons, max_frequency, fout=None, show=False
):
    spectra = {}
    bin_mids = {}
    for p in pops:
        spectra[p], bin_mids[p] = get_relative_spectra_histories(
            dataset, p, max_age=max_age, keep_singletons=keep_singletons
        )
    anchor = {c: spectra["ALL"][c][0] for c in classes}
    fig = plt.figure(figsize=(8, 5))
    fig.clf()
    for k, p in enumerate(pops):
        ax = plt.subplot(2, 3, k + 1)
        for j, c in enumerate(classes):
            y = (spectra[p][c] - anchor[c]) * 100
            ax.plot(bin_mids[p], y, c=colors[j], lw=0.1, label=None)
        for j, c in enumerate(classes):
            y = (spectra[p][c] - anchor[c]) * 100
            xout, yout, wout = loess_1d(bin_mids[p], y, frac=0.5, degree=2)
            ax.plot(xout, yout, c=colors[j % 6], lw=2.0, label=c)
        ax.set_xscale("log")
        if k == 4:
            ax.legend(ncol=1, fontsize=6, bbox_to_anchor=(1.0, 1.0))
        ax.set_title(p)
        ax.set_ylabel("Percent change")
        ax.set_xlabel("Generations ago")

    fig.tight_layout()
    if fout is not None:
        plt.savefig(fout)
    if show:
        plt.show()


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])
    (dataset, max_age, keep_singletons, max_frequency) = (
        args.dataset,
        args.max_age,
        args.singletons,
        args.frequency,
    )
    if keep_singletons:
        fname = (
            f"plots/spectrum_history.{dataset}.max_age.{int(max_age)}.singletons.pdf"
        )
    else:
        fname = f"plots/spectrum_history.{dataset}.max_age.{int(max_age)}.pdf"
    plot_spectra(
        dataset, max_age, keep_singletons, max_frequency, fout=fname, show=False
    )
