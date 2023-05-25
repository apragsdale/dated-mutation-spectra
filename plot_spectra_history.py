import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from loess.loess_1d import loess_1d
import argparse
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

from bokeh.palettes import Dark2
from bokeh.palettes import Bright

# D = Dark2[6]
# colors = [D[1], D[2], D[5], D[4], D[3], D[0]]
colors = Bright[6]


def make_parser():
    ADHF = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser("plot_spectra_history.py", formatter_class=ADHF)
    optional = parser.add_argument_group("Optional")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
    )
    optional.add_argument("--max_age", "-m", default=10000, type=int)
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
    optional.add_argument(
        "--num_bins",
        "-b",
        type=int,
        default=100,
    )
    optional.add_argument(
        "--CI",
        action="store_true",
    )
    return parser


def get_relative_spectra_histories(
    dataset,
    pop,
    max_age=10000,
    keep_singletons=False,
    max_frequency=0.98,
    num_bins=100,
    CI=False,
):
    if num_bins != 100:
        assert CI is False
        fname = f"./data/binned_ages.{dataset}.{pop}.max_age.{max_age}.singletons.{keep_singletons}.max_frequency.{max_frequency}.num_bins.{num_bins}.csv"
    elif CI is True:
        assert num_bins == 100
        fname = f"./data/binned_ages.{dataset}.{pop}.max_age.{max_age}.singletons.{keep_singletons}.max_frequency.{max_frequency}.CI.csv"
    elif num_bins == 100:
        fname = f"./data/binned_ages.{dataset}.{pop}.max_age.{max_age}.singletons.{keep_singletons}.max_frequency.{max_frequency}.csv"
    else:
        fname = f"./data/binned_ages.{dataset}.{pop}.max_age.{max_age}.singletons.{keep_singletons}.max_frequency.{max_frequency}.num_bins.{num_bins}.csv"
    df = pd.read_csv(fname, sep="\t")
    y = np.zeros((num_bins, len(classes)))
    for i, c in enumerate(classes):
        y[:, i] = df[c]
    for j in range(len(y)):
        y[j] /= y[j].sum()
    spectra = {}
    for i, c in enumerate(classes):
        spectra[c] = y[:, i]
    bin_mids = np.array((df["Min"] + df["Max"]) / 2)
    return spectra, bin_mids


def get_relative_spectra_histories_extra_classes(
    dataset,
    pop,
    max_age=10000,
    keep_singletons=False,
    max_frequency=0.98,
    num_bins=100,
    CpG=False,
):
    if num_bins == 100:
        fname = f"./data/binned_ages.{dataset}.{pop}.max_age.{max_age}.singletons.{keep_singletons}.max_frequency.{max_frequency}.csv"
    else:
        fname = f"./data/binned_ages.{dataset}.{pop}.max_age.{max_age}.singletons.{keep_singletons}.max_frequency.{max_frequency}.num_bins.{num_bins}.csv"
    df = pd.read_csv(fname, sep="\t")
    additional_classes = ["ACC>ATC", "CCC>CTC", "TCC>TTC", "TCT>TTT"]
    if CpG:
        additional_classes.append("CpG")
    classes_all = classes + additional_classes
    y = np.zeros((num_bins, len(classes_all)))
    for i, c in enumerate(classes_all):
        y[:, i] = df[c]
    for j in range(len(y)):
        y[j] /= y[j].sum()
    spectra = {}
    for i, c in enumerate(classes_all):
        spectra[c] = y[:, i]
    bin_mids = np.array((df["Min"] + df["Max"]) / 2)
    return spectra, bin_mids


def plot_spectra(
    dataset,
    max_age,
    keep_singletons,
    max_frequency,
    fout=None,
    show=False,
    CI=False,
    num_bins=100,
):
    spectra = {}
    bin_mids = {}
    for p in pops:
        spectra[p], bin_mids[p] = get_relative_spectra_histories(
            dataset,
            p,
            max_age=max_age,
            keep_singletons=keep_singletons,
            max_frequency=max_frequency,
            CI=CI,
            num_bins=num_bins,
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


def plot_spectra_extra_classes(
    dataset,
    max_age,
    keep_singletons,
    max_frequency,
    fout=None,
    show=False,
    CpG=False,
):
    spectra = {}
    bin_mids = {}
    for p in pops:
        spectra[p], bin_mids[p] = get_relative_spectra_histories_extra_classes(
            dataset, p, max_age=max_age, keep_singletons=keep_singletons, CpG=CpG
        )
        spectra[p]["C>T (pulse)"] = (
            spectra[p]["ACC>ATC"]
            + spectra[p]["CCC>CTC"]
            + spectra[p]["TCC>TTC"]
            + spectra[p]["TCT>TTT"]
        )
        for c in ["ACC>ATC", "CCC>CTC", "TCC>TTC", "TCT>TTT"]:
            spectra[p].pop(c)

    classes_all = classes + ["C>T (pulse)"]
    if CpG:
        classes_all.append("CpG")

    anchor = {c: spectra["ALL"][c][0] for c in classes_all}
    colors = list(Bright[6]) + ["black", "gray"]
    fig = plt.figure(figsize=(8, 5))
    fig.clf()
    for k, p in enumerate(pops):
        ax = plt.subplot(2, 3, k + 1)
        for j, c in enumerate(classes_all):
            y = (spectra[p][c] - anchor[c]) * 100
            ax.plot(bin_mids[p], y, c=colors[j], lw=0.1, label=None)
        for j, c in enumerate(classes_all):
            y = (spectra[p][c] - anchor[c]) * 100
            xout, yout, wout = loess_1d(bin_mids[p], y, frac=0.5, degree=2)
            ax.plot(xout, yout, c=colors[j], lw=2.0, label=c)
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
    (dataset, max_age, keep_singletons, max_frequency, num_bins, CI) = (
        args.dataset,
        args.max_age,
        args.singletons,
        args.frequency,
        args.num_bins,
        args.CI,
    )
    if num_bins != 100:
        assert CI is False
        assert keep_singletons is False
        fname = f"plots/spectrum_history.{dataset}.max_age.{int(max_age)}.num_bins.{num_bins}.pdf"
    elif CI:
        assert keep_singletons is False
        fname = f"plots/spectrum_history.{dataset}.max_age.{int(max_age)}.CI.pdf"
    elif keep_singletons:
        fname = (
            f"plots/spectrum_history.{dataset}.max_age.{int(max_age)}.singletons.pdf"
        )
    elif max_frequency != 0.98:
        fname = f"plots/spectrum_history.{dataset}.max_age.{int(max_age)}.max_freq.{max_frequency}.pdf"
    else:
        fname = f"plots/spectrum_history.{dataset}.max_age.{int(max_age)}.pdf"
    plot_spectra(
        dataset,
        max_age,
        keep_singletons,
        max_frequency,
        fout=fname,
        show=False,
        CI=CI,
        num_bins=num_bins,
    )
