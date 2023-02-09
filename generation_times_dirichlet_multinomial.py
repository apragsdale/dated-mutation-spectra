import pandas as pd
import numpy as np
import scipy.optimize
import scipy.stats
from loess.loess_1d import loess_1d
import matplotlib.pylab as plt
import matplotlib

plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=5)
matplotlib.rc("ytick", labelsize=5)
matplotlib.rc("axes", labelsize=6)
matplotlib.rc("axes", titlesize=6)
matplotlib.rc("legend", fontsize=6)

import pickle

from util import *

import argparse

from bokeh.palettes import Dark2
from bokeh.palettes import Bright

#D = Dark2[6]
#colors = [D[1], D[2], D[5], D[4], D[3], D[0]]
colors = Bright[6]

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
    optional.add_argument(
        "--max_age",
        "-m",
        type=int,
        default=10000,
    )
    return parser


nucleotides = ["A", "C", "G", "T"]

mut_classes = ["A>C", "A>G", "A>T", "C>A", "C>G", "C>T"]
complementary_classes = ["T>G", "T>C", "T>A", "G>T", "G>C", "G>A"]
class_map = {m1: m2 for m1, m2 in zip(complementary_classes, mut_classes)}


def predict_spectrum(ages):
    """
    This comes from the `age_modeling.R` script from Wang et al.
    """
    alpha = np.array([13.830321, 15.180457, 14.056053, 13.923672, 13.952551, 14.947698])
    beta0 = np.array([-0.316633, -0.327940, -0.322887, -0.329628, -0.321475, -0.326378])
    beta1 = np.array([0.252819, 0.265539, 0.249886, 0.264401, 0.262430, 0.256306])
    p = np.exp(alpha + ages[0] * beta0 + ages[1] * beta1)
    return p / np.sum(p)


def clr(x):
    geom_mean = np.prod(x) ** (1 / len(x))
    return np.log(x / geom_mean)


def cost_func(ages, data, model_func, bin_idx, iceland_spectrum, anchor):
    predicted = model_func(ages)
    predicted /= predicted.sum()
    compared = data[bin_idx] / np.sum(data[bin_idx])
    delta = anchor / np.sum(anchor) - iceland_spectrum / iceland_spectrum.sum()
    dist = scipy.linalg.norm(clr(predicted) - clr(compared - delta))
    return dist


def get_mutation_counts(dataset, pop="ALL", max_age=10000, singletons=False):
    df_counts = pd.read_csv(
        f"data/binned_ages.{dataset}.{pop}.max_age.{max_age}.singletons.{singletons}.max_frequency.0.98.csv",
        sep="\t",
    )
    return df_counts


def get_data_spectra(dataset, pop, max_age=10000):
    counts = get_mutation_counts(dataset, pop=pop, max_age=max_age)
    num_bins = len(counts)
    bins = list(zip(counts["Min"], counts["Max"]))
    y = np.zeros((num_bins, 6))
    for i in range(num_bins):
        y[i] = np.array([counts[c][i] for c in mut_classes])
    return y, bins


def fit_generation_times(dataset, pop, iceland_spectrum, max_age=10000):
    data_spectra, bins = get_data_spectra(dataset, pop, max_age=max_age)
    anchor_spectra, bins = get_data_spectra(dataset, "ALL", max_age=max_age)
    anchor = anchor_spectra[0]
    fit_ages = np.zeros((100, 2))
    for bin_idx in range(100):
        p0 = [30, 30]

        args = (
            data_spectra,
            predict_spectrum,
            bin_idx,
            iceland_spectrum,
            anchor,
        )

        ret = scipy.optimize.fmin_l_bfgs_b(
            cost_func, p0, args=args, approx_grad=True, bounds=[[14, 60], [14, 60]]
        )
        fit_ages[bin_idx] = ret[0]
    return bins, fit_ages


def plot_inferred_generation_times(bins, fit_ages, dataset, savefig=False):
    fig = plt.figure(3, figsize=(8, 5))
    fig.clf()
    ylim = (14, 50)
    sex_averaged = {}
    for i, pop in enumerate(bins.keys()):
        ages = fit_ages[pop]
        bin_mids = np.mean(bins[pop], axis=1)
        ax = plt.subplot(2, 3, i + 1)
        ax.plot(bin_mids, ages[:, 0], ".", color=colors[0], lw=0.5, label="Paternal")
        ax.plot(bin_mids, ages[:, 1], ".", color=colors[1], lw=0.5, label="Maternal")
        xout, yout1, wout = loess_1d(bin_mids, ages[:, 0], frac=0.5, degree=2)
        ax.plot(bin_mids, yout1, c=colors[0], lw=1.0, label=None)
        xout, yout2, wout = loess_1d(bin_mids, ages[:, 1], frac=0.5, degree=2)
        ax.plot(bin_mids, yout2, c=colors[1], lw=1.0, label=None)
        xout, yout3, wout = loess_1d(bin_mids, ages.mean(axis=1), frac=0.5, degree=2)
        sex_averaged[pop] = yout3
        if i == 0:
            ax.legend()
        ax.set_title(pop)
        ax.set_xlabel("Generations ago")
        ax.set_ylabel("Average age")
        ax.set_xscale("log")
        ax.set_ylim(ylim)

    ax = plt.subplot(2, 3, 6)
    for pop in bins.keys():
        ax.plot(xout, sex_averaged[pop], label=pop, lw=1.0)

    ax.legend()
    ax.set_xlabel("Generations ago")
    ax.set_ylabel("Sex-averaged age")
    ax.set_xscale("log")
    ax.set_ylim(ylim)

    fig.tight_layout()
    if savefig:
        plt.savefig(
            f"plots/inferred_generation_times.DM.{dataset}.max_age.{max_age}.pdf"
        )


def get_predicted_spectrum_history(ages, dataset, iceland_spectrum, max_age=10000):

    anchor_spectra, bins = get_data_spectra(dataset, "ALL", max_age=max_age)
    anchor = anchor_spectra[0]
    delta = anchor / np.sum(anchor) - iceland_spectrum / iceland_spectrum.sum()

    y = np.zeros((len(ages), 6))
    for i, age_pair in enumerate(ages):
        spec = predict_spectrum(age_pair)
        y[i] = spec / np.sum(spec) + delta

    return y


def plot_predicted_histories(
    bins, predicted_histories, dataset, max_age=10000, savefig=False
):
    fig = plt.figure(4, figsize=(8, 5))
    axes = {}
    fig.clf()
    for i, pop in enumerate(bins.keys()):
        # load the data
        data_spectra, b = get_data_spectra(dataset, pop, max_age=max_age)
        if pop == "ALL":
            anchor = data_spectra[0] / np.sum(data_spectra[0])
        for j in range(len(data_spectra)):
            data_spectra[j] /= np.sum(data_spectra[j])
        hist = predicted_histories[pop]
        bin_mids = np.mean(bins[pop], axis=1)
        axes[i] = plt.subplot(2, 3, i + 1)
        # plot the smoothed data histories
        for j, c in enumerate(mut_classes):
            y = (data_spectra[:, j] - anchor[j]) * 100
            xout, yout1, wout = loess_1d(bin_mids, y, frac=0.5, degree=2)
            axes[i].plot(
                bin_mids, yout1, "-", color=colors[j], lw=1, label=c + " (data)"
            )
        # plot the smoothed econstructed histories
        for j, c in enumerate(mut_classes):
            y = (hist[:, j] - anchor[j]) * 100
            xout, yout2, wout = loess_1d(bin_mids, y, frac=0.5, degree=2)
            axes[i].plot(
                bin_mids, yout2, "--", color=colors[j], lw=2, label=c + " (model fit)"
            )
        axes[i].set_title(pop)
        axes[i].set_xlabel("Generations ago")
        axes[i].set_ylabel("Percent change")
        axes[i].set_xscale("log")

    fig.tight_layout()
    axes[4].legend(ncol=2, fontsize=6, bbox_to_anchor=(1.0, 1.0))
    if savefig:
        plt.savefig(f"plots/goodness-of-fit.DM.{dataset}.max_age.{max_age}.pdf")


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])
    (dataset, max_age) = (args.dataset, args.max_age)

    # Icelander spectrum from other infer_age_model.py
    # iceland_spectrum = np.array([ 2886, 11053,  2697,  2909,  3685,  7063])
    # Icelander spectrum from age_modeling.R
    iceland_spectrum = np.array([2739, 10408, 2529, 2702, 3484, 6613])

    # fit the generation time history to the observed spectra history
    pops = ["ALL", "AFR", "EAS", "EUR", "SAS"]
    fit_ages = {}
    bins = {}
    for pop in pops:
        bins[pop], fit_ages[pop] = fit_generation_times(
            dataset, pop, iceland_spectrum, max_age=max_age
        )

    # plot the results
    plot_inferred_generation_times(bins, fit_ages, dataset, savefig=True)

    # do we recover the input mutation spectra?
    predicted_histories = {}
    for pop in pops:
        ages = fit_ages[pop]
        predicted_histories[pop] = get_predicted_spectrum_history(
            ages, dataset, iceland_spectrum, max_age=max_age
        )
    plot_predicted_histories(bins, predicted_histories, dataset, max_age=max_age, savefig=True)

    # save histories
    with open(f"data/predicted_ages.{dataset}.{max_age}.pkl", "wb+") as fout:
        pickle.dump({"bins": bins, "ages": fit_ages}, fout)
