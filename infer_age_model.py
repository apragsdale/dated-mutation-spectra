import pandas as pd
import numpy as np
from sklearn import linear_model
import scipy.optimize
import scipy.stats
from loess.loess_1d import loess_1d
import matplotlib.pylab as plt
from bokeh.palettes import Dark2
import matplotlib

plt.rcParams["legend.title_fontsize"] = "xx-small"
matplotlib.rc("xtick", labelsize=5)
matplotlib.rc("ytick", labelsize=5)
matplotlib.rc("axes", labelsize=6)
matplotlib.rc("axes", titlesize=6)
matplotlib.rc("legend", fontsize=6)


pd.set_option("mode.chained_assignment", None)

from Bio import SeqIO

from util import *

import argparse

from bokeh.palettes import Dark2

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
    optional.add_argument(
        "--trio_mode",
        "-t",
        type=str,
        default="combined",
    )
    optional.add_argument(
        "--max_age",
        "-m",
        type=int,
        default=10000,
    )
    return parser


def get_reference_genomes(build):
    ref_genomes = {}
    for chrom in range(1, 23):
        fasta = f"./data/reference_genomes/{build}/chr{chrom}.fa.gz"
        with gzip.open(fasta, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                ref_genomes[f"chr{chrom}"] = record
    return ref_genomes


def get_triplet(chrom, pos, ref_genomes):
    # chrom is "chr1", e.g.
    triplet = str(ref_genomes[chrom][pos - 2 : pos + 1].seq)
    return triplet.upper()


nucleotides = ["A", "C", "G", "T"]

mut_classes = ["A>C", "A>G", "A>T", "C>A", "C>G", "C>T"]
complementary_classes = ["T>G", "T>C", "T>A", "G>T", "G>C", "G>A"]
class_map = {m1: m2 for m1, m2 in zip(complementary_classes, mut_classes)}


def load_and_filter_trio_data():
    fname = "./data/decode_DNMs/decode_DNMs.tsv"
    df = pd.read_csv(fname, sep="\t")
    # keep only autosomes
    df = df[df["Chr"].isin([f"chr{i}" for i in range(1, 23)])]
    # filter out indels
    df = df[df["Ref"].isin(nucleotides)]
    df = df[df["Alt"].isin(nucleotides)]
    # keep only phased data
    df = df[df["Phase_combined"].isna() == False]
    # add reference triplets
    ref_genomes = get_reference_genomes("GRCh38")
    triplets = []
    for chrom, pos in zip(df["Chr"], df["Pos_hg38"]):
        triplets.append(get_triplet(chrom, pos, ref_genomes))
    df["Ref_triplet"] = triplets
    # remove CpGs
    CpGs = ["ACG", "CCG", "GCG", "TCG", "CGA", "CGC", "CGG", "CGT"]
    df = df[df["Ref_triplet"].isin(CpGs) == False]

    # add mutation classes
    df["mut_class"] = df["Ref"] + ">" + df["Alt"]
    # add triplet mutation classes
    df["Alt_triplet"] = [
        tri[0] + alt + tri[2] for (tri, alt) in zip(df["Ref_triplet"], df["Alt"])
    ]

    # collapse mutation classes
    muts = list(df["mut_class"])
    for i, mut in enumerate(muts):
        if mut in complementary_classes:
            muts[i] = class_map[mut]
    df["mut_class"] = muts

    # remove European TC triplets and their complements
    #euro_triplets = [["TCC", "TTC"], ["ACC", "ATC"], ["TCT", "TTC"], ["CCC", "CTC"]]
    #for tri_from, tri_to in euro_triplets:
        ## NOTE: commented out the full observed triplet mutation types
        # df = df[(df["Ref_triplet"] != tri_from) ! (df["Alt_triplet"] != tri_to)]
        # df = df[
        #    (df["Ref_triplet"] != reverse_complement(tri_from))
        #    ! (df["Alt_triplet"] != reverse_complement(tri_to))
        # ]
    for tri in ["TCC", "ACC", "TCT", "CCC"]:
        # Wang et al do it a bit differently
        df = df[(df["mut_class"] != "C>T") | (df["Ref_triplet"] != tri)]
        df = df[
            (df["mut_class"] != "C>T")
            | (df["Ref_triplet"] != reverse_complement(tri))
        ]

    return df


def aggregate_spectra_combined(df):
    # df_new = pd.DataFrame(
    #    columns=["Proband_nr", "Fathers_age_at_conception", "Mothers_age_at_conception"]
    #    + mut_classes
    # )
    data = {}
    proband_nrs = set(df["Proband_nr"])
    for proband_nr in proband_nrs:
        proband_data = {}
        proband_data["Proband_nr"] = proband_nr
        df2 = df[df["Proband_nr"] == proband_nr]
        total_mutations = len(df2)
        if total_mutations < 10:
            continue
        proband_data["Fathers_age_at_conception"] = list(
            df2["Fathers_age_at_conception"]
        )[0]
        proband_data["Mothers_age_at_conception"] = list(
            df2["Mothers_age_at_conception"]
        )[0]
        for c in mut_classes:
            num_muts = sum(df2["mut_class"] == c)
            proband_data[c] = num_muts
        data[proband_nr] = proband_data
    df_new = pd.DataFrame.from_dict(data, orient="index")
    return df_new


def basic_linear_regression_combined(df_new):
    x = df_new[["Fathers_age_at_conception", "Mothers_age_at_conception"]]
    y = df_new[mut_classes]

    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    return regr


def predict_spectrum_combined(ages, regr):
    return regr.intercept_ + ages[0] * regr.coef_[:, 0] + ages[1] * regr.coef_[:, 1]


def aggregate_spectra_phased(df):
    # separate out the phased data into mutations that come from father and mother
    proband_nrs = set(df["Proband_nr"])
    father_data = {}
    mother_data = {}
    for proband_nr in proband_nrs:
        proband_data_f = {}
        proband_data_m = {}
        df_prob = df[df["Proband_nr"] == proband_nr]
        df_f = df_prob[df_prob["Phase_combined"] == "father"]
        df_m = df_prob[df_prob["Phase_combined"] == "mother"]
        total_mutations = len(df_prob)
        # filter by a minimum number of mutations
        if total_mutations < 10:
            continue
        proband_data_f["Fathers_age_at_conception"] = list(
            df_prob["Fathers_age_at_conception"]
        )[0]
        proband_data_m["Mothers_age_at_conception"] = list(
            df_prob["Mothers_age_at_conception"]
        )[0]
        for c in mut_classes:
            num_muts = sum(df_f["mut_class"] == c)
            proband_data_f[c] = num_muts
            num_muts = sum(df_m["mut_class"] == c)
            proband_data_m[c] = num_muts
        father_data[proband_nr] = proband_data_f
        mother_data[proband_nr] = proband_data_m
    df_father = pd.DataFrame.from_dict(father_data, orient="index")
    df_mother = pd.DataFrame.from_dict(mother_data, orient="index")
    return df_father, df_mother


def basic_linear_regression_phased(df_father, df_mother):
    x = df_father[["Fathers_age_at_conception"]]
    y = df_father[mut_classes]

    regr_father = linear_model.LinearRegression()
    regr_father.fit(x, y)

    x = df_mother[["Mothers_age_at_conception"]]
    y = df_mother[mut_classes]

    regr_mother = linear_model.LinearRegression()
    regr_mother.fit(x, y)

    return (regr_father, regr_mother)


def predict_spectrum_phased(ages, regr):
    regr_father, regr_mother = regr
    return (
        regr_father.intercept_
        + regr_mother.intercept_
        + ages[0] * regr_father.coef_.flatten()
        + ages[1] * regr_mother.coef_.flatten()
    )


def predict_spectrum(ages):
    """
    This comes from the `age_modeling.R` script from Wang et al.
    """
    alpha = np.array([13.830321, 15.180457, 14.056053, 13.923672, 13.952551, 14.947698])
    beta0 = np.array([-0.316633, -0.327940, -0.322887, -0.329628, -0.321475, -0.326378])
    beta1 = np.array([0.252819, 0.265539, 0.249886, 0.264401, 0.262430, 0.256306])
    p = np.exp(alpha + ages[0] * beta0 + ages[1] * beta1)
    return p / np.sum(p)


def get_mutation_spectrum(df):
    return np.array(
        [
            sum(df["mut_class"] == "A>C"),
            sum(df["mut_class"] == "A>G"),
            sum(df["mut_class"] == "A>T"),
            sum(df["mut_class"] == "C>A"),
            sum(df["mut_class"] == "C>G"),
            sum(df["mut_class"] == "C>T"),
        ]
    )


def clr(x):
    geom_mean = np.prod(x) ** (1 / len(x))
    return np.log(x / geom_mean)


def cost_func(ages, data, model_func, bin_idx, regr, iceland_spectrum, anchor):
    predicted = model_func(ages, regr)
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


def fit_generation_times(
    dataset, pop, trio_mode, regr, iceland_spectrum, max_age=10000
):
    data_spectra, bins = get_data_spectra(dataset, pop, max_age=max_age)
    anchor_spectra, bins = get_data_spectra(dataset, "ALL", max_age=max_age)
    anchor = anchor_spectra[0]
    fit_ages = np.zeros((100, 2))
    for bin_idx in range(100):
        p0 = [30, 30]

        if trio_mode == "combined":
            model_func = predict_spectrum_combined
        elif trio_mode == "phased":
            model_func = predict_spectrum_phased

        args = (
            data_spectra,
            model_func,
            bin_idx,
            regr,
            iceland_spectrum,
            anchor,
        )

        ret = scipy.optimize.fmin_l_bfgs_b(
            cost_func, p0, args=args, approx_grad=True, bounds=[[14, 60], [14, 60]]
        )
        fit_ages[bin_idx] = ret[0]
    return bins, fit_ages


def plot_inferred_generation_times(bins, fit_ages, dataset):
    fig = plt.figure(3, figsize=(8, 5))
    fig.clf()
    ylim = (14, 50)
    sex_averaged = {}
    for i, pop in enumerate(bins.keys()):
        ages = fit_ages[pop]
        bin_mids = np.mean(bins[pop], axis=1)
        ax = plt.subplot(2, 3, i + 1)
        ax.plot(bin_mids, ages[:, 0], "b.--", lw=0.5, label="Paternal")
        ax.plot(bin_mids, ages[:, 1], "r.--", lw=0.5, label="Maternal")
        xout, yout1, wout = loess_1d(bin_mids, ages[:, 0], frac=0.5, degree=2)
        ax.plot(bin_mids, yout1, c="b", lw=1.0, label=None)
        xout, yout2, wout = loess_1d(bin_mids, ages[:, 1], frac=0.5, degree=2)
        ax.plot(bin_mids, yout2, c="r", lw=1.0, label=None)
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
    plt.savefig(f"plots/inferred_generation_times.{dataset}.max_age.{max_age}.pdf")


def get_predicted_spectrum_history(
    ages, regr, trio_mode, dataset, iceland_spectrum, max_age=10000
):
    if trio_mode == "combined":
        model_func = predict_spectrum_combined
    elif trio_mode == "phased":
        model_func = predict_spectrum_phased

    anchor_spectra, bins = get_data_spectra(dataset, "ALL", max_age=max_age)
    anchor = anchor_spectra[0]
    delta = anchor / np.sum(anchor) - iceland_spectrum / iceland_spectrum.sum()

    y = np.zeros((len(ages), 6))
    for i, age_pair in enumerate(ages):
        spec = model_func(age_pair, regr)
        y[i] = spec / np.sum(spec) + delta

    return y


def plot_predicted_histories(bins, predicted_histories, dataset, max_age=10000):
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
            y = (data_spectra[:, j] - anchor[j]) / 100
            xout, yout1, wout = loess_1d(bin_mids, y, frac=0.5, degree=2)
            axes[i].plot(
                bin_mids, yout1, ":", color=colors[j], lw=2, label=c + " (data)"
            )
        # plot the smoothed econstructed histories
        for j, c in enumerate(mut_classes):
            y = (hist[:, j] - anchor[j]) / 100
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
    plt.savefig(f"plots/goodness-of-fit.{dataset}.max_age.{max_age}.pdf")


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])
    (dataset, trio_mode, max_age) = (args.dataset, args.trio_mode, args.max_age)

    # load and filter trio data
    df = load_and_filter_trio_data()
    iceland_spectrum = get_mutation_spectrum(df)
    eprint("Iceland mutation spectrum:")
    eprint(iceland_spectrum / iceland_spectrum.sum())

    # fit the mutation model
    if trio_mode == "phased":
        df_father, df_mother = aggregate_spectra_phased(df)
        regr = basic_linear_regression_phased(df_father, df_mother)
        (regr_father, regr_mother) = regr
        eprint("predicted spectrum for average Icelander ages (32, 28.2):")
        y = predict_spectrum_phased((32, 28.2), (regr_father, regr_mother))
        eprint(y / y.sum())
    elif trio_mode == "combined":
        df_new = aggregate_spectra_combined(df)
        regr = basic_linear_regression_combined(df_new)
        eprint("predicted spectrum for average Icelander ages (32, 28.2):")
        y = predict_spectrum_combined((32, 28.2), regr)
        eprint(y / y.sum())

    # fit the generation time history to the observed spectra history
    pops = ["ALL", "AFR", "EAS", "EUR", "SAS"]
    fit_ages = {}
    bins = {}
    for pop in pops:
        bins[pop], fit_ages[pop] = fit_generation_times(
            dataset, pop, trio_mode, regr, iceland_spectrum, max_age=max_age
        )

    # plot the results
    plot_inferred_generation_times(bins, fit_ages, dataset)

    # do we recover the input mutation spectra?
    predicted_histories = {}
    for pop in pops:
        ages = fit_ages[pop]
        predicted_histories[pop] = get_predicted_spectrum_history(
            ages, regr, trio_mode, dataset, iceland_spectrum, max_age=max_age
        )
    plot_predicted_histories(bins, predicted_histories, dataset, max_age=max_age)
