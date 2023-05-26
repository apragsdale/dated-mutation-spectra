"""
This script processes the saved dataframes, which have columns:
    ['Chr', 'Pos', 'ID', 'AgeLower', 'AgeUpper', 'Ref', 'Alt', 'Anc', 'Der', 'Tri', 'AC',
       'ALL', 'AFR', 'EAS', 'EUR', 'SAS']

From this we get variant counts over time bins, for each of the mutation
types (and their complements).

The output is a csv with columns:
    ["MinAge", "MaxAge", "A>C", "A>G", "A>T", "C>A", "C>G", "C>T",
        "CpG", "ACC>ATC", "CCC>CTC", "TCC>TTC", "AGA>AAA"]

Usage:
python bin_variant_ages.py
    [required] -d [dataset]
    [optional, defaults to 10000] -m [max age]
    [optional, defaults to False] -s
    [optional, defaults to 0.98] -f [max frequency]

The dataset can be chosen from "geva", "tsdate", or "relate". The singletons
flag will include singletons if given, and will exclude them if left out.
The max frequency flag filters sites with total allele frequency greater
that the given value.
"""

import gzip
import sys, os
import numpy as np
import argparse
from collections import defaultdict

from util import *


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
    return parser


pops = ["ALL", "AFR", "EAS", "EUR", "SAS"]


classes = [
    "A>C",
    "A>G",
    "A>T",
    "C>A",
    "C>G",
    "C>T",
    "CpG",
    "ACC>ATC",
    "CCC>CTC",
    "TCC>TTC",
    "TCT>TTT",
]


def add_mut_class(df):
    mut_classes = []
    for triplet, anc, der in zip(df["Tri"], df["Anc"], df["Der"]):
        left = triplet[0]
        right = triplet[2]
        trip_from = "".join([left, anc, right])
        trip_to = "".join([left, der, right])
        if trip_from[1] not in ["A", "C"]:
            trip_from = reverse_complement(trip_from)
            trip_to = reverse_complement(trip_to)
        if "CG" in trip_from:
            mut_classes.append("CpG")
        elif trip_from + ">" + trip_to in classes:
            mut_classes.append(trip_from + ">" + trip_to)
        else:
            mut_classes.append(trip_from[1] + ">" + trip_to[1])
    df["Mut"] = mut_classes
    return df


def read_chromosome(chrom, dataset, max_age, keep_singletons, max_frequency):
    try:
        df = pd.read_pickle(
            f"./data/{dataset}/parsed_variants.chr{chrom}.upper_lower.gzip"
        )
    except IOError:
        return
    # filter by age
    df = df[df["AgeLower"] < max_age]
    # remove singletons
    if not keep_singletons:
        df = df[df["AC"] > 1]
    # filter by max frequency
    df = df[df["ALL"] <= max_frequency]
    # add mutation class column
    df = add_mut_class(df)
    return df


def get_bin_edges(dataset, max_age, num_bins=100):
    assert num_bins > 0, "num_bins must be positive integer"
    ages = {}
    for chrom in range(1, 23):
        try:
            df = pd.read_pickle(
                f"./data/{dataset}/parsed_variants.chr{chrom}.upper_lower.gzip"
            )
        except IOError:
            eprint(current_time(), "no data for chromosome", chrom)
            continue
        # bins based on expected ages
        ages[chrom] = (np.array(df["AgeLower"]) + np.array(df["AgeUpper"])) / 2
        ages[chrom] = ages[chrom][ages[chrom] < max_age]
    all_ages = np.concatenate(list(ages.values()))
    all_ages = np.sort(all_ages)
    bin_size = int(len(all_ages) / num_bins)
    bin_edges = []
    for i in range(num_bins):
        bin_edges.append(all_ages[bin_size * i])
    bin_edges[0] = 0
    bin_edges.append(max_age)
    return np.array(bin_edges)


def subset_to_bin(df, bin_min, bin_max):
    # all variants with CIs or branches overlapping with the bin
    df_bin = df[df["AgeUpper"] > bin_min]
    df_bin = df_bin[df_bin["AgeLower"] < bin_max]
    return df_bin


def get_overlap_proportions(df, bin_min, bin_max):
    branch_lengths = df["AgeUpper"] - df["AgeLower"]
    overlap_lengths = np.array(
        [min(bin_max, u) for u in df["AgeUpper"]]
    ) - np.array([max(bin_min, l) for l in df["AgeLower"]])
    return overlap_lengths / branch_lengths


def count_mutation_types(df, bin_min, bin_max):
    props = get_overlap_proportions(df, bin_min, bin_max)
    assert max(props) <= 1
    assert min(props) > 0
    class_counts = {c: sum((df["Mut"] == c) * props) for c in classes}
    return class_counts


def parse_variant_data(dataset, max_age, keep_singletons, max_frequency, num_bins=100):
    ## this isn't the fastest way to do this, but I was running into memory
    ## issues trying to load all data across chromosomes at once into one df
    # set up age bins
    bin_edges = get_bin_edges(dataset, max_age, num_bins)
    eprint(current_time(), "set up bin edges")
    # gather data across chromosomes
    pop_data = {p: {i: defaultdict(int) for i in range(num_bins)} for p in pops}
    for chrom in range(1, 23):
        # load chromosome data
        df = read_chromosome(chrom, dataset, max_age, keep_singletons, max_frequency)
        if df is None:
            continue
        for i, (bin_min, bin_max) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            # for each bin, subset data to that bin
            df_bin = subset_to_bin(df, bin_min, bin_max)
            for pop in pops:
                # for each population, keep data segregating in that population
                df_pop = df_bin[df_bin[pop] > 0]
                class_counts = count_mutation_types(df_pop, bin_min, bin_max)
                # add those counts to our data dict
                for c, v in class_counts.items():
                    pop_data[pop][i][c] += v
        eprint(current_time(), "parsed chromosome", chrom)

    all_pop_counts = {
        pop: {
            i: {"Min": bin_edges[i], "Max": bin_edges[i + 1]} for i in range(num_bins)
        }
        for pop in pops
    }
    for pop in pops:
        for i in range(num_bins):
            all_pop_counts[pop][i].update(pop_data[pop][i])
    pop_dfs = {}
    for pop in pops:
        pop_dfs[pop] = pd.DataFrame.from_dict(all_pop_counts[pop], orient="index")
    return pop_dfs


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])
    (dataset, max_age, keep_singletons, max_frequency, num_bins) = (
        args.dataset,
        args.max_age,
        args.singletons,
        args.frequency,
        args.num_bins,
    )

    if keep_singletons is True and dataset == "geva":
        raise ValueError("No singletons in GEVA!")

    pop_dfs = parse_variant_data(
        dataset, max_age, keep_singletons, max_frequency, num_bins=num_bins
    )
    for pop in pops:
        if num_bins == 100:
            fname = f"./data/binned_ages.{dataset}.{pop}.max_age.{int(max_age)}.singletons.{keep_singletons}.max_frequency.{max_frequency}.CI.csv"
        else:
            fname = f"./data/binned_ages.{dataset}.{pop}.max_age.{int(max_age)}.singletons.{keep_singletons}.max_frequency.{max_frequency}.num_bins.{int(num_bins)}.CI.csv"
        pop_dfs[pop].to_csv(fname, sep="\t", index=False)
    eprint(current_time(), "saved data!!")
