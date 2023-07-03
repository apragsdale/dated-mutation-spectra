"""
This script processes the saved dataframes, which have columns:
    ['Chr', 'Pos', 'ID', 'Age', 'Ref', 'Alt', 'Anc', 'Der', 'Tri', 'AC',
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
        df = pd.read_pickle(f"./data/{dataset}/parsed_variants.chr{chrom}.gzip")
    except IOError:
        return
    # filter by age
    df = df[df["Age"] < max_age]
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
            df = pd.read_pickle(f"./data/{dataset}/parsed_variants.chr{chrom}.gzip")
        except IOError:
            eprint(current_time(), "no data for chromosome", chrom)
            continue

        ages[chrom] = np.array(df["Age"])
        ages[chrom] = ages[chrom][ages[chrom] < max_age]
    all_ages = np.concatenate(list(ages.values()))
    all_ages = np.sort(all_ages)
    bin_size = int(len(all_ages) / num_bins)
    bin_edges = []
    for i in range(num_bins):
        bin_edges.append(all_ages[bin_size * i])
    bin_edges.append(max_age)
    return np.array(bin_edges)


def subset_to_bin(df, bin_min, bin_max):
    df_bin = df[df["Age"] >= bin_min]
    df_bin = df_bin[df_bin["Age"] < bin_max]
    return df_bin


def count_mutation_types(df):
    class_counts = {c: sum(df["Mut"] == c) for c in classes}
    return class_counts


def get_windows(df, bs_size):
    min_pos = min(df["Pos"])
    windows = [(min_pos // bs_size) * bs_size]
    if min_pos % bs_size > bs_size / 2:
        windows.append(windows[-1] + 2 * bs_size)
    else:
        windows.append(windows[-1] + bs_size)
    max_pos = max(df["Pos"])
    while windows[-1] < max_pos:
        windows.append(windows[-1] + bs_size)
    if windows[-1] - max_pos > bs_size / 2:
        windows.pop(windows.index(windows[-2]))
    return windows

def parse_variant_data_bs(
    dataset, max_age, keep_singletons, max_frequency, num_bins=100, bs_size=5000000
):
    bin_edges = get_bin_edges(dataset, max_age, num_bins)
    eprint(current_time(), "set up bin edges")
    # gather data across chromosomes
    all_data = {} # keys: (chrom, left, right)
    for chrom in range(1, 23):
        eprint(current_time(), "parsing chromosome", chrom)
        # load chromosome data
        df = read_chromosome(chrom, dataset, max_age, keep_singletons, max_frequency)
        if df is None:
            continue
        windows = get_windows(df, bs_size)
        eprint(current_time(), f"have {len(windows) - 1} windows")
        for win_left, win_right in zip(windows[:-1], windows[1:]):
            df_window = df[(df["Pos"] >= win_left) * (df["Pos"] < win_right)]
            pop_data = {p: {i: defaultdict(int) for i in range(num_bins)} for p in pops}
            for i, (bin_min, bin_max) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                # for each bin, subset data to that bin
                df_bin = subset_to_bin(df_window, bin_min, bin_max)
                for pop in pops:
                    # for each population, keep data segregating in that population
                    df_pop = df_bin[df_bin[pop] > 0]
                    class_counts = count_mutation_types(df_pop)
                    # add those counts to our data dict
                    for c, v in class_counts.items():
                        pop_data[pop][i][c] += v
            all_data[(chrom, win_left, win_right)] = pop_data
            eprint(current_time(), "processed window", win_left, win_right)
        eprint(current_time(), "parsed chromosome", chrom)

    all_dfs = {}
    for k in all_data.keys():
        all_pop_counts = {
            pop: {
                i: {"Min": bin_edges[i], "Max": bin_edges[i + 1]} for i in range(num_bins)
            }
            for pop in pops
        }
        for pop in pops:
            for i in range(num_bins):
                all_pop_counts[pop][i].update(all_data[k][pop][i])
        pop_dfs = {}
        for pop in pops:
            pop_dfs[pop] = pd.DataFrame.from_dict(all_pop_counts[pop], orient="index")
        all_dfs[k] = pop_dfs
    return all_dfs


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

    all_dfs = parse_variant_data_bs(
        dataset, max_age, keep_singletons, max_frequency, num_bins=num_bins
    )
    with gzip.open("bootstrap_dfs.pkl.gz", "wb+") as fout:
        pickle.dump(all_dfs, fout)

    eprint(current_time(), "saved data!!")
