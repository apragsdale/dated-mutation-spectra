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


def combine_chromosomes(dataset, max_age, keep_singletons, max_frequency):
    # load each chromosome data, filter by age, singletons, and max frequency
    dfs = {}
    # then concatenate all
    for chrom in range(1, 23):
        try:
            df = pd.read_pickle(f"./data/{dataset}/parsed_variants.chr{chrom}.gzip")
        except IOError:
            eprint(current_time(), "no data for chromosome", chrom)
            continue
        # filter by age
        df = df[df["Age"] < max_age]
        # remove singletons
        if not keep_singletons:
            df = df[df["AC"] > 1]
        # filter by max frequency
        df = df[df["ALL"] <= max_frequency]
        # add mutation class column
        df = add_mut_class(df)
        dfs[chrom] = df
    df = pd.concat(list(dfs.values()), ignore_index=True)
    return df


def get_bin_edges(df):
    # we always use 100 bins.. this could be changed if we need to
    ages = df["Age"]
    ages = np.sort(ages)
    bin_size = int(len(ages) / 100)
    bin_edges = []
    for i in range(100):
        bin_edges.append(ages[bin_size * i])
    bin_edges.append(max_age)
    return np.array(bin_edges)


def subset_to_bin(df, bin_min, bin_max):
    df_bin = df[df["Age"] >= bin_min]
    df_bin = df_bin[df_bin["Age"] < bin_max]
    return df_bin


def count_mutation_types(df):
    class_counts = {c: sum(df["Mut"] == c) for c in classes}
    return class_counts


def parse_variant_data(datset, max_age, keep_singletons, max_frequency):
    # gather data across chromosomes
    df = combine_chromosomes(dataset, max_age, keep_singletons, max_frequency)
    eprint(current_time(), "combined chromosome data")
    # set up age bins
    bin_edges = get_bin_edges(df)
    # filter to mutations found in the specified pop
    pop_data = {}
    for pop in pops:
        df_pop = df[df[pop] > 0]
        # subset to ages in each bin and count mutations
        age_counts = {}
        for i, (bin_min, bin_max) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            df_bin = subset_to_bin(df_pop, bin_min, bin_max)
            class_counts = {"Min": bin_min, "Max": bin_max}
            class_counts.update(count_mutation_types(df_bin))
            age_counts[i] = class_counts
        pop_data[pop] = pd.DataFrame.from_dict(age_counts, orient="index")
        eprint(current_time(), "summarized mutation spectra in", pop)
    return pop_data


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])
    (dataset, max_age, keep_singletons, max_frequency) = (
        args.dataset,
        args.max_age,
        args.singletons,
        args.frequency,
    )
    pop_data = parse_variant_data(dataset, max_age, keep_singletons, max_frequency)
    for pop in pops:
        fname = f"./data/binned_ages.{dataset}.{pop}.max_age.{max_age}.singletons.{keep_singletons}.max_frequency.{max_frequency}.csv"
        pop_data[pop].to_csv(fname, sep="\t", index=False)
    eprint(current_time(), "saved data!!")
