"""
The output of this script is a pandas dataframe, storing information about
positions, ancestral/derived mutations, allele frequencies, its triplet
context, variant IDs (rsid), and age.

This script, along with process_tsdate.py and process_geva.py, will create
dataframes with consistent column names and information. Parsed data is saved
as a pickled pandas df in `./data/relate/parsed_variants.chr[chrom].gzip` and can
be loaded as `df = pd.read_pickle(fname)`.

output columns:
    ['Chr', 'Pos', 'ID', 'Age', 'Ref', 'Alt', 'Anc', 'Der', 'Tri', 'AC',
       'ALL', 'AFR', 'EAS', 'EUR', 'SAS']
"""

import gzip
from Bio import SeqIO
import sys
import pandas as pd
import numpy as np
import sys, os
import argparse
from collections import defaultdict

from util import *

nucleotides = ["A", "C", "G", "T"]


def make_parser():
    ADHF = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser("process_relate.py", formatter_class=ADHF)
    optional = parser.add_argument_group("Optional")
    optional.add_argument(
        "--chrom",
        "-c",
        type=int,
        default=0,
        help="The chromosome to parse. If None given, we parse them all.",
    )
    return parser


pops = {
    "AFR": ["ACB", "ASW", "ESN", "GWD", "LWK", "MSL", "YRI"],
    "EAS": ["CDX", "CHB", "CHS", "JPT", "KHV"],
    "EUR": ["CEU", "FIN", "GBR", "IBS", "TSI"],
    "SAS": ["BEB", "GIH", "ITU", "PJL", "STU"],
}

sample_sizes = {"ALL": 2157, "AFR": 661, "EAS": 503, "EUR": 504, "SAS": 489}


# only run once.. I find it more manageable to work on each chrom separately
def split_data_by_chrom():
    for sup, ps in pops.items():
        for p in pops[sup]:
            fpath = f"./data/relate/allele_ages_{sup}/"
            fname_in = f"allele_ages_{p}.csv.gz"
            curr_chrom = 0
            eprint(current_time(), "splitting population", p)
            with gzip.open(fpath + fname_in, "rb") as fin:
                for line in fin:
                    if line.decode().startswith('"CHR"'):
                        header = line
                    else:
                        if int(line.decode().split(",")[0]) != curr_chrom:
                            if curr_chrom != 0:
                                fout.close()
                            if curr_chrom == 22:
                                break
                            curr_chrom += 1
                            eprint(current_time(), "starting chrom", curr_chrom)
                            fout = gzip.open(
                                fpath + f"allele_ages_{p}.chr{curr_chrom}.csv.gz", "wb+"
                            )
                            fout.write(header)
                        fout.write(line)
            fout.close()


def subset_relate(chrom):
    """
    Returns a pandas dataframe with relevant info from the GEVA Atlas file
    for the given chromosome.

    Returned columns have:
    Chr, Pos, ID, Ref, Alt, Anc, Der, Age, AC, ALL, AFR, EAS, EUR, SAS, Tri

    The age is computed as the arithmetic mean of the average of lower and upper
    ages from all populations that carry this mutation.
    """
    # ages[pos] = [[lower, upper]]
    ages = defaultdict(list)
    # allele_counts[super_pop][pos] = sum over pops in superpop
    allele_counts = {sup: defaultdict(int) for sup in pops.keys()}
    site_info = {}
    for sup, ps in pops.items():
        for p in ps:
            eprint(current_time(), "reading data from", p, "in", sup)
            fpath = f"./data/relate/allele_ages_{sup}/allele_ages_{p}.chr{chrom}.csv.gz"
            with gzip.open(fpath, "rb") as fin:
                for line in fin:
                    (
                        c,
                        pos,
                        ID,
                        lower,
                        upper,
                        ancder,
                        upstream,
                        downstream,
                        daf,
                        pval,
                    ) = line.decode().split(",")
                    if c == '"CHR"':
                        continue
                    pos = int(pos)
                    ages[pos].append([float(lower), float(upper)])
                    allele_counts[sup][pos] += int(daf)
                    anc = ancder[1]
                    der = ancder[3]
                    if pos not in site_info:
                        site_info[pos] = {
                            "Chr": chrom,
                            "Pos": pos,
                            "ID": ID[1:-1],
                            "Ref": ".",
                            "Alt": ".",
                            "Anc": anc,
                            "Der": der,
                            "Tri": upstream[1] + anc + downstream[1],
                        }
                    else:
                        assert anc == site_info[pos]["Anc"]
                        assert der == site_info[pos]["Der"]

    df = pd.DataFrame.from_dict(site_info, orient="index")
    df = df.sort_values(by="Pos")

    # add ages
    age_lower_col = np.zeros(len(df))
    age_upper_col = np.zeros(len(df))
    eprint(current_time(), "getting lower and upper ages")
    for i, pos in enumerate(df["Pos"]):
        age_lower_col[i] = np.mean([a[0] for a in ages[pos]])
        age_upper_col[i] = np.mean([a[1] for a in ages[pos]])
    df["AgeLower"] = age_lower_col
    df["AgeUpper"] = age_upper_col
    eprint(current_time(), "appended branch ages")
    
    eprint(current_time(), "computed average ages, getting frequencies")
    # add allele frequencies
    ALL = np.zeros(len(df))
    AFR = np.zeros(len(df))
    EAS = np.zeros(len(df))
    EUR = np.zeros(len(df))
    SAS = np.zeros(len(df))
    AC = np.zeros(len(df), dtype=int)
    for i, pos in enumerate(df["Pos"]):
        AC[i] = sum([allele_counts[p][pos] for p in pops.keys()])
        ALL[i] = AC[i] / sample_sizes["ALL"]
        AFR[i] = allele_counts["AFR"][pos] / sample_sizes["AFR"]
        EAS[i] = allele_counts["EAS"][pos] / sample_sizes["EAS"]
        EUR[i] = allele_counts["EUR"][pos] / sample_sizes["EUR"]
        SAS[i] = allele_counts["SAS"][pos] / sample_sizes["SAS"]

    df["AC"] = AC
    df["ALL"] = ALL
    df["AFR"] = AFR
    df["EAS"] = EAS
    df["EUR"] = EUR
    df["SAS"] = SAS

    # reorder columns and reset index
    df = df[
        [
            "Chr",
            "Pos",
            "ID",
            "Ref",
            "Alt",
            "Anc",
            "Der",
            "AgeLower",
            "AgeUpper",
            "AC",
            "ALL",
            "AFR",
            "EAS",
            "EUR",
            "SAS",
            "Tri",
        ]
    ]
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])
    if args.chrom == 0:
        chroms = range(1, 23)
    else:
        chroms = [args.chrom]

    for chrom in chroms:
        # I'm impatient.. do the small ones first
        eprint(current_time(), "Parsing chromosome", chrom)
        # process geva file
        df = subset_relate(chrom)
        eprint(current_time(), "subset relate data")
        df.to_pickle(f"./data/relate/parsed_variants.chr{chrom}.upper_lower.gzip")
        eprint(current_time(), "saved data!!")
