"""
The output of this script is a pandas dataframe, storing information about
positions, ancestral/derived mutations, allele frequencies, its triplet
context, variant IDs (rsid), and age.

This script, along with process_tsdate.py and process_relate.py, will create
dataframes with consistent column names and information. Parsed data is saved
as a pickled pandas df in `./data/geva/parsed_variants.chr[chrom].gzip` and can
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
import sys
import argparse

from util import *

nucleotides = ["A", "C", "G", "T"]


def make_parser():
    ADHF = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser("process_geva.py", formatter_class=ADHF)
    optional = parser.add_argument_group("Optional")
    optional.add_argument(
        "--chrom",
        "-c",
        type=int,
        default=0,
        help="The chromosome to parse. If None given, we parse them all.",
    )
    return parser


def subset_geva(chrom, data_source="TGP"):
    """
    Returns a pandas dataframe with relevant info from the GEVA Atlas file
    for the given chromosome.

    We need columns
    VariantID, Chromosome, Position, AlleleRef, AlleleAlt, AlleleAnc, AgeMedian_Jnt

    Returned columns have:
    Chr, Pos, ID, Ref, Alt, Anc, Der, Age

    age is taken to be AgeMedian_Jnt
    """
    geva_fname = f"./data/geva/atlas.chr{chrom}.csv.gz"
    df = pd.read_csv(geva_fname, sep=", ", engine="python", header=3, index_col=False)
    # we're using TGP data, so use DataSoure = TGP
    df = df[df["DataSource"] == data_source]
    # only keep sites thare have ancestral site information
    df = df[df["AlleleAnc"].isin(nucleotides)]
    df = df[
        [
            "Chromosome",
            "Position",
            "VariantID",
            "AgeCI95Lower_Jnt",
            "AgeCI95Upper_Jnt",
            "AlleleRef",
            "AlleleAlt",
            "AlleleAnc",
        ]
    ]
    # remove sites where the ancestral allele
    # doesn't match the reference or derived allele
    df = df[(df["AlleleAnc"] == df["AlleleRef"]) | (df["AlleleAnc"] == df["AlleleAlt"])]
    # set the derived allele
    df["AlleleDer"] = np.where(
        df["AlleleAnc"] == df["AlleleRef"], df["AlleleAlt"], df["AlleleRef"]
    )
    # rename columns
    col_names = {
        "Chromosome": "Chr",
        "Position": "Pos",
        "VariantID": "ID",
        "AgeCI95Lower_Jnt": "AgeLower",
        "AgeCI95Upper_Jnt": "AgeUpper",
        "AlleleRef": "Ref",
        "AlleleAlt": "Alt",
        "AlleleAnc": "Anc",
        "AlleleDer": "Der",
    }
    df = df.rename(columns=col_names)
    df = df.reset_index(drop=True)
    return df


def get_allele_frequencies(df, chrom):
    """
    Allele frequencies are listed in the INFO field.

    Note: AFR populations include ACB and ASW, which are admixed African
    Americans and Afro-Caribbean. We have kept them here, to match the Wang et
    al study but note that it may be preferable to disclude them.
    """
    num_samples = 2 * 2504
    vcf_fname = f"./data/1000G/ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5.20130502.sites.annotation.vcf.gz"
    d = df[["Pos", "Anc", "Der"]].set_index("Pos").to_dict("index")
    with gzip.open(vcf_fname, "rb") as fin:
        for line in fin:
            l = line.decode()
            if l.startswith("#"):
                continue
            (_, pos, rsid, ref, alt, _, _, info) = l.split()
            pos = int(pos)
            if pos not in d.keys():
                continue
            # make sure we're dealing with nucleotides
            if ref not in nucleotides or alt not in nucleotides:
                continue
            # check that anc and der alleles match ref and alt from 1kg
            if ref != d[pos]["Anc"] and ref != d[pos]["Der"]:
                continue
            if alt != d[pos]["Anc"] and alt != d[pos]["Der"]:
                continue
            # if everything looks good, parse the info and get allele freqs
            flip = False
            if d[pos]["Anc"] == alt:
                flip = True
            elif d[pos]["Anc"] != ref:
                # ancestral allel does't match alt or ref
                continue
            infodict = parse_vcf_info(info)
            d[pos]["AltID"] = rsid
            if flip:
                d[pos]["AC"] = num_samples - int(infodict["AC"])
                d[pos]["ALL"] = 1 - float(infodict["AF"])
                d[pos]["AFR"] = 1 - float(infodict["AFR_AF"])
                d[pos]["EAS"] = 1 - float(infodict["EAS_AF"])
                d[pos]["EUR"] = 1 - float(infodict["EUR_AF"])
                d[pos]["SAS"] = 1 - float(infodict["SAS_AF"])
            else:
                d[pos]["AC"] = int(infodict["AC"])
                d[pos]["ALL"] = float(infodict["AF"])
                d[pos]["AFR"] = float(infodict["AFR_AF"])
                d[pos]["EAS"] = float(infodict["EAS_AF"])
                d[pos]["EUR"] = float(infodict["EUR_AF"])
                d[pos]["SAS"] = float(infodict["SAS_AF"])

    df["AC"] = [d[pos]["AC"] for pos in df["Pos"]]
    df["ALL"] = [d[pos]["ALL"] for pos in df["Pos"]]
    df["AFR"] = [d[pos]["AFR"] for pos in df["Pos"]]
    df["EAS"] = [d[pos]["EAS"] for pos in df["Pos"]]
    df["EUR"] = [d[pos]["EUR"] for pos in df["Pos"]]
    df["SAS"] = [d[pos]["SAS"] for pos in df["Pos"]]

    df["ID"] = [
        id1 + ";" + id2 if id1 != id2 else id1
        for id1, id2 in zip(df["ID"], [d[pos]["AltID"] for pos in df["Pos"]])
    ]
    return df


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])
    if args.chrom == 0:
        chroms = range(1, 23)
    else:
        chroms = [args.chrom]

    # NOTE: GEVA provides multiple estimates for some alleles, whether
    # estimates are available from TPG, SGDP, or both (in which case a
    # "Combined" estimate is give). It appears that mutation spectra
    # over time are sensitive to which of the estimates you pick, but
    # this should be checked
    data_source = "TGP"

    for chrom in chroms:
        eprint("Parsing chromosome", chrom)
        eprint(current_time())
        # process geva file
        df = subset_geva(chrom, data_source=data_source)
        eprint(current_time(), "subset GEVA data")
        # add reference triplet from reference genome
        df = append_reference_triplet(df, chrom, "GRCh37")
        eprint(current_time(), "determined reference triplets")
        # add allele frequencies and allele count from 1KG data
        df = get_allele_frequencies(df, chrom)
        eprint(current_time(), "extracted allele frequencies")
        # save data - if we used a data source other than TGP, we
        # include that in the filename
        if data_source != "TGP":
            f"./data/geva/parsed_variants.{data_source}.chr{chrom}.upper_lower.gzip"
        else:
            fname = f"./data/geva/parsed_variants.chr{chrom}.upper_lower.gzip"
        df.to_pickle(fname)
        eprint(current_time(), "saved data!!")
