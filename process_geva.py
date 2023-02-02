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
            "AgeMedian_Jnt",
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
        "AgeMedian_Jnt": "Age",
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
    vcf_fname = f"./data/1000G/ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
    # set up allele frequency vectors
    ALL = np.zeros(len(df))
    AFR = np.zeros(len(df))
    EAS = np.zeros(len(df))
    EUR = np.zeros(len(df))
    SAS = np.zeros(len(df))
    AC = np.zeros(len(df))
    positions = np.array(list(df["Pos"]))
    ancestral = list(df["Anc"])
    derived = list(df["Der"])
    IDs = list(df["ID"])
    with gzip.open(vcf_fname, "rb") as fin:
        for line in fin:
            l = line.decode()
            if l.startswith("#"):
                if l.startswith("#CHROM"):
                    num_samples = 2 * len(l.split()[9:])
                continue
            (_, pos, rsid, ref, alt, _, _, info) = l.split()[:8]
            pos = int(pos)
            pos_idx = np.where(positions == pos)[0]
            if len(pos_idx) == 0:
                # not a position in our list
                continue
            else:
                idx = pos_idx[0]
            # make sure we're dealing with nucleotides
            if ref not in nucleotides or alt not in nucleotides:
                continue
            # make sure IDs match
            if rsid != IDs[idx]:
                continue
            # check that anc and der alleles match ref and alt from 1kg
            if ref != ancestral[idx] and ref != derived[idx]:
                continue
            if alt != ancestral[idx] and alt != derived[idx]:
                continue
            # if everything looks good, parse the info and get allele freqs
            flip = False
            if ancestral[idx] == alt:
                flip = True
            elif ancestral[idx] != ref:
                # ancestral allel does't match alt or ref
                continue
            infodict = parse_vcf_info(info)
            if flip:
                AC[idx] = num_samples - int(infodict["AC"])
                ALL[idx] = 1 - float(infodict["AF"])
                AFR[idx] = 1 - float(infodict["AFR_AF"])
                EAS[idx] = 1 - float(infodict["EAS_AF"])
                EUR[idx] = 1 - float(infodict["EUR_AF"])
                SAS[idx] = 1 - float(infodict["SAS_AF"])
            else:
                AC[idx] = int(infodict["AC"])
                ALL[idx] = float(infodict["AF"])
                AFR[idx] = float(infodict["AFR_AF"])
                EAS[idx] = float(infodict["EAS_AF"])
                EUR[idx] = float(infodict["EUR_AF"])
                SAS[idx] = float(infodict["SAS_AF"])
            assert AC[idx] > 0 and AC[idx] < num_samples
    df["AC"] = AC
    df["ALL"] = ALL
    df["AFR"] = AFR
    df["EAS"] = EAS
    df["EUR"] = EUR
    df["SAS"] = SAS
    # remove sites where AFR, EAS, EUR, and SAS all have zero frequencies
    focal_afs = AFR + EAS + EUR + SAS
    df = df[focal_afs > 0]
    ## might want this to be df[df["AC"]] > 0]
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
            f"./data/geva/parsed_variants.{data_source}.chr{chrom}.gzip"
        else:
            fname = f"./data/geva/parsed_variants.chr{chrom}.gzip"
        df.to_pickle(fname)
        eprint(current_time(), "saved data!!")
