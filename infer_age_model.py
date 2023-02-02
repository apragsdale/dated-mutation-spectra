import pandas as pd
import numpy as np

pd.set_option("mode.chained_assignment", None)

from Bio import SeqIO

from util import *


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

    # remove European TC triplets and their complements
    euro_triplets = [["TCC", "TTC"], ["ACC", "ATC"], ["TCT", "TTC"], ["CCC", "CTC"]]
    for tri_from, tri_to in euro_triplets:
        df = df[(df["Ref_triplet"] != tri_from) & (df["Alt_triplet"] != tri_to)]
        df = df[
            (df["Ref_triplet"] != reverse_complement(tri_from))
            & (df["Alt_triplet"] != reverse_complement(tri_to))
        ]

    # collapse mutation classes
    mut_classes = ["A>C", "A>G", "A>T", "C>A", "C>G", "C>T"]
    complementary_classes = ["T>G", "T>C", "T>A", "G>T", "G>C", "G>A"]
    class_map = {m1: m2 for m1, m2 in zip(complementary_classes, mut_classes)}
    muts = list(df["mut_class"])
    for i, mut in enumerate(muts):
        if mut in complementary_classes:
            muts[i] = class_map[mut]
    df["mut_class"] = muts
    return df


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


if __name__ == "__main__":
    # load and filter trio data
    df = load_and_filter_trio_data()
    mutation_spectrum = get_mutation_spectrum(df)
    eprint("mutation_spectrum:", mutation_spectrum / mutation_spectrum.sum())
