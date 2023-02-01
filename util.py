import pandas as pd
import gzip
from Bio import SeqIO
from datetime import datetime
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()


def current_time():
    return " [" + datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S") + "]"


def append_reference_triplet(df, chrom, genome_build):
    """
    Given a data frame, add a column with the reference triplet for each
    site.
    """
    fasta = f"./data/reference_genomes/{genome_build}/chr{chrom}.fa.gz"
    with gzip.open(fasta, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            ref_genome = record
    triplets = []
    for pos in df["Pos"]:
        triplet = str(ref_genome[pos - 2 : pos + 1].seq).upper()
        if "N" in triplet:
            triplets.append(".")
        else:
            triplets.append(triplet)
    df["Tri"] = triplets
    df = df[df["Tri"] != "."]
    return df


def parse_vcf_info(info):
    data = {}
    for f in info.split(";"):
        if "=" not in f:
            continue
        k, v = f.split("=")
        data[k] = v
    return data


complement = {"C": "G", "G": "C", "T": "A", "A": "T"}


def reverse_complement(seq):
    new = list(seq[::-1])
    for i, v in enumerate(new):
        new[i] = complement[v]
    return "".join(new)


