import gzip
import pandas as pd
import numpy as np

from util import *

import sys

chrom = int(sys.argv[1])

# load data, get IDs, ages, ancestral, and derived alleles

geva_df = pd.read_pickle(f"./data/geva/parsed_variants.chr{chrom}.gzip")
tsdate_df = pd.read_pickle(f"./data/tsdate/parsed_variants.chr{chrom}.gzip")
relate_df = pd.read_pickle(f"./data/relate/parsed_variants.chr{chrom}.gzip")

geva_df = geva_df.sort_values(by="Pos")
relate_df = relate_df.sort_values(by="Pos")
tsdate_df = tsdate_df.sort_values(by="Pos")

# tsdate data is in GRCh38, while geva and relate are in GRCh37
# we can compare position, ref, and alt between geva and relate,
# but will use IDs to compare those two with tsdate

# compare geva and relate
overlapping_positions = np.intersect1d(geva_df["Pos"], relate_df["Pos"])
eprint(
    current_time(),
    f"{len(overlapping_positions)} positions overlap between GEVA and Relate",
)

ol_geva = geva_df[geva_df["Pos"].isin(overlapping_positions)][
    ["Pos", "Anc", "Der", "Age"]
]
ol_relate = relate_df[relate_df["Pos"].isin(overlapping_positions)][
    ["Pos", "Anc", "Der", "Age"]
]

matching = np.logical_and(
    np.array(ol_geva["Anc"]) == np.array(ol_relate["Anc"]),
    np.array(ol_geva["Der"]) == np.array(ol_relate["Der"]),
)

ol_geva = ol_geva[matching]
ol_relate = ol_relate[matching]

ages_geva = ol_geva["Age"]
ages_relate = ol_relate["Age"]

corr_geva_relate = np.corrcoef(ages_geva, ages_relate)[0][1]

eprint(current_time(), "correlation coef between geva and relate:", corr_geva_relate)
eprint(current_time(), "with r^2 =", corr_geva_relate ** 2)

# compare tsdate to geva and relate

# b38 positions map
tsdate_id_map = {}
for ID, age in zip(tsdate_df["ID"], tsdate_df["Age"]):
    if ID is None:
        continue
    tsdate_id_map[ID] = age

geva_id_map = {}
for ID, age in zip(geva_df["ID"], geva_df["Age"]):
    if ID is None:
        continue
    for k in ID.split(";"):
        if k != ".":
            geva_id_map[k] = age

relate_id_map = {}
for ID, age in zip(relate_df["ID"], relate_df["Age"]):
    if ID is None:
        continue
    if ID.startswith("rs"):
        ID = ID.split(":")[0]
        relate_id_map[ID] = age

tsdate_geva_id_overlap = set(tsdate_id_map.keys()).intersection(set(geva_id_map.keys()))
tsdate_relate_id_overlap = set(tsdate_id_map.keys()).intersection(
    set(relate_id_map.keys())
)

# compare ages for overlapping IDs
ages_tsdate_geva = np.zeros((len(tsdate_geva_id_overlap), 2))
for i, k in enumerate(tsdate_geva_id_overlap):
    ages_tsdate_geva[i] = [tsdate_id_map[k], geva_id_map[k]]

corr_geva_tsdate = np.corrcoef(ages_tsdate_geva[:, 0], ages_tsdate_geva[:, 1])[0, 1]

eprint(
    current_time(),
    f"{len(tsdate_geva_id_overlap)} positions overlap between GEVA and tsdate",
)
eprint(current_time(), "correlation coef between GEVA and tsdate:", corr_geva_tsdate)
eprint(current_time(), "with r^2 =", corr_geva_tsdate ** 2)

ages_tsdate_relate = np.zeros((len(tsdate_relate_id_overlap), 2))
for i, k in enumerate(tsdate_relate_id_overlap):
    ages_tsdate_relate[i] = [tsdate_id_map[k], relate_id_map[k]]

corr_tsdate_relate = np.corrcoef(ages_tsdate_relate[:, 0], ages_tsdate_relate[:, 1])[
    0, 1
]

eprint(
    current_time(),
    f"{len(tsdate_relate_id_overlap)} positions overlap between tsdate and Relate",
)
eprint(
    current_time(), "correlation coef between tsdate and Relate:", corr_tsdate_relate
)
eprint(current_time(), "with r^2 =", corr_tsdate_relate ** 2)
