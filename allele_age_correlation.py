import gzip
import pandas as pd
import numpy as np

from util import *
from bin_variant_ages import add_mut_class
import sys

corr = {}

mut_classes = ["A>C", "A>G", "A>T", "C>A", "C>G", "C>T"]
comp_classes = ["T>G", "T>C", "T>A", "G>T", "G>C", "G>A"]
comp_map = {c: m for c, m in zip(mut_classes, comp_classes)}


def filter_df(df):
    df = df.sort_values(by="Pos")
    df = add_mut_class(df)
    df = df[df["Mut"].isin(mut_classes)]
    df = df.reset_index()
    return df


overlapping_variants = {"geva-relate": []}
unique_variants = {"geva": [], "relate": []}


for chrom in range(1, 23):
    print("chromosome", chrom)
    # load data, get IDs, ages, ancestral, and derived alleles
    geva_df = pd.read_pickle(f"./data/geva/parsed_variants.chr{chrom}.gzip")
    tsdate_df = pd.read_pickle(f"./data/tsdate/parsed_variants.chr{chrom}.gzip")
    relate_df = pd.read_pickle(f"./data/relate/parsed_variants.chr{chrom}.gzip")
    geva_df = filter_df(geva_df)
    tsdate_df = filter_df(tsdate_df)
    relate_df = filter_df(relate_df)

    # tsdate data is in GRCh38, while geva and relate are in GRCh37
    # we can compare position, ref, and alt between geva and relate,
    # but will use IDs to compare those two with tsdate

    # compare geva and relate
    overlapping_positions = np.intersect1d(geva_df["Pos"], relate_df["Pos"])
    eprint(
        f"{len(overlapping_positions)} positions overlap between GEVA and Relate",
    )
    ol_geva = geva_df[geva_df["Pos"].isin(overlapping_positions)][["Pos", "Age", "Mut"]]
    ol_relate = relate_df[relate_df["Pos"].isin(overlapping_positions)][
        ["Pos", "Age", "Mut"]
    ]
    ol_geva = ol_geva.reset_index(drop=True)
    ol_relate = ol_relate.reset_index(drop=True)
    unique_geva = geva_df[~geva_df["Pos"].isin(overlapping_positions)][
        ["Pos", "Age", "Mut"]
    ]
    unique_relate = relate_df[~relate_df["Pos"].isin(overlapping_positions)][
        ["Pos", "Age", "Mut"]
    ]

    # get matching mutations
    matching = ol_geva["Mut"] == ol_relate["Mut"]
    ol_geva = ol_geva[matching]
    ol_relate = ol_relate[matching]

    overlapping_variants["geva-relate"].append(
        pd.DataFrame.from_dict(
            {
                "Chr": [chrom] * len(ol_geva),
                "Pos": ol_geva["Pos"],
                "AgeGEVA": ol_geva["Age"],
                "AgeRelate": ol_relate["Age"],
                "Mut": ol_geva["Mut"],
            }
        )
    )
    unique_variants["geva"].append(
        pd.DataFrame.from_dict(
            {
                "Chr": [chrom] * len(unique_geva),
                "Pos": unique_geva["Pos"],
                "Age": unique_geva["Age"],
                "Mut": unique_geva["Mut"],
            }
        )
    )
    unique_variants["relate"].append(
        pd.DataFrame.from_dict(
            {
                "Chr": [chrom] * len(unique_relate),
                "Pos": unique_relate["Pos"],
                "Age": unique_relate["Age"],
                "Mut": unique_relate["Mut"],
            }
        )
    )

    # get correlations, overall and for each type
    ages1 = ol_geva["Age"]
    ages2 = ol_relate["Age"]
    mean1 = np.mean(ages1)
    mean2 = np.mean(ages2)
    corr["GEVA-Relate"] = {"ALL": np.corrcoef(ages1, ages2)[0][1]}
    eprint("correlation coef between geva and relate:", corr["GEVA-Relate"]["ALL"])
    eprint("with r^2 =", corr["GEVA-Relate"]["ALL"] ** 2)
    for m in mut_classes:
        ages1 = ol_geva["Age"][ol_geva["Mut"] == m]
        ages2 = ol_relate["Age"][ol_relate["Mut"] == m]
        c = np.corrcoef(ages1, ages2)[0][1]
        corr["GEVA-Relate"][m] = c
        mean1 = np.mean(ages1)
        mean2 = np.mean(ages2)
        eprint(m, ", r^2 =", c ** 2, ", mean difference:", mean1 - mean2)
    print()
    ###
    # compare tsdate to geva and relate
    # b38 positions map
    tsdate_id_map = {}
    for ID, age, mut in zip(tsdate_df["ID"], tsdate_df["Age"], tsdate_df["Mut"]):
        if ID is None:
            continue
        tsdate_id_map[ID] = {"Age": age, "Mut": mut}

    geva_id_map = {}
    for ID, age, mut in zip(geva_df["ID"], geva_df["Age"], geva_df["Mut"]):
        if ID is None:
            continue
        for k in ID.split(";"):
            if k != ".":
                geva_id_map[k] = {"Age": age, "Mut": mut}

    relate_id_map = {}
    for ID, age, mut in zip(relate_df["ID"], relate_df["Age"], relate_df["Mut"]):
        if ID is None:
            continue
        if ID.startswith("rs"):
            ID = ID.split(":")[0]
            relate_id_map[ID] = {"Age": age, "Mut": mut}

    tsdate_geva_id_overlap = set(tsdate_id_map.keys()).intersection(
        set(geva_id_map.keys())
    )
    tsdate_relate_id_overlap = set(tsdate_id_map.keys()).intersection(
        set(relate_id_map.keys())
    )
    tsdate_geva_id_overlap_match = []
    tsdate_relate_id_overlap_match = []
    for i in tsdate_geva_id_overlap:
        if tsdate_id_map[i]["Mut"] == geva_id_map[i]["Mut"]:
            tsdate_geva_id_overlap_match.append(i)
    for i in tsdate_relate_id_overlap:
        if tsdate_id_map[i]["Mut"] == relate_id_map[i]["Mut"]:
            tsdate_relate_id_overlap_match.append(i)

    # compare ages for overlapping IDs
    ages = np.zeros((len(tsdate_geva_id_overlap_match), 2))
    muts = np.array([tsdate_id_map[_]["Mut"] for _ in tsdate_geva_id_overlap_match])
    for i, k in enumerate(tsdate_geva_id_overlap_match):
        ages[i] = [geva_id_map[k]["Age"], tsdate_id_map[k]["Age"]]

    # overlapping_variants["geva-tsdate"].append(
    #    pd.DataFrame.from_dict(
    #        {
    #            "Chr": [chrom] * len(ages),
    #            "ID": tsdate_geva_id_overlap_match,
    #            "AgeGEVA": ages[:, 0],
    #            "Agetsdate": ages[:, 1],
    #            "Mut": muts,
    #        }
    #    )
    # )

    corr["GEVA-tsdate"] = {"ALL": np.corrcoef(ages[:, 0], ages[:, 1])[0, 1]}

    eprint(
        f"{len(tsdate_geva_id_overlap)} positions overlap between GEVA and tsdate",
    )
    eprint("correlation coef between GEVA and tsdate:", corr["GEVA-tsdate"]["ALL"])
    eprint("with r^2 =", corr["GEVA-tsdate"]["ALL"] ** 2)
    for m in mut_classes:
        ages_mut = ages.compress(muts == m, axis=0)
        ages1 = ages_mut[:, 0]
        ages2 = ages_mut[:, 1]
        c = np.corrcoef(ages1, ages2)[0][1]
        corr["GEVA-tsdate"][m] = c
        mean1 = np.mean(ages1)
        mean2 = np.mean(ages2)
        eprint(m, ", r^2 =", c ** 2, ", mean difference:", mean1 - mean2)

    print()
    # compare ages for overlapping IDs for tsdate and Relate
    ages = np.zeros((len(tsdate_relate_id_overlap_match), 2))
    muts = np.array([tsdate_id_map[_]["Mut"] for _ in tsdate_relate_id_overlap_match])
    for i, k in enumerate(tsdate_relate_id_overlap_match):
        ages[i] = [relate_id_map[k]["Age"], tsdate_id_map[k]["Age"]]

    # overlapping_variants["relate-tsdate"].append(
    #    pd.DataFrame.from_dict(
    #        {
    #            "Chr": [chrom] * len(ages),
    #            "ID": tsdate_relate_id_overlap_match,
    #            "AgeGEVA": ages[:, 0],
    #            "AgeRelate": ages[:, 1],
    #            "Mut": muts,
    #        }
    #    )
    # )

    corr["Relate-tsdate"] = {"ALL": np.corrcoef(ages[:, 0], ages[:, 1])[0, 1]}

    eprint(
        f"{len(tsdate_relate_id_overlap)} positions overlap between Relate and tsdate",
    )
    eprint("correlation coef between Relate and tsdate:", corr["Relate-tsdate"]["ALL"])
    eprint("with r^2 =", corr["Relate-tsdate"]["ALL"] ** 2)
    for m in mut_classes:
        ages_mut = ages.compress(muts == m, axis=0)
        ages1 = ages_mut[:, 0]
        ages2 = ages_mut[:, 1]
        c = np.corrcoef(ages1, ages2)[0][1]
        corr["Relate-tsdate"][m] = c
        mean1 = np.mean(ages1)
        mean2 = np.mean(ages2)
        eprint(m, ", r^2 =", c ** 2, ", mean difference:", mean1 - mean2)

    print()

for k, v in overlapping_variants.items():
    for j, df in enumerate(v):
    #df = pd.concat(v, ignore_index=True)
        pd.to_pickle(df, f"data/ages.shared.chr{j}." + k + ".pkl.gz")

for k, v in unique_variants.items():
    #df = pd.concat(v, ignore_index=True)
    for j, df in enumerate(v):
        pd.to_pickle(df, f"data/ages.unique.chr{j}." + k + ".pkl.gz")

### NOTE: What about those mutations that are not aged in either. Do they contribute
### most to any shifts between methods?
