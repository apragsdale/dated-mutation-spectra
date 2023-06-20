"""
The output of this script is a pandas dataframe, storing information about
positions, ancestral/derived mutations, allele frequencies, its triplet
context, variant IDs (rsid), and age.

This script, along with process_geva.py and process_relate.py, will create
dataframes with consistent column names and information. Parsed data is saved
as a pickled pandas df in `./data/tsdate/parsed_variants.chr[chrom].gzip` and can
be loaded as `df = pd.read_pickle(fname)`.

output columns:
    ['Chr', 'Pos', 'ID', 'Age', 'Ref', 'Alt', 'Anc', 'Der', 'Tri', 'AC',
       'ALL', 'AFR', 'EAS', 'EUR', 'SAS']

usage: python process_trees.py [optional] -c [chrom]

If the chrom flag is not given, we process all chromosomes in series.
"""

import tskit, tsdate, tszip
import gzip
import sys, os
import json
import numpy as np
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


def load_ts(fname):
    """
    Trees are stored in zipped format. This function unzips those tree
    files, loads as a tree sequence object, and the re=zips the files.
    It returns the tree sequence.
    """
    if not os.path.isfile(f"{fname}.tsz"):
        raise IOError
    return tszip.decompress(f"{fname}.tsz")


def get_sample_node_map(ts):
    # we want the AFR, EUR, EAS, and SAS samples from the thousand genomes
    # load the pop IDs
    pops = ["AFR", "EUR", "EAS", "SAS"]
    pop_samples = {p: [] for p in pops}
    for line in open("./data/1000G/integrated_call_samples_v3.20130502.ALL.panel", "r"):
        sample, pop, super_pop, gender = line.split()
        if super_pop in pops:
            pop_samples[super_pop].append(sample)
    pop_nodes = {p: [] for p in pops}
    for i in ts.individuals():
        ind_metadata = json.loads(i.metadata)
        if "sample" in ind_metadata:
            sample = ind_metadata["sample"]
        elif "individual_id" in ind_metadata:
            sample = ind_metadata["individual_id"]
        elif "sample_id" in ind_metadata:
            sample = ind_metadata["sample_id"]
        else:
            raise (ValueError)
        for p in pops:
            if sample in pop_samples[p]:
                for n in i.nodes:
                    pop_nodes[p].append(n)
    node_map = ["SKIP" for _ in range(ts.num_samples)]
    for p in pops:
        for n in pop_nodes[p]:
            node_map[n] = p
    return np.array(node_map)


def get_variant_data(ts, chrom):
    """
    For each site, get mutation data, which we will store as a pandas dataframe
    """
    pops = ["AFR", "EAS", "EUR", "SAS"]
    data = {}
    kept_pos = []
    kept_nodes = []
    # get site times
    site_times = tsdate.sites_time_from_ts(ts, node_selection="arithmetic")
    # we first go through the trees and get site and mutation information
    for t, s in zip(site_times, ts.sites()):
        # only keep sites with a single mapped mutation
        if len(s.mutations) != 1:
            continue
        # get the site metadata
        site_metadata = json.loads(s.metadata)
        pos = int(s.position)
        pos_data = {}
        pos_data["Chr"] = chrom
        pos_data["Pos"] = pos
        pos_data["ID"] = site_metadata["ID"]
        pos_data["Ref"] = site_metadata["REF"]
        pos_data["Alt"] = s.mutations[0].derived_state
        pos_data["Anc"] = s.ancestral_state
        if pos_data["Ref"] == pos_data["Anc"]:
            pos_data["Der"] = pos_data["Alt"]
        else:
            pos_data["Der"] = pos_data["Ref"]
        pos_data["Age"] = t
        kept_pos.append(pos)
        kept_nodes.append(s.mutations[0].node)
        data[pos] = pos_data
    # we then go through the trees to pull out frequencies from node data
    node_map = get_sample_node_map(ts)
    sample_counts = {p: np.sum(node_map == p) for p in pops}
    pos_counter = 0
    for tree in ts.trees():
        left, right = tree.interval
        if pos_counter == len(kept_pos):
            # we're done
            break
        if kept_pos[pos_counter] >= right:
            # advance trees until our position is within the tree interval
            continue
        while kept_pos[pos_counter] < right:
            # process all positions on this tree
            pos = kept_pos[pos_counter]
            assert pos >= left and pos < right
            node = kept_nodes[pos_counter]
            samples = list(tree.samples(u=node))
            AC = len(samples)
            ALL = AC / ts.num_samples
            ACs = {}
            mut_pops = node_map[samples]
            for p in pops:
                ACs[p] = np.sum(mut_pops == p)
            data[pos]["AC"] = AC
            data[pos]["ALL"] = ALL
            for p in pops:
                data[pos][p] = ACs[p] / sample_counts[p]
            pos_counter += 1
            if pos_counter == len(kept_pos):
                break

    # create df
    df = pd.DataFrame.from_dict(data, orient="index")
    # remove sites with zero frequency in our focal populations
    df = df[df["AFR"] + df["EUR"] + df["EAS"] + df["SAS"] > 0]
    # append triplets using util function
    df = append_reference_triplet(df, chrom, "GRCh38")
    # reorder columns
    df = df[
        [
            "Chr",
            "Pos",
            "ID",
            "Age",
            "Ref",
            "Alt",
            "Anc",
            "Der",
            "Tri",
            "AC",
            "ALL",
            "AFR",
            "EAS",
            "EUR",
            "SAS",
        ]
    ]
    return df


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])
    if args.chrom == 0:
        chroms = range(1, 23)
    else:
        chroms = [args.chrom]

    for chrom in chroms:
        eprint("Parsing chromosome", chrom)
        eprint(current_time())
        dfs = {}
        for arm in ["p", "q"]:
            eprint(current_time(), "processing arm", arm)
            # load the tree sequence
            try:
                ts = load_ts(f"./data/tsdate/hgdp_tgp_sgdp_chr{chrom}_{arm}.dated.trees")
                eprint(current_time(), "loaded data for arm", arm)
            except IOError:
                eprint(current_time(), "no data for arm", arm)
                continue
            # get the variant information from the trees
            dfs[arm] = get_variant_data(ts, chrom)
            eprint(current_time(), "finished parsing arm", arm)
        # combine the chromosome arms
        if "p" in dfs:
            df = dfs["p"]
            if "q" in dfs:
                df = pd.concat([df, dfs["q"]], ignore_index=True)
        elif "q" in dfs:
            df = dfs["q"]
        # save the data
        df.to_pickle(f"./data/tsdate/parsed_variants.chr{chrom}.gzip")
        eprint(current_time(), "saved data!!")
