#!/usr/bin/env python3
# Author: Yonglin Wang
# Date: 2021/1/29
# Functions to generate csv summaries of data statistics and result statistics
# Can run as long as the result csv files exists

import os
import argparse

import pandas as pd

import consts as C


def merge_results(verbose: bool=False) -> None:
    """merge results and configuration files into results/hpcc_results.csv"""
    # read files
    if verbose:
        print("Now loading csv files...")
    exp_ID_name = C.EXP_COL_CONV[C.EXP_ID_COL]
    res_df = pd.read_csv(C.ALL_RES_CSV_PATH, dtype={exp_ID_name: int}, index_col=exp_ID_name)
    config_df = pd.read_csv(C.EXP_ID_LOG, dtype={exp_ID_name: int}, index_col=exp_ID_name)

    assert len(res_df.index) == len(config_df.index), "Numbers of experiments recorded don't match!"

    # check output path to avoid overwriting previous combined results
    comb_output_path = C.COMBINED_FILE_FORMAT
    collision_n = 0
    if os.path.exists(comb_output_path):
        while os.path.exists(comb_output_path.format("_" + str(collision_n))):
            collision_n += 1
        comb_output_path = comb_output_path.format("_" + str(collision_n))
    else:
        comb_output_path = C.COMBINED_FILE_FORMAT.format("")

    # join and save
    if verbose:
        print(f"Now saving {len(res_df.index)} combined experiment entries to {comb_output_path}...")
    res_df.join(config_df, on=exp_ID_name).to_csv(comb_output_path)
    if verbose:
        print("Done!")

def main():

    # Argparser
    # noinspection PyTypeChecker
    argparser = argparse.ArgumentParser(prog="Summary Argparser",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # argparser.add_argument(
    #     '--conv_crit', type=str.lower, default=C.VAL_AUC, choices=C.CONV_CRIT,
    #     help='type of convergence criteria, if early stopping')
    argparser.add_argument(
        '--merge', action='store_true',
        help='whether to merge results and experiment configuration files')
    argparser.add_argument(
        '--silent', action='store_true',
        help='whether to disable console output')

    args = argparser.parse_args()

    if args.merge:
        merge_results(verbose=not args.silent)


if __name__ == "__main__":
    main()