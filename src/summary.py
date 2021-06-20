#!/usr/bin/env python3
# Author: Yonglin Wang
# Date: 2021/1/29
# Functions to generate csv summaries of data statistics and merge result statistics

import os
import argparse
import re
from typing import Tuple

import pandas as pd
import numpy as np

import consts as C
from processing.marsdataloader import MARSDataLoader


def merge_results(verbose: bool=False) -> None:
    """merge results and configuration files into results/hpcc_results.csv"""
    # read files
    exp_ID_name = C.EXP_COL_CONV[C.EXP_ID_COL]
    res_df = pd.read_csv(C.ALL_RES_CSV_PATH, dtype={exp_ID_name: int}, index_col=exp_ID_name)
    config_df = pd.read_csv(C.EXP_ID_LOG, dtype={exp_ID_name: int}, index_col=exp_ID_name)

    assert len(res_df.index) == len(config_df.index), "Numbers of experiments recorded don't match!"

    # check output path to avoid overwriting previous combined results
    comb_output_path = __find_next_available_filename(C.COMBINED_FILE_FORMAT)

    # join and save
    res_df.join(config_df, on=exp_ID_name).to_csv(comb_output_path)
    if verbose:
        print(f"{len(res_df.index)} experiment entries combined and saved at {comb_output_path}")

def generate_dataset_stats(dataset_parent_dir:str, stats_csv_save_path: str, verbose=False):
    """generate dataset stats and save at given path"""
    # find dataset folders in dataset dir
    datasets = []
    pat = re.compile(C.DATASET_PATTERN)
    for path in os.listdir(dataset_parent_dir):
        match = pat.match(path)
        if match:
            # record dataset path, win, ahead, rolling
            datasets.append([os.path.join(dataset_parent_dir, match.group(0)),
                             match.group(1), match.group(2), match.group(3)])

    all_dataset_stats = []
    for dataset_dir, window, ahead, rolling in datasets:
        # get train, test inds, and labels
        train_inds, test_inds, labels = __get_train_test_inds_labels(dataset_dir)
        train_labels, test_labels = labels[train_inds], labels[test_inds]
        exc_non, exc_crash = __get_excluded_number(dataset_dir)

        # Record stats
        all_dataset_stats.append({
                                "window": window,
                                "time ahead": ahead,
                                "rolling step": rolling,
                                "train crash count": np.sum(train_labels == 1),
                                "train noncrash count": np.sum(train_labels == 0),
                                "train 0:1 ratio": np.sum(train_labels == 0)/np.sum(train_labels == 1),
                                "test crash count": np.sum(test_labels == 1),
                                "test noncrash count": np.sum(test_labels == 0),
                                "test 0:1 ratio": np.sum(test_labels == 0) / np.sum(test_labels == 1),
                                "dataset total": labels.shape[0],
                                "train set total": train_labels.shape[0],
                                "test set total": test_labels.shape[0],
                                "excluded non-crashed segment total": exc_non,
                                "excluded crashed segment total": exc_crash
                                })
    # generate and save dataset csv file
    pd.DataFrame(all_dataset_stats).to_csv(stats_csv_save_path)

    if verbose:
        print(f"Statistics for {len(datasets)} datasets saved at {stats_csv_save_path}")


def __find_next_available_filename(path_format):
    """find the next available relative path name"""
    # check output path to avoid overwriting previous combined results
    if os.path.exists(path_format.format("")):
        collision_n = 2
        while os.path.exists(path_format.format("_" + str(collision_n))):
            collision_n += 1
        return path_format.format("_" + str(collision_n))
    else:
        return path_format.format("")


def __get_excluded_number(dataset_dir: str) -> Tuple[int, int]:
    """return the numbers of 1) excluded non-crashed human segments, and 2) excluded crashed segments in given dataset directory"""
    # get excluded crash numbers
    debug_pat = re.compile(C.DEBUG_PATTERN)
    # load label and train, test inds
    try:
        exclude_path = [path for path in os.listdir(dataset_dir) if debug_pat.match(path)][0]
    except IndexError:
        raise ValueError(f"Cannot find excluded crashes under {dataset_dir}")

    # get excluded segments
    df = pd.read_csv(os.path.join(dataset_dir, exclude_path))
    non_crash_segs = sum(df.crash_ind == -1)

    return non_crash_segs, len(df.index) - non_crash_segs


def __get_train_test_inds_labels(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """return the train inds, test inds, and labels as three numpy arrays"""
    try:
        train_inds = np.load(os.path.join(dataset_dir, C.INDS_PATH["train"]))
        test_inds = np.load(os.path.join(dataset_dir, C.INDS_PATH["test"]))
        labels = np.load(os.path.join(dataset_dir, C.COL_PATHS["label"]))
    except IOError:
        raise FileNotFoundError(f"At least 1 of the following files missing under {dataset_dir}: "
                                f"{C.INDS_PATH['train']}, {C.INDS_PATH['test']}, {C.COL_PATHS['label']}")

    return train_inds, test_inds, labels


def debug_datasets(dataset_parent_dir:str, stats_csv_save_path: str, verbose=False):
    """generate dataset stats and save at given path"""
    # find dataset folders in dataset dir
    datasets = []
    pat = re.compile(C.DATASET_PATTERN)
    for path in os.listdir(dataset_parent_dir):
        match = pat.match(path)
        if match:
            # record dataset path, win, ahead, rolling
            datasets.append([os.path.join(dataset_parent_dir, match.group(0)),
                             match.group(1), match.group(2), match.group(3)])

    all_dataset_stats = []
    for dataset_dir, window, ahead, rolling in datasets:
        # get train, test inds, and labels
        train_inds, test_inds, labels = __get_train_test_inds_labels(dataset_dir)
        train_labels, test_labels = labels[train_inds], labels[test_inds]
        exc_non, exc_crash = __get_excluded_number(dataset_dir)

        # Record stats
        all_dataset_stats.append({
                                "window": window,
                                "time ahead": ahead,
                                "rolling step": rolling,
                                "train crash count": np.sum(train_labels == 1),
                                "train noncrash count": np.sum(train_labels == 0),
                                "train 0:1 ratio": np.sum(train_labels == 0)/np.sum(train_labels == 1),
                                "test crash count": np.sum(test_labels == 1),
                                "test noncrash count": np.sum(test_labels == 0),
                                "test 0:1 ratio": np.sum(test_labels == 0) / np.sum(test_labels == 1),
                                "dataset total": labels.shape[0],
                                "train set total": train_labels.shape[0],
                                "test set total": test_labels.shape[0],
                                "excluded non-crashed segment total": exc_non,
                                "excluded crashed segment total": exc_crash
                                })
    # generate and save dataset csv file
    pd.DataFrame(all_dataset_stats).to_csv(stats_csv_save_path)

    if verbose:
        print(f"Statistics for {len(datasets)} datasets saved at {stats_csv_save_path}")


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
        '--dataset', action='store_true',
        help='whether to summarize dataset stats (e.g. total samples, train-test split sizes, crash-noncrash ratios, etc.)')
    argparser.add_argument(
        '--silent', action='store_true',
        help='whether to disable console output')

    args = argparser.parse_args()

    if args.merge:
        merge_results(verbose=not args.silent)

    if args.dataset:
        generate_dataset_stats(C.DATA_DIR, __find_next_available_filename(C.DATASET_STATS_FORMAT), verbose=not args.silent)


if __name__ == "__main__":
    main()