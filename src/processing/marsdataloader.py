#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yonglin Wang
# Date: 2021/1/27
# Data loader class for organizing numpy feature arrays and assembling training matrices

import os
import pickle
from collections import OrderedDict
from typing import Union, Tuple

import pandas as pd
import numpy as np

from processing.extract_features import generate_feature_files, save_col, extract_destabilize
import consts as C


class MARSDataLoader():
    """
    Lightweight loader class that extracts, saves, and loads features upon initialization, saves only links to files.
    Pickling not recommended; instead, create a new MARSDataLoader in every script and just keep input consistent.
    """

    def __init__(self,
                 window_size=1.0,
                 time_ahead=0.5,
                 sampling_rate=50,
                 time_gap=5,
                 rolling_step=0.5,
                 verbose=True
                 ):

        # ensure time_gap has the right size
        if time_gap < (2 * window_size + time_ahead):
            time_gap = 2 * window_size + time_ahead

        # record dataset info, all time in seconds
        self.window = window_size
        self.ahead = time_ahead
        self.sampling_rate = sampling_rate
        self.time_gap = time_gap
        self.rolling = rolling_step
        self.verbose = verbose

        # convert commonly used info to ms
        self.window_ms = int(window_size * 1000)
        self.ahead_ms = int(time_ahead * 1000)
        self.rolling_ms = int(rolling_step * 1000)

        # record directory to save feature columns
        self.data_dir = C.DATA_OUT_DIR_FORMAT.format(self.window_ms,
                                                     self.ahead_ms,
                                                     self.rolling_ms)
        # record directory to save loader and models
        self.exp_dir = C.EXP_OUT_DIR_FORMAT.format(self.window_ms,
                                                   self.ahead_ms,
                                                   self.rolling_ms)

        # ensure all features in COL_PATH, and regenerate all features
        missing_feats = self.__missing_feature_files()
        if missing_feats:
            if self.verbose:
                print("Missing feature(s): {}\nNow generating...".format(str(missing_feats)[1:-1]))

            # generate initial features if missing any
            if missing_feats.intersection(C.INIT_FEATURES):
                # generate features in COL_PATH and record sample size
                generate_feature_files(self.window,
                                       self.ahead,
                                       self.sampling_rate,
                                       self.time_gap,
                                       self.rolling,
                                       self.data_dir)

            # generate and save additional features here: destabilizing, nomalized, etc...
            non_init_feats = set(C.COL_PATHS.keys()).difference(C.INIT_FEATURES)
            missing_non_init = non_init_feats.intersection(missing_feats)

            # ### Add new features here
            if "destabilizing" in missing_non_init:
                self.__save_new_feature(extract_destabilize(self.basic_triples()), "destabilizing")

        # record sample size, assuming all columns have equal lengths
        self.n = self.retrieve_col("label").shape[0]

    def __save_new_feature(self, generated_col: np.ndarray, col_name: str):
        """internal helper for validating and creating new feature and checking spelling"""
        assert col_name in C.COL_PATHS, "cannot recognize column name {} in COL_PATHS. " \
                                        "Following README.md to add new features".format(col_name)
        self._save_col(generated_col, col_name)

    def _save_col(self, col, col_name):
        """helper function to save given column"""
        assert col_name in C.COL_PATHS, "Cannot recognize column name {} in COL_PATHS".format(col_name)
        save_col(col, col_name, self.data_dir)

    def data_file_path(self, basename: str) -> str:
        """
        helper function to return save path for given file under data/
        :param basename: basename of file
        :return: joined path
        """
        return os.path.join(self.data_dir, basename)

    def exp_file_path(self, basename: str) -> str:
        """
        helper function to return save path for given file under exp/
        :param basename: basename of file
        :return: joined path
        """
        return os.path.join(self.exp_dir, basename)

    def __missing_feature_files(self) -> set:
        """
        helper function check if all feature paths in COL_PATHS
        :return: set of missing features in data directory
        """
        # all missing if path not exists
        if not os.path.exists(self.data_dir):
            return set(C.COL_PATHS.keys())

        missing_feats = []

        # check files under data path
        for feat, feature_basename in C.COL_PATHS.items():
            # not all present if one not exist
            if not os.path.exists(self.data_file_path(feature_basename)):
                missing_feats.append(feat)

        return set(missing_feats)

    def basic_triples(self,
                      vel_mode=C.ORIGINAL) -> np.ndarray:
        """
        returns basic triples of shape (n, sample_rate, 3), last dimension in order of (vel, pos, joystick)
        :param vel_mode: velocity mode
        :return: basic data triples
        """
        # get velocity (n, sample_rate)
        if vel_mode == C.ORIGINAL:
            velocity = self.retrieve_col("velocity")
        elif vel_mode == C.CALCULATED:
            velocity = self.retrieve_col("velocity_cal")
        else:
            raise ValueError("Cannot recognize velocity mode: {}".format(vel_mode))

        # get position and deflection (n, sample_rate)
        position = self.retrieve_col("position")
        joystick = self.retrieve_col("joystick")

        # TODO add normalize and standardize handle?

        # return d-stacked array (n, sample_rate, 3)
        return np.dstack([velocity, position, joystick])

    def retrieve_col(self, col_name: Union[str, list]) -> np.ndarray:
        """
        retrieve column with specified name. Must be one of keys in COL_PATHS.
        :param col_name: name or names of the column to return
        :return: column of shape (n, ) or (n, sampling_rate) or (n, sampling_rate, n_features)
        """
        if col_name not in C.COL_PATHS:
            raise ValueError("Cannot find column name {} in columns listed in COL_PATHS.".format(col_name))

        col_path = self.data_file_path(C.COL_PATHS[col_name])

        try:
            return np.load(col_path)

        except FileNotFoundError:
            raise FileNotFoundError("Cannot find column array file {}".format(col_path))

    def retrieve_all_cols(self) -> OrderedDict:
        """
        retrieve all columns with shape (n, ...) listed in COL_PATHS
        :return: dictionary of arrays indexed by column names
        """
        output = OrderedDict()

        for col_name in C.COL_PATHS:
            output[col_name] = self.retrieve_col(col_name)

        return output

    def retrieve_col_slices(self, col_name: str, inds: Union[list, np.ndarray]) -> np.ndarray:
        """helper function to retrieve elements of a given column"""
        assert col_name in C.COL_PATHS, "Cannot find column name {} in columns listed in COL_PATHS.".format(col_name)

        # load array
        col_array = self.retrieve_col(col_name)

        # return slicing
        return col_array[inds]


#
# def retrieve_loader_with(window: float,
#                          ahead: float,
#                          rolling: float)->MARSDataLoader:
#     """retrieve loader with given configuration"""
#     try:
#         pickle.load(os.path.join(C.DATA_OUT_DIR_FORMAT, ))


def generate_all_feat_df(loader, used_col=None) -> pd.DataFrame:
    """
    generate DataFrame containing all np columns, while changing name of used columns to USED_COL_FORMAT
    :param loader: MARSDataLoader to retrieve np columns from
    :param used_col: list of columns to be marked used
    :return: DataFrame containing all raw feat columns, name of used columns to USED_COL_FORMAT
    """

    # generate DataFrame
    output = pd.DataFrame()
    feat_cols = loader.retrieve_all_cols()
    for col_name, arr in feat_cols.items():
        output[col_name] = arr.tolist()

    # change name
    if used_col:
        __rename_used_cols(output, used_col)

    return output


def __rename_used_cols(df, train_cols: list):
    """helper for renaming columns used in train"""
    return df.rename(columns={col: C.USED_COL_FORMAT.format(col) for col in train_cols})


if __name__ == "__main__":
    from processing.dataset_config import load_splits
    l = MARSDataLoader()
    train_inds, test_inds, X_train, X_test, y_train, y_test = load_splits(l, 1)
