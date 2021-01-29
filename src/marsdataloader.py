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
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import tqdm

from generate_data import COL_PATHS, RANDOM_SEED, generate_feature_files, save_col, extract_destabilize

# velocity mode tag
CALCULATED = "calc"
ORIGINAL = "orig"

# paths for saving output
OUT_DIR_FORMAT = "data/data_{}window_{}ahead_{}rolling/"

# path to pickle dataloader
LOADER_BASENAME = "dataloader.pkl"

# train test split config constants
CONFIG_NUMS = {1, 3}
X_COL_NAME = "X_train"
TEST_SIZE = 0.2


class MARSDataLoader():
    """
    Processes raw data; Extracts, saves, and loads features
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

        # record dataset info
        self.window_size = window_size
        self.time_ahead = time_ahead
        self.sampling_rate = sampling_rate
        self.time_gap = time_gap
        self.rolling_step = rolling_step
        self.verbose = verbose

        # record and create directory to save feature columns
        self.data_dir = OUT_DIR_FORMAT.format(int(self.window_size * 1000),
                                              int(self.time_ahead * 1000),
                                              int(self.rolling_step * 1000))

        # sample size
        self.n = None

        # ensure all features in COL_PATH, and regenerate all features
        if self.__missing_feature_files():
            if self.verbose:
                print("Missing features, new regenerating...")
            # generate features in COL_PATH and record sample size
            self.n = generate_feature_files(self.window_size,
                                            self.time_ahead,
                                            self.sampling_rate,
                                            self.time_gap,
                                            self.rolling_step,
                                            self.data_dir)

            # generate and save additional features here: destabilizing, nomalized, etc...
            self._save_col(extract_destabilize(self.basic_triples()), "destabilizing")
        else:
            # if all feature present, get sample size from label length
            self.n = self.retrieve_col("label").shape[0]

        # in the end. pickle self in output dir
        pickle.dump(self, open(self._data_path(LOADER_BASENAME), "wb"))

    def _save_col(self, col, col_name):
        """helper function to save given column"""
        if col_name in COL_PATHS:
            save_col(col, col_name, self.data_dir)
        else:
            raise ValueError("Cannot recognize column name {} in COL_PATHS")

    def _data_path(self, basename: str) -> str:
        """
        helper function to return save path for given file under data/
        :param basename: basename of file
        :return: joined path
        """
        return os.path.join(self.data_dir, basename)

    def __missing_feature_files(self) -> list:
        """
        helper function check if all feature paths in COL_PATHS
        :return: whether all feature files exist
        """
        # not present if path not exists
        if not os.path.exists(self.data_dir):
            return list(COL_PATHS.keys())

        missing_feats = []

        # check files under data path
        for feat, feature_basename in COL_PATHS.items():
            # not all present if one not exist
            if not os.path.exists(self._data_path(feature_basename)):
                missing_feats.append(feat)

        return missing_feats

    def basic_triples(self,
                      vel_mode=ORIGINAL) -> np.ndarray:
        """
        returns basic triples of shape (n, sample_rate, 3), last dimension in order of (vel, pos, joystick)
        :param vel_mode: velocity mode
        :return: basic data triples
        """
        # get velocity (n, sample_rate)
        if vel_mode == ORIGINAL:
            velocity = self.retrieve_col("velocity")
        elif vel_mode == CALCULATED:
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
        if col_name not in COL_PATHS:
            raise ValueError("Cannot find column name {} in columns listed in COL_PATHS.".format(col_name))

        col_path = self._data_path(COL_PATHS[col_name])

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

        for col_name in COL_PATHS:
            output[col_name] = self.retrieve_col(col_name)

        return output

    def retrieve_col_slices(self, col_name: str, inds: Union[list, np.ndarray]) -> np.ndarray:
        """helper function to retrieve elements of a given column"""
        if col_name not in COL_PATHS:
            raise ValueError("Cannot find column name {} in columns listed in COL_PATHS.".format(col_name))

        # load array
        col_array = self.retrieve_col(col_name)

        # return slicing
        return col_array[inds]

    def load_splits(self, config_num: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        load or create X matrix and y matrix with given config number. See each config info, use >>> help(dataloader.generate_config<num>)
        :param signature: unique name for this configuration, also used as base name to save path
        :return: train_inds, test_inds, X_train, X_test, y_train, y_test; X of shape (split_size, sampling_rate, num_feats), y of shape (split_size, ...)
        """
        # # prepare paths
        # X_train_path, y_train_path, X_test_path, y_test_path = map(lambda filler: CONFIG_PATHS[config_num].format(filler),
        #                                                            (X_TRAIN, Y_TRAIN, X_TEST, Y_TEST))
        #
        # # return arrays
        # try:
        #     return np.load(self._data_path(X_train_path)), np.load(self._data_path(y_train_path)), \
        #            np.load(self._data_path(X_test_path)), np.load(self._data_path(y_test_path))
        #
        # except FileNotFoundError:

        # to add more configurations:
        # 1) add config number to CONFIG_NUMS
        # 2) create and call _generate_confign function in conditionals below
        if config_num == 1:
            inds_train, inds_test, X_train, X_test, y_train, y_test = _generate_config1(self)
        else:
            raise ValueError("Cannot recognize config_num: {}".format(config_num))

        # print stats
        if self.verbose:
            print(
                "Train-test split done!\n"
                "Total sample size: {:>12}\n"
                "Train sample size: {:>12}\n"
                "Test sample size: {:>13}\n"
                "X_train shape: {:>20}\n"
                "X_test shape: {:>21}\n"
                "y_train shape: {:>20}\n"
                "y_test shape: {:>21}\n".
                    format(self.n,
                           inds_train.shape[0],
                           inds_test.shape[0],
                           str(X_train.shape),
                           str(X_test.shape),
                           str(y_train.shape),
                           str(y_test.shape))
            )

        return inds_train, inds_test, X_train, X_test, y_train, y_test


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


USED_COL_FORMAT = "{}_InTrain"


def __rename_used_cols(df, train_cols: list):
    """helper for renaming columns used in train"""
    return df.rename(columns={col: USED_COL_FORMAT.format(col) for col in train_cols})


def _generate_config1(loader: MARSDataLoader) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    generates and saves X y after train-test split.
    features used: original velocity, position, joystick
    :param loader: DataLoader to generate features from
    :return: train indices, test indices; X train (n_train, sampling_rate, 3), X test (n_test, sampling_rate, 3), y train array (n_train, 2), y test array (n_test, 2)
    """
    # load data needed into DataFrame
    X_all = loader.basic_triples()
    y_all = loader.retrieve_col("label").reshape(-1, 1)

    # record indices for later extracting splitting X and loading all feature entries for false positive analysis
    inds_all = np.arange(y_all.shape[0])

    # train test split
    inds_train, inds_test, y_train, y_test = train_test_split(inds_all, y_all, test_size=TEST_SIZE,
                                                              random_state=RANDOM_SEED)
    # retrieve cols from inds
    X_train = X_all[inds_train]
    X_test = X_all[inds_test]

    # process x or y

    X_train = sequence.pad_sequences(X_train, maxlen=50, padding='post', dtype='float', truncating='post')
    # y_train = y_train.reshape(-1, 1)

    X_test = sequence.pad_sequences(X_test, maxlen=50, padding='post', dtype='float', truncating='post')

    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(y_train)
    y_train = enc.transform(y_train)
    y_test = enc.transform(y_test)

    # return processed x and y
    return inds_train, inds_test, X_train, X_test, y_train, y_test


def _generate_config3(basic_triples: np.ndarray,
                      destabilizing: np.ndarray,
                      y_all: np.ndarray,
                      data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    generates and saves X y after train-test split.
    features used: original velocity, position, joystick, if destabilizing
    :param basic_triples: vel, pos, joy tripes of shape (n, sampling_rate, 3)
    :param destabilizing: boolean column
    :param y_all: labels of shape (n,)
    :param data_dir: data directory to find all feature columns
    :return: X train (n_train, sampling_rate, 4), y train (n_train, 2), X test (n_test, sampling_rate, 4), y test (n_test, 2)
    """

    # load data

    # mark feature columns used

    # train test split

    # vectorize x or y

    # return vectorized x and y

    pass


if __name__ == "__main__":
    l = MARSDataLoader()
    train_inds, test_inds, X_train, X_test, y_train, y_test = l.load_splits(1)
