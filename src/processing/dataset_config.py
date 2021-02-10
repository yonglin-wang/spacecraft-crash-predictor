# -*- coding: utf-8 -*-
# Author: Yonglin Wang
# Date: 2021/1/29
"""Data wrangling configurations for X and y after train test split, output ready for fitting in model"""

from typing import Tuple
from functools import partial

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence

from processing.marsdataloader import MARSDataLoader
from processing.extract_features import broadcast_to_sampled
import consts as C


def load_splits(loader: MARSDataLoader, config_id: int) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    load or create X matrix and y matrix with given config number. See each config info, use >>> help(
    dataloader.generate_config<num>)
    :param config_id: ID for configuration to use
    :return: train_inds, test_inds, X_train, X_test, y_train, y_test; X of shape (split_size,
    sampling_rate, num_feats), y of shape (split_size, ...)
    """

    # ### Load features and labels in specified configuration ID
    # to add more configurations:
    # 1) add unique config number n to C.CONFIG_NUMS
    # 2) create and call _generate_config_n function in conditionals below
    # load config function (maybe partially filled)
    if config_id == 1:
        # configuration 1: original vel, pos, joy
        config = partial(_generate_config_1_2, vel_mode=C.ORIGINAL)
    elif config_id == 2:
        # configuration 2: calculated vel, pos, joy
        config = partial(_generate_config_1_2, vel_mode=C.CALCULATED)
    elif config_id == 3:
        # configuration 3: original vel, pos, joy, destabilizing deflection
        config = _generate_config_3
    else:
        raise ValueError("Cannot recognize config_num: {}".format(config_id))

    inds_train, inds_test, X_train, X_test, y_train, y_test = config(loader)

    # print stats
    if loader.verbose:
        print(
            "Train-test split done!\n"
            "Total sample size: {:>16}\n"
            "Train sample size: {:>16}\n"
            "Test sample size: {:>17}\n"
            "Input shapes:\n"
            "X_train shape: {:>20}\n"
            "X_test shape: {:>21}\n"
            "y_train shape: {:>20}\n"
            "y_test shape: {:>21}\n".
                format(loader.total_sample_size,
                       inds_train.shape[0],
                       inds_test.shape[0],
                       str(X_train.shape),
                       str(X_test.shape),
                       str(y_train.shape),
                       str(y_test.shape))
        )

    return inds_train, inds_test, X_train, X_test, y_train, y_test


def _generate_config_1_2(loader: MARSDataLoader,
                         vel_mode:str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    generates and saves X y after train-test split.
    features used: velocity, position, joystick
    :param loader: DataLoader to generate features from
    :return: train indices, test indices; X train (n_train, sampling_rate, 3), X test (n_test, sampling_rate, 3), y train array (n_train, 2), y test array (n_test, 2)
    """

    # ### load data needed
    X_all = loader.basic_triples(vel_mode=vel_mode)
    y_all = loader.retrieve_col("label").reshape(-1, 1)

    # record indices for later extracting splitting X and loading all feature entries for false positive analysis
    inds_all = np.arange(y_all.shape[0])

    # ### train test split
    inds_train, inds_test, y_train, y_test = train_test_split(inds_all, y_all, test_size=C.TEST_SIZE,
                                                              random_state=C.RANDOM_SEED)
    # retrieve cols from inds
    X_train = X_all[inds_train]
    X_test = X_all[inds_test]

    # ### process x or y
    # pad X sequences
    # X_train = sequence.pad_sequences(X_train, maxlen=50, padding='post', dtype='float', truncating='post')
    # X_test = sequence.pad_sequences(X_test, maxlen=50, padding='post', dtype='float', truncating='post')

    # return processed x and y
    return inds_train, inds_test, X_train, X_test, y_train, y_test


def _generate_config_3(loader: MARSDataLoader) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    generates and saves X y after train-test split.
    features used: original velocity, position, joystick, destabilizing
    :param loader: DataLoader to generate features from
    :return: X train (n_train, sampling_rate, 4), y train (n_train, 2), X test (n_test, sampling_rate, 4), y test (n_test, 2)
    """

    # load data
    # X_all = broadcast_to_sampled(loader.retrieve_col("destabilizing"), loader.basic_triples())
    X_all = np.dstack([loader.basic_triples(), loader.retrieve_col("destabilizing")])
    y_all = loader.retrieve_col("label").reshape(-1, 1)

    # record indices for later extracting splitting X and loading all feature entries for false positive analysis
    inds_all = np.arange(y_all.shape[0])

    # ### train test split
    inds_train, inds_test, y_train, y_test = train_test_split(inds_all, y_all,
                                                              test_size=C.TEST_SIZE,
                                                              random_state=C.RANDOM_SEED)
    # retrieve cols from inds
    X_train = X_all[inds_train]
    X_test = X_all[inds_test]

    return inds_train, inds_test, X_train, X_test, y_train, y_test



if __name__ == "__main__":
    loader = MARSDataLoader(window_size=2.0, time_ahead=1.0, verbose=True)
    inds_train, inds_test, X_train, X_test, y_train, y_test = load_splits(loader, 3)