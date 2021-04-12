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


def load_dataset(loader: MARSDataLoader, config_id: int, test_split=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    load X matrix and y matrix with given config number.
    :param loader: loader to retrieve columns from
    :param config_id: ID for configuration to use
    :return: X and y, from train or test split
    """
    # ### Load features and labels in specified configuration ID
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

    X_all, y_all = config(loader)

    if test_split:
        inds = loader.retrieve_inds(get_train_split=False)
    else:
        inds = loader.retrieve_inds(get_train_split=True)

    return X_all[inds], y_all[inds]



def _generate_config_1_2(loader: MARSDataLoader,
                         vel_mode:str) -> Tuple[np.ndarray, np.ndarray]:
    """
    generates and saves X y after train-test split.
    features used: velocity, position, joystick
    :param loader: DataLoader to generate features from
    :return: X_all, y_all
    """

    # ### load data needed
    X_all = loader.basic_triples(vel_mode=vel_mode)
    y_all = loader.retrieve_col("label").reshape(-1, 1)

    # return x and y
    return X_all, y_all


def _generate_config_3(loader: MARSDataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    generates and saves X y after train-test split.
    features used: original velocity, position, joystick, destabilizing
    :param loader: DataLoader to generate features from
    :return: X_all, y_all
    """

    # load data
    X_all = np.dstack([loader.basic_triples(), loader.retrieve_col("destabilizing")])
    y_all = loader.retrieve_col("label").reshape(-1, 1)

    return X_all, y_all


if __name__ == "__main__":
    loader = MARSDataLoader(window_size=0.3, time_ahead=0.5, rolling_step=0.7, verbose=True)
    X_all, y_all = load_dataset(loader, 3)