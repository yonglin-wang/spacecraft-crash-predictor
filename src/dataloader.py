#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yonglin Wang
# Date: 2021/1/27
# Data loader class for organizing numpy feature arrays

import os
import glob
import pickle

import pandas as pd
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from datetime import timedelta
# import matplotlib.pyplot as plt
import warnings
from pandas.core.common import SettingWithCopyWarning
from typing import Union
import tqdm

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from generate_data import COL_PATHS, preprocess_data, save_col, extract_destabilize

# argparser value checker
MIN_STEP = 0.04

# crash event criteria
MIN_ENTRIES_BETWEEN = 2     # minimum # of entries between two crash events

# velocity mode tag
CALCULATED = "calc"
ORIGINAL = "orig"

# paths for saving output
OUT_DIR_FORMAT = "data/data_{}window_{}ahead_{}rolling/"

# path to pickle dataloader
LOADER_PATH = "dataloader.pkl"

CRASH_FILE_FORMAT = "crash_feature_label_{}ahead_{}scale_test"
NONCRASH_FILE_FORMAT = "noncrash_feature_label_{}ahead_{}scale_test"
DEBUG_EXCLUDE_FORMAT = "exclude_{}ahead_{}scale_test.csv"

# columns with sampling rate in their shape (don't need broadcasting to stack with other non-sampled features)
SAMPLED_COLS = ("velocity", "velocity_cal", "position", "joystick", "destabilizing")
# COLS_TO_INTERPOLATE = ('currentVelRoll', 'currentPosRoll', 'calculated_vel', 'joystickX')
# OUT_COLS_AFTER_INTERPOLATE = ("features_cal_vel", "features_org_vel", 'label', 'peopleTrialKey',
#                               'start_seconds', 'end_seconds')


class DataLoader():
    """
    Processes raw data; Extracts, saves, and loads features
    """
    def __init__(self,
                 window_size=1.0,
                 time_ahead=0.5,
                 sampling_rate=50,
                 time_gap=5,
                 rolling_step=0.5
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

        # record and create directory to save feature columns
        self.outdir = OUT_DIR_FORMAT.format(int(self.window_size*1000),
                                            int(self.time_ahead*1000),
                                            int(self.rolling_step*1000))

        # generate basic features here, import from generate_data.py
        preprocess_data(window_size, time_ahead, sampling_rate, time_gap, rolling_step, self.outdir)

        # generate and save additional features here: destabilizing, nomalized, etc...
        self._generate_save_col(extract_destabilize(self.basic_triples()), "destabilizing")

        # in the end. pickle self in output dir
        pickle.dump(self, open(self._data_path(LOADER_PATH), "wb"))

    def _generate_save_col(self, col, col_name):
        """helper function to save given column"""
        if col_name in COL_PATHS:
            save_col(col, col_name, self.outdir)
        else:
            raise ValueError("Cannot recognize column name {} in COL_PATHS")

    def _data_path(self, basename:str)->str:
        """
        helper function to return save path for given file under data/
        :param basename: basename of file
        :return: joined path
        """
        return os.path.join(self.outdir, basename)

    def basic_triples(self, mode=ORIGINAL)->np.ndarray:
        """
        returns basic triples of shape (n, sample_rate, 3), last dimension in order of (vel, pos, joystick)
        :param mode: velocity mode
        :return: basic data triples
        """
        # get velocity (n, sample_rate)
        if mode == ORIGINAL:
            velocity = self.retrieve_col("velocity")
        elif mode == CALCULATED:
            velocity = self.retrieve_col("velocity_cal")
        else:
            raise ValueError("Cannot recognize velocity mode: {}".format(mode))

        # get position and deflection (n, sample_rate)
        position = self.retrieve_col("position")
        joystick = self.retrieve_col("joystick")

        #TODO add normalize handle?

        # return d-stacked array (n, sample_rate, 3)
        return np.dstack([velocity, position, joystick])


    def retrieve_col(self, col_name:Union[str, list])->np.ndarray:
        """
        retrieve column with specified name. Must be one of keys in COL_PATHS.
        :param col_name: name or names of the column to return
        :return: column of shape (n, ) or (n, sampling_rate) or (n, sampling_rate, n_features)
        """
        # TODO allow list of cols via np.broadcast_to(arr4.reshape(-1,1), arr1.shape).shape

        if col_name not in COL_PATHS:
            raise ValueError("Cannot recognize column name {} in preset dictionary.".format(col_name))

        col_path = self._data_path(COL_PATHS[col_name])

        try:
            return np.load(col_path)

        except FileNotFoundError:
            raise FileNotFoundError("Cannot find column array file {}".format(col_path))

if __name__ == "__main__":
    pass