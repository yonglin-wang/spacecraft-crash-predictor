#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yonglin Wang
# Date: 2021/1/27
# Functions for extracting and saving feature numpy arrays

import os
import time
import pandas as pd
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d

from typing import Union
from tqdm import tqdm

import argparse
import random

from utils import display_exec_time
import consts as C

# control for randomness in case of any
RANDOM_SEED = 2020
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def sliding_window(interval_series: pd.Series, time_scale: float, rolling_step: float, avoid_duplicate=False) -> list:
    """
    Identify extractable valid window boundaries (inclusive) in given slice of the seconds column.
    Valid window is defined as window that contains and only contains more than 1 non-crash events.
    :authors: Jie Tang, Yonglin Wang
    :param interval_series: series of time points between 2 crash events
    :param time_scale: time scale length of data used for prediction in seconds, i.e. size of window
    :param rolling_step: length of rolling step in seconds
    :param avoid_duplicate: whether to check for duplicate windows before appending (time consuming)
    :return: tuples of valid start and end of valid windows, in seconds
    """
    # precondition: rolling step must be larger than this to avoid duplicate windows
    assert rolling_step >= C.MIN_STEP, "Rolling step length must be greater than {} seconds".format(C.MIN_STEP)

    # keep track of valid series of timestamps
    valid_windows = []

    # Sliding window cannot perform when 1) there are fewer than 2 data points, or
    # 2) entire input is shorter than given time scale
    if len(interval_series) <= 1 or interval_series.iloc[-1] - interval_series.iloc[0] <= time_scale:
        return valid_windows

    # initialize window boundary, the window has length of time_scale, which is identical to the crash event window.
    left_bound = interval_series.iloc[0]
    right_bound = left_bound + time_scale

    # Iterate over input series by rolling step to extract all possible time points within given scale.
    # Stop iteration once right bound is out of given interval.
    # for entry_index
    while right_bound <= interval_series.iloc[-1]:
        # extract window series, bounds inclusive
        window_series = interval_series[interval_series.between(left_bound, right_bound)]

        # only keep unique windows with more than 1 data points
        if len(window_series) >= C.MIN_ENTRIES_IN_WINDOW:
            # if specified, check for duplicate window to strictly avoid duplicate (time consuming)
            if avoid_duplicate:
                # append if windows list empty
                if not valid_windows:
                    valid_windows.append((window_series.iloc[0], window_series.iloc[-1]))
                # append if not a duplicate
                elif not any([window_series.equals(existing_window) for existing_window in valid_windows]):
                    valid_windows.append((window_series.iloc[0], window_series.iloc[-1]))
            else:
                valid_windows.append((window_series.iloc[0], window_series.iloc[-1]))

        # increment boundary
        left_bound += rolling_step
        right_bound += rolling_step

    return valid_windows


def interpolate_entries(entries,
                        sampling_rate=50,
                        cols_to_interpolate=None,
                        x_col="seconds") -> dict:
    """
    interpolate specified columns in given rows ordered by time
    :param entries: dataframe of entries ordered by time
    :param sampling_rate: data points after interpolation
    :param cols_to_interpolate: entry columns to interpolate, will use default if not set
    :param x_col: column used as x axis for all interpolation
    :return: dictionary of interpolated results, indexed by column name in entries
    """
    if cols_to_interpolate is None:
        cols_to_interpolate = C.COLS_TO_INTERPOLATE  # does not include seconds

    # get x axis for interpolating
    x = entries[x_col]
    x_sample = np.linspace(x.min(), x.max(), sampling_rate)

    # record interpolation result for each column. Each value entry has shape (50, )
    output = {col_name: sp.interpolate.interp1d(x, entries[col_name], kind='linear')(x_sample)
              for col_name in cols_to_interpolate}

    return output


def save_col(col_array: Union[list, np.ndarray], col_name: str, out_dir: str, expect_len=-1, dtype=None):
    """
    save array as numpy .npy file
    :param col_array: list or ndarray to be saved
    :param col_name: name of column to be saved, must be key in COL_PATHS
    :param out_dir: output directory to save object to
    :param expect_len: expected length of the input; if specified, error if different
    :param dtype: dtype for numpy array to be saved, used to save space
    """
    # check if col_name correct
    assert col_name in C.COL_PATHS, "Cannot recognize column name {} in COL_PATHS".format(col_name)

    # check length if needed
    if expect_len > -1:
        if len(col_array) != expect_len:
            raise ValueError("Column array length {} does not match expected length {}".format(
                len(col_array), expect_len))

    # convert list to array before saving
    if isinstance(col_array, list):
        col_array = np.array(col_array)

    # save to given output directory
    if dtype:
        np.save(os.path.join(out_dir, C.COL_PATHS[col_name]), col_array.astype(dtype))
    else:
        np.save(os.path.join(out_dir, C.COL_PATHS[col_name]), col_array)


def extract_destabilize(feature_matrix: np.ndarray) -> np.ndarray:
    """
    generate destabilization column based on the base feature matrix of (sampling_rate, 3).
    Destabilizing is defined as all 3 basic features 1) are non zero and 2) have same direction (i.e. sign)
    :param feature_matrix: feature matrix of original (velocity, position, joystick) tuple
    :return: a (sampling_rate, 1) column of boolean, indicating if row is destabilizing
    """
    # get signs of each element in matrix (0 will get nan).
    # Note that nan != nan, and this works with our definition since we don't destabilizing involves only non-zeros
    with np.errstate(divide="ignore", invalid="ignore"):
        signs = feature_matrix / np.abs(feature_matrix)

    # get booleans of whether all columns have same sign value as 1st (i.e. all same sign)
    # same_sign shape: (sampling_rate, 3)
    same_sign = np.equal(signs, signs[:, 0:1])

    # sum up booleans, only destabilizing when all 3 columns in a row are true (i.e. row sums to 3)
    # return shape: (sampling_rate,)
    num_feats = feature_matrix.shape[1]
    return np.equal(same_sign.sum(axis=1), num_feats)


def generate_feature_files(window_size: float,
                           time_ahead: float,
                           sampling_rate: int,
                           time_gap: float,
                           time_step: float,
                           out_dir: str) -> int:
    """
    Extract basic features columns from raw data and saving them to disk
    :author: Yonglin Wang, Jie Tang
    :param window_size: time length of data used for training, in seconds
    :param time_ahead: time in advance to predict, in seconds
    :param sampling_rate: sampling rate in each window
    :param time_gap: minimal length of time allowed between two crash events for sliding windows extraction
    :param time_step: the step to move window ahead in sliding window
    :param out_dir: output directory to save all features to
    :return: total number of samples generated
    """
    # record time used for preprocessing
    begin = time.time()

    # print training stats
    print("Feature generation settings: \n"
          "Window size: {}s\n"
          "Time ahead: {}s\n"
          "Sampling rate: {}\n"
          "Time gap: {}s\n"
          "Rolling step: {}s\n"
          "Output directory: {}".format(window_size,
                                        time_ahead,
                                        sampling_rate,
                                        time_gap,
                                        time_step,
                                        out_dir))

    # initial settings (to be moved)
    # np.set_printoptions(suppress=True)

    # ensure output folder exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # #### output and helper functions
    # lists to convert to np for save, each of length n_samples
    vel_ori_list = []
    vel_cal_list = []
    position_list = []
    joystick_list = []
    label_list = []
    trial_key_list = []
    start_list = []
    end_list = []

    def process_entries(entries_for_inter, trial_key: str, label: int):
        """helper function to interpolate entries and record output"""
        int_results = interpolate_entries(entries_for_inter, sampling_rate=sampling_rate)

        vel_ori_list.append(int_results["currentVelRoll"])
        vel_cal_list.append(int_results["calculated_vel"])
        position_list.append(int_results["currentPosRoll"])
        joystick_list.append(int_results["joystickX"])
        label_list.append(label)
        trial_key_list.append(trial_key)
        start_list.append(entries_for_inter['seconds'].iloc[0])
        end_list.append(entries_for_inter['seconds'].iloc[-1])

    # #### load raw data
    # extract all needed columns
    raw_data = pd.read_csv('data/data_all.csv',
                           usecols=['seconds', 'trialPhase', 'currentPosRoll', 'currentVelRoll', 'calculated_vel',
                                    'joystickX', 'peopleTrialKey'])

    # filter out non-human controls in data
    raw_data = raw_data[raw_data.trialPhase != 1]

    # get unique peopleTrialKeys that have crashes for skipping no-crash trials
    crash_keys_all = set(raw_data[raw_data.trialPhase == 4].peopleTrialKey.unique())

    # #### initialize auxiliary objects for debugging
    # record crash events that are not too short from the previous crash
    all_valid_crashes = pd.DataFrame()

    # record excluded crashes for validation analysis
    excluded_crashes_too_close = pd.DataFrame()
    excluded_crashes_too_few = pd.DataFrame()

    print("Total number of trials to process: {}".format(len(crash_keys_all)))

    # #### iterate through trials to process extract features
    with tqdm(total=raw_data.peopleTrialKey.nunique()) as pbar:
        # extract crash events features from each trial
        for current_trial_key, trial_raw_data in raw_data.groupby("peopleTrialKey"):
            # only process keys that has crashes
            if current_trial_key in crash_keys_all:
                # find all crash data points in this trial
                crashes_this_trial = trial_raw_data[trial_raw_data.trialPhase == 4]

                # Calculate each crash event's elapsed time since last crash (defined as difference since 0 for first
                # crash) Using assign to create new columns without evoking SettingWithCopyWarning
                crashes_this_trial = crashes_this_trial.assign(
                    preceding_crash_seconds=crashes_this_trial.seconds.shift(1, fill_value=0))
                crashes_this_trial = crashes_this_trial.assign(
                    seconds_since_last_crash=crashes_this_trial.seconds - crashes_this_trial.preceding_crash_seconds)

                # Keep only crash events longer than given time gap away since last (NOT sliding window yet!)
                valid_crash_entries = crashes_this_trial[crashes_this_trial["seconds_since_last_crash"] > time_gap]

                # record crash events this trial for later use
                all_valid_crashes = pd.concat([all_valid_crashes, valid_crash_entries])

                # For validation, include excluded crash events too
                invalid_crash_entries = crashes_this_trial[crashes_this_trial["seconds_since_last_crash"] <= time_gap]
                excluded_crashes_too_close = pd.concat([excluded_crashes_too_close, invalid_crash_entries])

                # iterate through each valid crash to create data entry
                for crash_time in valid_crash_entries.seconds:

                    # ### (1/2) Extract Crash Events in each group
                    # find corresponding data points between time scale start and crash event to generate training data
                    entries_for_train = trial_raw_data[trial_raw_data.seconds.between(
                        crash_time - window_size - time_ahead, crash_time - time_ahead)]

                    # only process entries with more than one data points
                    if len(entries_for_train) >= C.MIN_ENTRIES_IN_WINDOW:
                        # resample & interpolate
                        process_entries(entries_for_train, current_trial_key, 1)

                    else:
                        # print("Found a crash window with # of entries < {}!".format(MIN_ENTRIES_IN_WINDOW))
                        ex_crash = trial_raw_data[trial_raw_data.seconds == crash_time]
                        ex_crash["entries_since_last_crash"] = len(entries_for_train)
                        excluded_crashes_too_few = pd.concat([excluded_crashes_too_few, ex_crash])

                    # ### (2/2) Extract Noncrash Events in each group
                    # find bounds to perform sliding window
                    # left bound: last crash time of current valid crash, safe to use [0] since seconds are unique
                    left = \
                        valid_crash_entries["preceding_crash_seconds"].loc[
                            valid_crash_entries.seconds == crash_time].iloc[
                            0]
                    # right bound: crash interval ahead of current crash
                    right = crash_time - window_size - time_ahead

                    # crucially not include left boundary (last crash entry)
                    sliding_series = trial_raw_data.seconds[
                        (trial_raw_data.seconds > left) & (trial_raw_data.seconds <= right)]

                    # run sliding window on noncrash event
                    all_windows = sliding_window(sliding_series, window_size, time_step)

                    # only record list with more than 1 data points
                    if len(all_windows) >= 2:
                        # process each window
                        for win_start, win_end in all_windows:
                            # restore all window entries, bounds inclusive by default
                            entries_for_train = trial_raw_data[trial_raw_data.seconds.between(win_start, win_end)]
                            # resample & interpolate
                            process_entries(entries_for_train, current_trial_key, 0)

            # update progress bar
            pbar.update(1)

    # #### Save feature output
    print("Processing done! \nNow validating and saving features to \"{}\"...".format(out_dir), end="")

    # record expected length
    expected_length = len(label_list)
    # save column as .npy files, if disk is a concern in future, specify dtype in save_col
    [save_col(value, col_name, out_dir, expect_len=expected_length)
     for value, col_name in [(vel_ori_list, "velocity"), (vel_cal_list, "velocity_cal"),
                             (position_list, "position"), (joystick_list, "joystick"),
                             (label_list, "label"), (trial_key_list, "trial_key"),
                             (trial_key_list, "trial_key"), (start_list, "start_seconds"),
                             (end_list, "end_seconds")]]

    print("Done!\n")

    # #### For debugging
    # report crash event stats
    print("Total crashes in all raw data: {}\n"
          "{} crashes excluded due to following last crash in less than {}s\n"
          "{} crashes excluded due to having fewer than {} entries since last crash\n"
          "{} crashes included in training data\n".format(len(excluded_crashes_too_close) +
                                                          len(excluded_crashes_too_few) +
                                                          sum(label_list), len(excluded_crashes_too_close),
                                                          time_gap,
                                                          len(excluded_crashes_too_few), C.MIN_ENTRIES_IN_WINDOW,
                                                          sum(label_list)))

    # record excluded entries for analysis
    debug_base = C.DEBUG_FORMAT.format(int(time_ahead * 1000), int(window_size * 1000))
    excluded_crashes_too_close.to_csv(os.path.join(out_dir, "too_close_to_last_" + debug_base), index=False)
    excluded_crashes_too_few.to_csv(os.path.join(out_dir, "too_few_between_" + debug_base), index=False)
    all_valid_crashes.to_csv(os.path.join(out_dir, "all_valid_crashes_" + debug_base), index=False)

    print("Total crash samples: {}\n"
          "Total noncrash samples: {}\n"
          "Total sample size".format(label_list.count(1), label_list.count(0), len(label_list)))

    print("Feature generation done!")
    display_exec_time(begin, scr_name=__file__)

    return expected_length


def broadcast_to_sampled(arr: np.ndarray, arr_sampled: np.ndarray) -> np.ndarray:
    """
    append non-sampled array to a sampled array for generating training input
    :param arr: non-sampled array of (n, ) to broadcast to (n.sampling_rate)
    :param arr_sampled: sampled array of shape (n, sampling_rate, sampled_features)
    :return:
    combined array of (n, sampling_rate, sampled_features + 1), with broadcast array at the end of sampled entry
    """

    # broad cast new array to desired size, shape (n, sampling_rate)
    arr_aligned = np.broadcast_to(arr.reshape(-1, 1), arr_sampled.shape[:2])

    # append new data to sampled data and return
    return np.dstack(arr_sampled, arr_aligned)


if __name__ == "__main__":
    # noinspection PyTypeChecker
    argparser = argparse.ArgumentParser(prog="Data Pre-processing Argparser",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument(
        '--window', type=float, default=1.0,
        help='size of sliding window to generate non-crash training data, in seconds')
    argparser.add_argument(
        '--ahead', type=float, default=0.5, help='prediction timing ahead of event, in seconds')
    argparser.add_argument(
        '--rate', type=int, default=50, help='number of samples obtained per time scale extracted')
    argparser.add_argument(
        '--gap', type=int, default=5, help='minimal time gap allowed between two crash events for data extraction')
    argparser.add_argument(
        '--rolling', type=float, default=0.5, help='length of rolling time step of sliding window, in seconds')

    args = argparser.parse_args()

    print("Generating features...")
    print(f"current working directory: {os.getcwd()}")

    # check for minimal rolling step
    if args.rolling < C.MIN_STEP:
        raise argparse.ArgumentTypeError("Rolling step must be smaller than {}".format(C.MIN_STEP))

    # Ensure gap large enough to accommodate data extraction.
    # Time gap is used to exclude those consecutive crash events happened within less than this length.
    # When two crash events happened too closely, we could not generate enough data to for interpolation.
    # In principle: time gap >= non crashing time scale + crushing time scale + time ahead of predicted event
    if args.gap < (2 * args.window + args.ahead):
        args.gap = (2 * args.window + args.ahead)

    # generate feature files, for debug only
    generate_feature_files(window_size=args.window,
                           time_ahead=args.ahead,
                           sampling_rate=args.rate,
                           time_gap=args.gap,
                           time_step=args.rolling,
                           out_dir="data/default_test_{}window_{}ahead_{}rolling/".format(int(args.window * 1000),
                                                                                          int(args.ahead * 1000),
                                                                                          int(args.rolling * 1000)))
