#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yonglin Wang
# Date: 2021/1/27
# Functions for extracting and saving feature numpy arrays

import os
import time
from typing import Union

import pandas as pd
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedShuffleSplit

from tqdm import tqdm

import argparse
import random

from utils import calculate_exec_time
import consts as C
from processing.segment_raw_readings import load_segment_data

# control for randomness in case of any
np.random.seed(C.RANDOM_SEED)
random.seed(C.RANDOM_SEED)


def _extract_sliding_windows(interval_series: pd.Series, window_size: float, rolling_step: float, avoid_duplicate=False) -> list:
    """
    Identify extractable valid window boundaries (inclusive) in given slice of the seconds column.
    Valid window is defined as window that contains and only contains more than 1 non-crash events.
    :authors: Jie Tang, Yonglin Wang
    :param interval_series: series of time points between 2 crash events
    :param window_size: time scale length of data used for prediction in seconds, i.e. size of window
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
    if len(interval_series) <= 1 or interval_series.iloc[-1] - interval_series.iloc[0] <= window_size:
        return valid_windows

    # initialize window boundary, the window has length of time_scale, which is identical to the crash event window.
    left_bound = interval_series.iloc[0]
    right_bound = left_bound + window_size

    # Iterate over input series by rolling step to extract all possible time points within given scale.
    # Stop iteration once right bound is out of given interval.
    while right_bound <= interval_series.iloc[-1]:
        # extract window series, bounds inclusive
        window_series = interval_series[interval_series.between(left_bound, right_bound)]

        # only keep unique windows with more than 1 data points
        if len(window_series) >= C.MIN_ENTRIES_IN_WINDOW:
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


def save_col(col_array: Union[list, np.ndarray],
             col_name: str,
             out_dir: str,
             expect_len=-1, dtype=None):
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
    # same_sign shape: (sample_size, sampling_rate, 3)
    same_sign = np.equal(signs, signs[:, :, 0:1])

    # sum up booleans, only destabilizing when all 3 columns in a row are true (i.e. row sums to 3)
    # return shape: (sample_size, sampling_rate)
    num_feats = feature_matrix.shape[2]
    return np.equal(same_sign.sum(axis=2), num_feats)


def _clean_metadata(meta_df: pd.DataFrame) -> pd.DataFrame:
    """remove buggy data entry from the given segment metadata df, return the debugged df"""
    # remove human control segments
    meta_df = meta_df[meta_df.phase == 3]
    # locate buggy indices. current condition: 1-reading human segments and their corresponding crashes
    buggy_human_segs = meta_df[meta_df.reading_num <= C.MIN_ENTRIES_IN_WINDOW].index
    assert len(buggy_human_segs) == 16

    return meta_df.drop(meta_df[meta_df.index.isin(set(buggy_human_segs))].index)


def _process_window(output_arrays: dict, entries_for_inter, trial_key: str, label: int, sampling_rate: int):
    """helper function to interpolate entries in a window and record output"""
    int_results = interpolate_entries(entries_for_inter, sampling_rate=sampling_rate)

    output_arrays["vel_ori"].append(int_results["currentVelRoll"])
    output_arrays["vel_cal"].append(int_results["calculated_vel"])
    output_arrays["position"].append(int_results["currentPosRoll"])
    output_arrays["joystick"].append(int_results["joystickX"])
    output_arrays["label"].append(label)
    output_arrays["trial_key"].append(trial_key)
    output_arrays["person"].append(entries_for_inter["peopleName"].iloc[0])
    output_arrays["start_sec"].append(entries_for_inter['seconds'].iloc[0])
    output_arrays["end_sec"].append(entries_for_inter['seconds'].iloc[-1])


def generate_feature_files(window_size: float,
                           time_ahead: float,
                           sampling_rate: int,
                           time_gap: float,
                           time_step: float,
                           out_dir: str,
                           show_pbar=False) -> int:
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
    # ensure raw data file exists
    if not os.path.exists(C.RAW_DATA_PATH):
        raise FileNotFoundError("Raw data file cannot be found at {}".format(C.RAW_DATA_PATH))

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

    assert time_gap >= window_size + time_ahead, "Gap too short"

    # ensure output folder exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # #### output and helper functions
    # dict of nested lists to convert to np for save, each of length n_samples
    feat_arrays = {k: [] for k in ("vel_ori", "vel_cal", "position", "joystick", "label", "trial_key", "person", "start_sec", "end_sec")}

    # #### load and prepare raw data
    # extract all needed columns
    raw_data = pd.read_csv(C.RAW_DATA_PATH, usecols=C.ESSENTIAL_RAW_COLS)

    # filter out non-human controls in data
    raw_data = raw_data[raw_data.trialPhase != 1]

    # load preprocessed segment metadata (will create if not exist)
    meta_df, seg_dict = load_segment_data(C.RAW_DATA_PATH, C.SEGMENT_DICT_PATH, C.SEGMENT_STATS_PATH, verbose=show_pbar)

    # clean metadata df, outputs only non-bug human segments for further selection
    meta_df = _clean_metadata(meta_df)

    # ### filter out crashes based on dataset params
    # decide which human segments to include based on gap
    # count of human segs before selection
    human_seg_count = {"orig_non": sum(meta_df.crash_ind==-1),
                           "orig_crash": sum(meta_df.crash_ind!=-1)}
    # keep only seg_duration >= gap
    excluded_segments = meta_df[meta_df.duration <= time_gap]
    meta_df = meta_df[meta_df.duration > time_gap]

    human_seg_count.update({"new_non": sum(meta_df.crash_ind == -1),
                           "new_crash": sum(meta_df.crash_ind != -1)})

    # print how many crashed human segments excluded
    print(f"Human control segments:\n"
          f"{human_seg_count['new_non']} out of {human_seg_count['orig_non']} non-crashed human control segments "
          f"selected, {human_seg_count['orig_non'] - human_seg_count['new_non']} excluded\n"
          f"{human_seg_count['new_crash']} out of {human_seg_count['orig_crash']} crashed human control segments "
          f"selected, {human_seg_count['orig_crash'] - human_seg_count['new_crash']} excluded\n")

    # ### create index dictionary for faster retrieval later from raw data
    # 1. dict of {<trial key>: [crash segment ids]}. containing valid crashed segments in each trial
    # 2. dict of {<trial key>: [non-crash segment ids]}. containing valid non-crash segments in each trial

    valid_crashed_ids, valid_non_crashed_ids = dict(), dict()
    for trial_key, segments in meta_df.groupby("trial_key"):
        valid_crashed_ids[trial_key] = segments[segments.crash_ind != -1].index.tolist()
        valid_non_crashed_ids[trial_key] = segments[segments.crash_ind == -1].index.tolist()

    # #### iterate through trials to process extract features
    with tqdm(total=raw_data.peopleTrialKey.nunique(), disable=not show_pbar) as pbar:
        # extract crash events features from each trial
        for current_trial_key, trial_raw_data in raw_data.groupby("peopleTrialKey"):
            # for trial key, for each crashed segment in this trial (if any), record crash and non crash
            for crashed_id in valid_crashed_ids[current_trial_key]:
                # 1. find corresponding window of the crash and interpolate.
                # extract entries within crash window
                crash_time = trial_raw_data.loc[seg_dict[crashed_id]["crash_ind"]].seconds
                crash_window_df = trial_raw_data[trial_raw_data.seconds.between(
                    crash_time - window_size - time_ahead, crash_time - time_ahead)]
                _process_window(feat_arrays, crash_window_df, current_trial_key, 1, sampling_rate)
                # 2. generate sliding windows of the segment as noncrash samples and interpolate.
                seg_df = trial_raw_data.loc[seg_dict[crashed_id]["indices"]]
                # calculate sliding window bounds; left: start of segment, right: right before the time-ahead range
                left_bound = seg_dict[crashed_id]["start_sec"]
                right_bound = crash_time - time_step

                # extract and record sliding windows
                window_bounds = _extract_sliding_windows(seg_df[seg_df.seconds.between(left_bound, right_bound)].seconds,
                                                         window_size, time_step)
                # process each window
                for win_start, win_end in window_bounds:
                    # restore all window entries, bounds inclusive by default
                    entries_in_win = seg_df[seg_df.seconds.between(win_start, win_end)]
                    # resample & interpolate
                    _process_window(feat_arrays, entries_in_win, current_trial_key, 0, sampling_rate)

            # for the non-crashed segment in this trial (if any), do sliding window and interpolate non crash data
            for non_crashed_id in valid_non_crashed_ids[current_trial_key]:
                seg_df = trial_raw_data.loc[seg_dict[non_crashed_id]["indices"]]
                # get sliding window bounds
                left_bound = seg_dict[non_crashed_id]["start_sec"]
                right_bound = seg_dict[non_crashed_id]["end_sec"]
                # extract and record sliding windows
                window_bounds = _extract_sliding_windows(seg_df[seg_df.seconds.between(left_bound, right_bound)].seconds,
                                                         window_size, time_step)
                # process each window
                for win_start, win_end in window_bounds:
                    # restore all window entries, bounds inclusive by default
                    entries_in_win = seg_df[seg_df.seconds.between(win_start, win_end)]
                    # resample & interpolate
                    _process_window(feat_arrays, entries_in_win, current_trial_key, 0, sampling_rate)

            pbar.update(1)

    # #### Save feature output
    print("Processing done! \nNow validating and saving features to \"{}\"...".format(out_dir), end="")

    # record expected length
    expected_length = len(feat_arrays["label"])

    # split data into train and test based on label
    _save_test_train_split(feat_arrays["label"], out_dir)

    # save column as .npy files, if disk is a concern in future, specify dtype in save_col
    [save_col(value, col_name, out_dir, expect_len=expected_length)
     for value, col_name in [(feat_arrays["vel_ori"], "velocity"),
                             (feat_arrays["vel_cal"], "velocity_cal"),
                             (feat_arrays["position"], "position"),
                             (feat_arrays["joystick"], "joystick"),
                             (feat_arrays["label"], "label"),
                             (feat_arrays["person"], "person"),
                             (feat_arrays["trial_key"], "trial_key"),
                             (feat_arrays["start_sec"], "start_seconds"),
                             (feat_arrays["end_sec"], "end_seconds")]]
    print("Done!\n")

    # record excluded entries for analysis
    debug_base = C.DEBUG_FORMAT.format(int(time_ahead * 1000), int(window_size * 1000))
    excluded_segments.to_csv(os.path.join(out_dir, debug_base), index=False)

    # print processing results
    crash_total = feat_arrays["label"].count(1)
    noncrash_total = feat_arrays["label"].count(0)
    print("Total crash samples: {}\n"
          "Total noncrash samples: {}\n"
          "Total sample size: {}".format(crash_total, noncrash_total, expected_length))

    print("Feature generation done!")
    calculate_exec_time(begin, scr_name=__file__)

    return expected_length


def _save_test_train_split(y_labels: list, out_dir:str):
    """save stratified test vs. train+val splits"""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=C.RANDOM_SEED)
    train_inds, test_inds = list(sss.split(np.zeros(len(y_labels)), y_labels))[0]
    # save split indices
    np.save(os.path.join(out_dir, C.INDS_PATH['train']), np.array(train_inds))
    np.save(os.path.join(out_dir, C.INDS_PATH['test']), np.array(test_inds))


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
    exp_len = generate_feature_files(0.3, 0.5, 50, 0.8, 0.7, "data/300window_500ahead_700rolling", show_pbar=True)
