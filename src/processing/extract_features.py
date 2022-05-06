#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yonglin Wang
# Date: 2021/1/27
# Functions for extracting and saving feature numpy arrays

import os
import time
from typing import Union
import random


import pandas as pd
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from tqdm import tqdm


from processing.split_data import _save_test_train_split
from utils import calculate_exec_time
import consts as C
from processing.episode_raw_readings import load_episode_data
from recording.log import init_logger

# control for randomness in case of any
np.random.seed(C.RANDOM_SEED)
random.seed(C.RANDOM_SEED)


def _extract_sliding_windows(interval_series: pd.Series, window_size: float, rolling_step: float) -> list:
    """
    Identify extractable valid window boundaries (inclusive) in given slice of the seconds column.
    1st window aligns with interval start, while last is last one possible before window end exceeds interval end.
    :authors: Jie Tang, Yonglin Wang
    :param interval_series: series of time points between 2 crash events
    :param window_size: time scale length of data used for prediction in seconds, i.e. size of window
    :param rolling_step: length of rolling step in seconds
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

    output["seconds"] = x_sample

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


def extract_destabilize(feature_matrix: np.ndarray, single_entry=False) -> np.ndarray:
    """
    generate destabilization column based on the base feature matrix of (sampling_rate, 3).
    Destabilizing is defined as all 3 basic features 1) are non zero and 2) have same direction (i.e. sign)
    :param feature_matrix: feature matrix of original (velocity, position, joystick) tuple
    :param single_entry: whether the feature matrix is single entry, of shape (sampling_rate, 3), or multi entry,
    of shape (sample_size, sampling_rate, 3)
    :return: a (sampling_rate, 1) column of boolean, indicating if row is destabilizing
    """
    # get signs of each element in matrix (0 will get nan).
    # Note that nan != nan, and this works with our definition since we don't destabilizing involves only non-zeros
    with np.errstate(divide="ignore", invalid="ignore"):
        signs = feature_matrix / np.abs(feature_matrix)

    if single_entry:
        # get booleans of whether all columns have same sign value as 1st (i.e. all same sign)
        # same_sign shape: (sampling_rate, 3)
        same_sign = np.equal(signs, signs[:, 0:1])
        # sum up booleans, only destabilizing when all 3 columns in a row are true (i.e. row sums to 3)
        num_feats = feature_matrix.shape[1]
        # return shape: (sampling_rate,)
        return np.equal(same_sign.sum(axis=1), num_feats)
    else:
        # get booleans of whether all columns have same sign value as 1st (i.e. all same sign)
        # same_sign shape: (sample_size, sampling_rate, 3)
        same_sign = np.equal(signs, signs[:, :, 0:1])
        # sum up booleans, only destabilizing when all 3 columns in a row are true (i.e. row sums to 3)
        num_feats = feature_matrix.shape[2]
        # return shape: (sample_size, sampling_rate)
        return np.equal(same_sign.sum(axis=2), num_feats)


def _process_window(output_arrays: dict, entries_for_inter, trial_key: str,
                    crash_cutoff: Union[float, np.float], sampling_rate: int,
                    episode_id: int) -> None:
    """
    helper function to interpolate entries in a window and record output
    :param output_arrays:
    :param entries_for_inter:
    :param trial_key:
    :param crash_cutoff: cutoff time in seconds, time steps beyond which has the a corresponding seq label of 1
    :param sampling_rate:
    :return:
    """

    int_results = interpolate_entries(entries_for_inter, sampling_rate=sampling_rate)

    # record features
    output_arrays["vel_ori"].append(int_results["currentVelRoll"])
    output_arrays["vel_cal"].append(int_results["calculated_vel"])
    output_arrays["position"].append(int_results["currentPosRoll"])
    output_arrays["joystick"].append(int_results["joystickX"])
    output_arrays["trial_key"].append(trial_key)
    output_arrays["person"].append(entries_for_inter["peopleName"].iloc[0])
    output_arrays["start_sec"].append(entries_for_inter['seconds'].iloc[0])
    window_end = entries_for_inter['seconds'].iloc[-1]
    output_arrays["end_sec"].append(window_end)
    output_arrays["episode_id"].append(episode_id)

    # record labels
    seq_labels = np.zeros(sampling_rate)
    single_label = 0
    if window_end >= crash_cutoff:
        # If window touches or crosses cutoff, there is at least one time step that has label 1.
        # First, retrieve interpolated seconds for locating cutoff.
        seconds = int_results["seconds"]
        # Then, any time step that touches or crosses cutoff receives a label of 1
        assert seq_labels.shape[0] == seconds.shape[0], "Length mismatch between interpolated sequence length and seconds"
        seq_labels[seconds >= crash_cutoff] = 1
        assert np.sum(seq_labels) > 0, "no 1 labels assigned!"
        # lastly, set single label to 1 to signal that there is a crash within time ahead
        single_label = 1

    # now, record both single and seq label
    output_arrays["label"].append(single_label)
    output_arrays["seq_label"].append(seq_labels)


def generate_feature_files(window_size: float,
                           time_ahead: float,
                           sampling_rate: int,
                           time_gap: float,
                           time_step: float,
                           out_dir: str,
                           show_pbar: bool=False,
                           test_only_dataset: bool=False) -> int:
    """
    Extract basic features columns from raw data and saving them to disk; main function of this script
    :author: Yonglin Wang
    :param window_size: time length of data used for training, in seconds
    :param time_ahead: time in advance to predict, in seconds
    :param sampling_rate: sampling rate in each window
    :param time_gap: minimal length of time allowed between two crash events for sliding windows extraction
    :param time_step: the step to move window ahead in sliding window
    :param out_dir: output directory to save all features to
    :param test_only_dataset: if True, all of the windows extracted will be test data, and none will be training data
    :return: total number of samples generated
    """
    # ensure output folder exists
    if not os.path.exists(out_dir):
        # if race condition happens, the line below SHOULD raise exception (i.e. do NOT set exists_OK=True)
        os.makedirs(out_dir, exist_ok=False)

    # get logger to log information
    logger = init_logger(__name__, out_dir, C.EXTRACT_LOG_NAME)

    # ensure raw data file exists
    if not os.path.exists(C.RAW_DATA_PATH):
        raise FileNotFoundError("Raw data file cannot be found at {}".format(C.RAW_DATA_PATH))

    # record time used for preprocessing
    begin = time.time()

    # print training stats
    logger.info("Feature generation settings: \n"
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

    assert time_ahead > time_step, f"Lookahead time {time_ahead} should be larger than rolling step {time_step}; " \
                                   f"otherwise some crashed episodes may not have crashed windows."

    # #### output and helper functions
    # dict of int lists to convert to np for save, each of length n_samples;
    # not including later inferred features here (e.g. destab)
    feat_arrays = {k: [] for k in ("vel_ori", "vel_cal", "position", "joystick",
                                   "label", "trial_key", "person", "start_sec", "end_sec", "seq_label", "episode_id")}

    # #### load and prepare raw data
    # extract all needed columns
    raw_data = pd.read_csv(C.RAW_DATA_PATH, usecols=C.ESSENTIAL_RAW_COLS)

    # filter out non-human controls in data
    raw_data = raw_data[raw_data.trialPhase != 1]

    # load preprocessed episode metadata (will create if not exist);
    # 1 episode = human control starts -> crash/trial end
    meta_df, episode_dict = load_episode_data(C.RAW_DATA_PATH, C.SEGMENT_DICT_PATH, C.SEGMENT_STATS_PATH,
                                              clean_data=True, verbose=show_pbar)

    # ### filter out crashes based on dataset params, decide which episodes to include based on gap
    # count of human segs before selection, for display only
    human_epi_count = {"orig_non": sum(meta_df.crash_ind==-1),
                           "orig_crash": sum(meta_df.crash_ind!=-1)}
    # keep only episode_duration > gap
    excluded_episodes = meta_df[meta_df.duration <= time_gap]
    meta_df = meta_df[meta_df.duration > time_gap]
    human_epi_count.update({"new_non": sum(meta_df.crash_ind == -1),
                           "new_crash": sum(meta_df.crash_ind != -1)})

    # print how many crashed human episodes excluded
    logger.info(f"Human control episodes:\n"
          f"{human_epi_count['new_non']} out of {human_epi_count['orig_non']} non-crashed human control episodes "
          f"selected, {human_epi_count['orig_non'] - human_epi_count['new_non']} excluded\n"
          f"{human_epi_count['new_crash']} out of {human_epi_count['orig_crash']} crashed human control episodes "
          f"selected, {human_epi_count['orig_crash'] - human_epi_count['new_crash']} excluded\n")

    # ### create index dictionary for faster retrieval later from raw data
    # valid_crashed_ids: dict of {<trial key>: [crash episode ids]}. containing valid crashed episodes in each trial
    # valid_non_crashed_ids: dict of {<trial key>: [non-crash episode ids]}. containing valid non-crash episodes in each trial
    valid_crashed_ids, valid_non_crashed_ids = dict(), dict()
    for trial_key, episodes in meta_df.groupby("trial_key"):
        valid_crashed_ids[trial_key] = episodes[episodes.crash_ind != -1].index.tolist()
        valid_non_crashed_ids[trial_key] = episodes[episodes.crash_ind == -1].index.tolist()

    # #### iterate through trials to process extract features
    with tqdm(total=raw_data.peopleTrialKey.nunique(), disable=not show_pbar) as pbar:
        # extract crash events features from each trial
        for current_trial_key, trial_raw_data in raw_data.groupby("peopleTrialKey"):
            # for trial key, for each crashed episode in this trial (if any), record crash and non crash windows
            if current_trial_key in valid_crashed_ids:
                for crashed_id in valid_crashed_ids[current_trial_key]:
                    # time point of crash
                    crash_time = trial_raw_data.loc[episode_dict[crashed_id]["crash_ind"]].seconds
                    # cut off time, time steps on or right of this time receives label 1; else 0
                    crash_cutoff = crash_time - time_ahead

                    # extract windows boundaries
                    seg_df = trial_raw_data.loc[episode_dict[crashed_id]["indices"]]
                    seg_windows = _extract_sliding_windows(seg_df.seconds, window_size, time_step)

                    # process each window
                    for win_start, win_end in seg_windows:
                        # restore all window entries, bounds inclusive by default
                        entries_in_win = seg_df[seg_df.seconds.between(win_start, win_end)]
                        # interpolate the window; feature arrays will be updated in place
                        _process_window(feat_arrays, entries_in_win, current_trial_key, crash_cutoff, sampling_rate, crashed_id)

            # for the non-crashed episode in this trial (if any), do sliding window and interpolate non crash data
            if current_trial_key in valid_non_crashed_ids:
                for non_crashed_id in valid_non_crashed_ids[current_trial_key]:
                    # get entries of this episode
                    seg_df = trial_raw_data.loc[episode_dict[non_crashed_id]["indices"]]
                    # extract and record sliding windows
                    seg_windows = _extract_sliding_windows(seg_df.seconds, window_size, time_step)
                    # process each window
                    for win_start, win_end in seg_windows:
                        # restore all window entries, bounds inclusive by default
                        entries_in_win = seg_df[seg_df.seconds.between(win_start, win_end)]
                        # interpolate and process window; cutoff of inf -> labels all 0;
                        # feature arrays will be updated in place
                        _process_window(feat_arrays, entries_in_win, current_trial_key, np.inf, sampling_rate, non_crashed_id)

            pbar.update(1)

    # #### Save feature output
    logger.info("Processing done! \nNow validating and saving features to \"{}\"...".format(out_dir))

    # record expected length
    expected_length = len(feat_arrays["label"])

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
                             (feat_arrays["end_sec"], "end_seconds"),
                             (feat_arrays["seq_label"], "seq_label"),
                             (feat_arrays["episode_id"], "episode_id")]]

    # split data into train and test based on label
    train_inds, test_inds = _save_test_train_split(feat_arrays["episode_id"], out_dir,
                                                   valid_non_crashed_ids, valid_crashed_ids,
                                                   test_only_dataset=test_only_dataset)

    logger.info("File generation done!\n")

    # record excluded entries for analysis
    debug_base = C.DEBUG_FORMAT.format(int(time_ahead * 1000), int(window_size * 1000))
    excluded_episodes.to_csv(os.path.join(out_dir, debug_base), index=False)

    # print processing results
    crash_total = feat_arrays["label"].count(1)
    noncrash_total = feat_arrays["label"].count(0)

    # display train-test split statistics
    y_label = np.array(feat_arrays["label"])
    # TODO WHY len(test_y) only 565 samples when expected_length is 195031?
    train_y = y_label[train_inds]
    test_y = y_label[test_inds]

    if test_only_dataset:
        train0, train1 = 0, 0
    else:
        _, (train0, train1) = np.unique(train_y, return_counts=True)

    _, (test0, test1) = np.unique(test_y, return_counts=True)

    logger.info("Total crash samples: {}\n"
                "Total noncrash samples: {}\n"
                "Total sample size: {}".format(crash_total, noncrash_total, expected_length))
    if test_only_dataset:
        logger.info("This is a test-only dataset. Training stats are not applicable.")
    else:
        logger.info(f"In training set: \n"
                    f"Noncrash samples: {train0}\n"
                    f"Crash samples: {train1}\n"
                    f"Total training set size: {len(train_y)}\n"
                    f"Training set 0:1 ratio: {train0/train1 if not test_only_dataset else 'NA'}\n"
                    )
    logger.info(f"In test set:\n"
                f"Noncrash samples: {test0}\n"
                f"Crash samples: {test1}\n"
                f"Total test set size: {len(test_y)}\n"
                f"Test set 0:1 ratio: {test0/test1}\n")

    logger.info("Feature generation done!")

    time_taken = calculate_exec_time(begin, scr_name=__file__)
    logger.info("Total time taken: {}".format(time_taken))

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
    # why 2000 below ends up with "KeyError: '1_as_P31/01_600back_Block1_trial_001.csv'"???
    exp_len = generate_feature_files(1.0, 1.0, 50, 3.0, 0.1, "data/1000window_1000ahead_100rolling", show_pbar=True)
