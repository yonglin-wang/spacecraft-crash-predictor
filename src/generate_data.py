import os
import glob
import pandas as pd
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from datetime import timedelta
# import matplotlib.pyplot as plt
import warnings
from pandas.core.common import SettingWithCopyWarning
import tqdm

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import argparse
import random

# control for randomness in case of any
RANDOM_SEED = 2020
np.random.seed(seed=RANDOM_SEED)
random.seed(RANDOM_SEED)

# argparser value checker
MIN_STEP = 0.04

# crash event criteria
MIN_ENTRIES_IN_WINDOW = 2     # minimum # of entries between two crash events (i.e. in a window)

# paths for saving output
OUT_DIR_FORMAT = "data/data_{}ms/"
CRASH_FILE_FORMAT = "crash_feature_label_{}ahead_{}scale_test"
NONCRASH_FILE_FORMAT = "noncrash_feature_label_{}ahead_{}scale_test"
DEBUG_EXCLUDE_FORMAT = "exclude_{}ahead_{}scale_test.csv"

# columns we will use for interpolation
COLS_TO_INTERPOLATE = ('currentVelRoll', 'currentPosRoll', 'calculated_vel', 'joystickX')
OUT_COLS_AFTER_INTERPOLATE = ("features_cal_vel", "features_org_vel", 'label', 'peopleTrialKey',
                              'start_seconds', 'end_seconds')


####### define my own sliding window function
def sliding_window(interval_series:pd.Series, time_scale:float, rolling_step:float, avoid_duplicate=False) -> list:
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
    if rolling_step < MIN_STEP:
        raise ValueError("Rolling step length must be greater than {} seconds".format(MIN_STEP))

    # keep track of valid series of timestamps
    valid_windows = []

    # Sliding window cannot perform when 1) there are fewer than 2 data points, or
    # 2) entire input is shorter than given time scale
    if len(interval_series) <= 1 or interval_series.iloc[-1] - interval_series.iloc[0] <= time_scale:
        return valid_windows

    # initialize window boundary, the window has length of time_scale, which is identical to the crash event window.
    left_bound = interval_series.iloc[0]
    right_bound = left_bound + time_scale

    # entry_index = 0

    # Iterate over input series by rolling step to extract all possible time points within given scale.
    # Stop iteration once right bound is out of given interval.
    # for entry_index
    while right_bound <= interval_series.iloc[-1]:
        # extract window series, bounds inclusive
        window_series = interval_series[interval_series.between(left_bound, right_bound)]

        # only keep unique windows with more than 1 data points
        if len(window_series) >= MIN_ENTRIES_IN_WINDOW:
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
                        x_col="seconds")->dict:
    """
    interpolate specified columns in given rows ordered by time
    :param entries: dataframe of entries ordered by time
    :param sampling_rate: data points after interpolation
    :param cols_to_interpolate:
    :param x_col:
    :return:
    """
    if cols_to_interpolate is None:
        cols_to_interpolate = COLS_TO_INTERPOLATE  # does not include seconds

    # get x axis for interpolating
    x = entries[x_col]
    x_sample = np.linspace(x.min(), x.max(), sampling_rate)

    # record interpolation result for each column. Each value entry has shape (50, )
    output = {col_name: sp.interpolate.interp1d(x, entries[col_name], kind='linear')(x_sample)
              for col_name in cols_to_interpolate}

    return output

def assemble_feature_row(window_data)->pd.DataFrame:
    """
    generate row of features for concatenation and feature extraction, given a window of raw data
    If new features are added, add their extraction output here
    :param window_data: DataFrame containing rows of raw data in a window
    :return: row containing processed features
    """
    pass

def get_basic_feature_matrix(int_results:dict,
                       sampling_rate:int,
                       use_cal=False,
                       )->np.ndarray:
    """
    reads results from interpolate_entries and generate a feature matrix of (sampling_rate, feature_num).
    Each row has 3 features: velocity, position, joystick
    :param int_results: dictionary results from interpolate_entries
    :param destabilize: whether to include destabilizing joystick movement
    :return: feature matrix of (sampling_rate, feature_num)
    """


    if use_cal:
        # use calculated velocity
        new_vel = int_results["calculated_vel"]
    else:
        # use original velocity
        new_vel = int_results["currentVelRoll"]

    new_currentPosRoll = int_results["currentPosRoll"]
    new_joystickX = int_results["joystickX"]

    # output the basic rows of triples
    output = np.dstack([new_vel, new_currentPosRoll, new_joystickX]).reshape(sampling_rate, 3)

    return output

def extract_destabilize(feature_matrix: np.ndarray)->np.ndarray:
    """
    generate destabilization column based on the base feature matrix of (sampling_rate, 3).
    Destabilizing is defined as all 3 basic features 1) are non zero and 2) have same direction (i.e. sign)
    :param feature_matrix: feature matrix of original (velocity, position, joystick) tuple
    :return: a (sampling_rate, 1) column of boolean, indicating if row is destabilizing
    """
    # get signs of each element in matrix (0 will get nan).
    # Note that nan != nan, and this works with our definition since we don't destabilizing involves only non-zeros
    with np.errstate(divide="ignore"):
        signs = feature_matrix/np.abs(feature_matrix)

    # get booleans of whether all columns have same sign value as 1st (i.e. all same sign)
    # same_sign shape: (sampling_rate, 3)
    same_sign = np.equal(signs, signs[:, 0:1])

    # sum up booleans, only destabilizing when all 3 columns in a row are true (i.e. row sums to 3)
    # return shape: (sampling_rate,)
    num_feats = feature_matrix.shape[1]
    return np.equal(same_sign.sum(axis=1), num_feats)

def preprocess_data(window_size:float,
                    time_ahead:float,
                    sampling_rate:int,
                    time_gap:float,
                    time_step:float,
                    out_dir:str):
    """
    Extract basic features columns from raw data and saving them to disk
    :param window_size: time length of data used for training, in seconds
    :param time_ahead: time in advance to predict, in seconds
    :param sampling_rate: sampling rate in each window
    :param time_gap: minimal length of time allowed between two crash events for sliding windows extraction
    :param time_step: the step to move window ahead in sliding window
    :param out_dir: output directory to save all features to
    """


    '''
    author: Jie Tang， Yonglin Wang
    inputs:
        time_scale_train: time length of data used for training, in seconds
        time_ahead: time in advance to predict, in seconds
        sampling_rate: sampling rate in each time_scale_train
        time_gap: minimal length of time allowed between two crash events for sliding windows extraction

        time_step: the step we will move when we do sliding window
    '''
    # initial settings (to be moved)
    np.set_printoptions(suppress=True)
    #
    # # convert time unit and get output paths
    # ahead_ms_str = str(int(time_ahead * 1000))
    # scale_ms_str = str(int(time_scale_train * 1000))

    # crash_pickle_path = os.path.join(OUT_DIR_FORMAT.format(scale_ms_str), CRASH_FILE_FORMAT.format(ahead_ms_str, scale_ms_str))
    # noncrash_pickle_path = os.path.join(OUT_DIR_FORMAT.format(scale_ms_str), NONCRASH_FILE_FORMAT.format(ahead_ms_str, scale_ms_str))

    # ensure output folder exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # getting data
    # data_all = pd.read_csv('data/data_all.csv')

    ##### extract all needed columns ######
    cols_used = ['seconds', 'trialPhase', 'currentPosRoll', 'currentVelRoll', 'calculated_vel',
                  'joystickX', 'peopleName', 'trialName', 'peopleTrialKey']

    raw_data = pd.read_csv('data/data_all.csv', usecols=cols_used)
    # raw_data['datetimeNew'] = pd.to_datetime(raw_data['datetimeNew'])
    # filter out non-human controls in data
    raw_data = raw_data[raw_data.trialPhase != 1]
    # raw_data.set_index('datetimeNew', inplace=True)

    # find all crash controls by trial phase
    crash_data = raw_data[raw_data.trialPhase == 4]

    # get unique peopleTrialKeys that have crashes
    crash_keys_all = set(crash_data.peopleTrialKey.unique())


    # record crash events that are not too short from the previous crash
    all_valid_crashes = pd.DataFrame()

    # record excluded crashes
    excluded_crashes_too_close = pd.DataFrame()
    excluded_crashes_too_few = pd.DataFrame()

    # keep track of valid crash events for validation
    crashes_in_training = 0

    # generate feature and save
    crash_feature_label_df = pd.DataFrame()

    # lists to convert to np for save, each of length n_samples
    vel_ori_list = []
    vel_cal_list = []
    position_list = []
    joystick_list = []
    label_list = []
    trial_key_list = []
    start_list = []
    end_list = []

    def process_entries(entries_for_train, trial_key:str, label:int):
        """helper function to interpolate entries and record output"""
        int_results = interpolate_entries(entries_for_train, sampling_rate=50)

        vel_ori_list.append(int_results["currentVelRoll"])
        vel_cal_list.append(int_results["calculated_vel"])
        position_list.append(int_results["currentPosRoll"])
        joystick_list.append(int_results["joystickX"])
        label_list.append(label)
        trial_key_list.append(trial_key)
        start_list.append(entries_for_train['seconds'].iloc[0])
        end_list.append(entries_for_train['seconds'].iloc[-1])

    # extract crash events features from each trial
    for current_trial_key, trial_raw_data in raw_data.groupby("peopleTrialKey"):
        # only process keys that has crashes
        if current_trial_key in crash_keys_all:
            # find all crash data points in this trial
            crashes_this_trial = trial_raw_data[trial_raw_data.trialPhase == 4]

            # Calculate each crash event's elapsed time since last crash (defined as difference since 0 for first crash)
            crashes_this_trial["preceding_crash_seconds"] = crashes_this_trial.seconds.shift(1, fill_value=0)
            crashes_this_trial["seconds_since_last_crash"] = crashes_this_trial.seconds - crashes_this_trial.preceding_crash_seconds

            # Keep only crash events longer than given time gap away since last (NOT sliding window yet!)
            valid_crash_entries = crashes_this_trial[crashes_this_trial["seconds_since_last_crash"] > time_gap]

            # record crash events this trial for later use
            all_valid_crashes = pd.concat([all_valid_crashes, valid_crash_entries])

            # For validation, include excluded crash events too
            invalid_crash_entries = crashes_this_trial[crashes_this_trial["seconds_since_last_crash"] <= time_gap]
            excluded_crashes_too_close = pd.concat([excluded_crashes_too_close, invalid_crash_entries])

            # iterate through each valid crash to create data entry
            for crash_time in valid_crash_entries.seconds:

                # (1/2) Extract Crash Events in each group
                # find corresponding data points between time scale start and crash event to generate training data
                entries_for_train = trial_raw_data[trial_raw_data.seconds.between(
                    crash_time - window_size - time_ahead, crash_time - time_ahead)]

                # only process entries with more than one data points
                if len(entries_for_train) >= MIN_ENTRIES_IN_WINDOW:
                    # resample & interpolate
                    process_entries(entries_for_train, current_trial_key, 1)

                else:
                    print("Found a crash window with # of entries < {}!".format(MIN_ENTRIES_IN_WINDOW))
                    ex_crash = trial_raw_data[trial_raw_data.seconds == crash_time]
                    ex_crash["entries_since_last_crash"] = len(entries_for_train)
                    excluded_crashes_too_few = pd.concat([excluded_crashes_too_few, ex_crash])
                    # # Get copy of current len = 1 or 0 entry
                    # lone_entry_copy = entries_for_train.copy()

                # (2/2) Extract Noncrash Events in each group

                # find bounds to perform sliding window
                # left bound: last crash time of current valid crash, safe to use [0] since seconds are unique
                left = valid_crash_entries["preceding_crash_seconds"].loc[valid_crash_entries.seconds == crash_time][0]
                # right bound: crash interval ahead of current crash
                right = crash_time - window_size - time_ahead

                # crucially not include left boundary (last crash entry)
                sliding_series = trial_raw_data.seconds[(trial_raw_data.seconds > left) & (trial_raw_data.seconds <= right)]

                # run sliding window on noncrash event
                all_windows = sliding_window(sliding_series, window_size, time_step)

                # only record list with more than 1 data points
                if len(all_windows) >= 2:
                    for win_start, win_end in all_windows:
                        # restore all window entries
                        entries_for_train = trial_raw_data[trial_raw_data.seconds.between(win_start, win_end)]

                        # resample & interpolate
                        process_entries(entries_for_train, current_trial_key, 0)

    # report crash event stats
    print("Total crashes in all raw data: {}\n"
          "{} crashes excluded due to following last crash in less than {}s\n"
          "{} crashes excluded due to having fewer than {} entries since last crash\n"
          "{} crashes included in training data".format(
        len(excluded_crashes_too_close) + len(excluded_crashes_too_few) + len(crash_feature_label_df),
        len(excluded_crashes_too_close), time_gap,
        len(excluded_crashes_too_few), MIN_ENTRIES_IN_WINDOW,
        len(crash_feature_label_df)))

    # record excluded entries for analysis TODO one to csv function?
    excluded_crashes_too_close.to_csv(os.path.join(out_dir, "too_close_to_last" + DEBUG_EXCLUDE_FORMAT.format(int(time_ahead*1000), scale_ms_str)))
    excluded_crashes_too_few.to_csv(os.path.join(out_dir, "too_few_between" + DEBUG_EXCLUDE_FORMAT.format(int(time_ahead*1000), scale_ms_str)))

    # for trial_key in crash_keys_all:
    #     # First collect all valid
    #     # Collect all crashes for given person and trial
    #     trial_data = crash_data[crash_data.peopleTrialKey == trial_key]
    #     # Calculate each crash event's elapsed time since last crash (defined as difference since 0 for first crash)
    #     times_since_last = trial_data['seconds'] - trial_data['seconds'].shift(1, fill_value=0)
    #
    #     # Keep only crash events longer than given time gap away since last
    #     valid_trial_data = trial_data[times_since_last > time_gap]
    #     crash_events_valid = pd.concat([crash_events_valid, valid_trial_data])
    #
    #     # For validation, include excluded crash events
    #     invalid_trial_data = trial_data[times_since_last <= time_gap]
    #     times_since_last.name = "seconds_since_last_crash"
    #     invalid_trial_data.join(times_since_last)
    #     excluded_crashes = pd.concat([excluded_crashes, invalid_trial_data])

    # debug only
    # valid_crashes.to_csv("valids_new.csv")

    # save crash features as pickle for training
    # crash_feature_label_df.to_pickle(crash_pickle_path)
    #
    # ####### noncrash event data info  ######
    #
    # peopleTrialHasCrash_ex = all_valid_crashes.peopleTrialKey.unique()
    #
    # noncrash_feature_label_df = pd.DataFrame()
    #
    # print("extracting")
    # for num in range(len(peopleTrialHasCrash_ex)):
    #     curr_crash_key = peopleTrialHasCrash_ex[num]
    #     # print(num)
    #     # again, find valid intervals between crashes
    #     trial_data = all_valid_crashes[all_valid_crashes.peopleTrialKey == curr_crash_key]
    #     trial_data['seconds_shift'] = trial_data['seconds'].shift(1)
    #     trial_data.fillna(0, inplace=True)
    #     trial_data['time_gap'] = trial_data['seconds'] - trial_data['seconds_shift']
    #
    #     # all trial data at current key
    #     df_trial = raw_data[(raw_data['peopleTrialKey'] == curr_crash_key)]
    #
    #     for crash_time in (all_valid_crashes.loc[
    #         (all_valid_crashes['peopleTrialKey'] == curr_crash_key), 'seconds']):
    #         pass
    #         # left bound: last crash time of current valid crash
    #         # left = trial_data.seconds_shift[trial_data.seconds == crash_time].iloc[0]
    #         # # right bound: crash interval ahead of current crash
    #         # right = crash_time - time_scale_train - time_ahead
    #         #
    #         # # crucially not include left boundary (last crash entry)
    #         # temp_serie = df_trial.loc[(df_trial.seconds > left) & (df_trial.seconds <= right), 'seconds']
    #         #
    #         # # run sliding window on noncrash event
    #         # list_all = sliding_window(temp_serie, time_scale_train, time_step)
    #         #
    #         # # only record list with more than 2 data points
    #         # if len(list_all) >= 2:
    #         #     for l_num in range(len(list_all)):
    #         #         # print("l_num")
    #         #         # print(l_num)
    #         #         # print("list_all[l_num]")
    #         #         # print(list_all[l_num])
    #         #         entries_for_train = df_trial[(df_trial.seconds >= list_all[l_num].iloc[0]) \
    #         #                            & (df_trial.seconds <= list_all[l_num].iloc[-1])]
    #         #
    #         #         ##### resample & interpolate
    #         #         entries_for_train = entries_for_train[
    #         #             ['seconds', 'currentVelRoll', 'currentPosRoll', 'calculated_vel', 'joystickX', 'peopleTrialKey']]
    #         #         x = entries_for_train.seconds
    #         #         y_calculated_vel = entries_for_train.calculated_vel
    #         #         y_org_vel = entries_for_train.currentVelRoll
    #         #         y_currentPosRoll = entries_for_train.currentPosRoll
    #         #         y_joystickX = entries_for_train.joystickX
    #         #
    #         #         new_x = np.linspace(x.min(), x.max(), sampling_rate)
    #         #         new_y_calculated_vel = sp.interpolate.interp1d(x, y_calculated_vel, kind='linear')(new_x)
    #         #         new_y_original_vel = sp.interpolate.interp1d(x, y_org_vel, kind='linear')(new_x)
    #         #         new_y_currentPosRoll = sp.interpolate.interp1d(x, y_currentPosRoll, kind='linear')(new_x)
    #         #         new_y_joystickX = sp.interpolate.interp1d(x, y_joystickX, kind='linear')(new_x)
    #         #
    #         #         arr11 = np.dstack([new_y_calculated_vel, new_y_currentPosRoll, new_y_joystickX]).reshape(sampling_rate,
    #         #                                                                                                  3)
    #         #         arr22 = np.dstack([new_y_original_vel, new_y_currentPosRoll, new_y_joystickX]).reshape(sampling_rate, 3)
    #         #         arr33 = 0
    #         #         arr44 = entries_for_train['peopleTrialKey'].iloc[0]
    #         #         arr55 = entries_for_train['seconds'].iloc[0]
    #         #         arr66 = entries_for_train['seconds'].iloc[-1]
    #         #
    #         #         noncrash_feature_label_df = pd.concat(
    #         #             [noncrash_feature_label_df, pd.DataFrame([[arr11, arr22, arr33, arr44, arr55, arr66]],
    #         #                                                      columns=["features_cal_vel", "features_org_vel", 'label',
    #         #                                                               'peopleTrialKey', 'start_seconds',
    #         #                                                               'end_seconds'])])
    #
    #             ## save it as pickle for training
    # # noncrash_feature_label_df.to_pickle(noncrash_pickle_path)

    # TODO save columns instead

    print("Feature generation done!")


if __name__ == "__main__":
    # noinspection PyTypeChecker
    argparser = argparse.ArgumentParser(prog="Data Pre-processing Argparser",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument(
        '--scale', type=float, default=1.0, help='time scale of training data, in seconds')
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
    if args.rolling < MIN_STEP:
        raise argparse.ArgumentTypeError("Rolling step must be smaller than {}".format(MIN_STEP))

    # Ensure gap large enough to accommodate data extraction.
    # Time gap is used to exclude those consecutive crash events happened within less than this length.
    # When two crash events happened too closely, we could not generate enough data to for interpolation.
    # In principle: time gap >= non crashing time scale + crushing time scale + time ahead of predicted event
    if args.gap < (2 * args.scale + args.ahead):
        args.gap = (2 * args.scale + args.ahead)

    # generate feature files
    preprocess_data(window_size=args.scale,
                    time_ahead=args.ahead,
                    sampling_rate=args.rate,
                    time_gap=args.gap,
                    time_step=args.rolling)
