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

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import argparse
import random

# control for randomness in case of any
RANDOM_SEED = 2020
np.random.seed(seed=RANDOM_SEED)
random.seed(RANDOM_SEED)

# argparser value checker
MIN_STEP = 0.04

# paths for saving output
OUT_DIR_FORMAT = "data/data_{}ms/"
CRASH_FILE_FORMAT = "crash_feature_label_{}ahead_{}scale_test"
NONCRASH_FILE_FORMAT = "noncrash_feature_label_{}ahead_{}scale_test"

# columns we will use for interpolation
COLS_TO_INTERPOLATE = ['currentVelRoll', 'currentPosRoll', 'calculated_vel', 'joystickX']


####### define my own sliding window function
def sliding_window(interval_series:pd.Series, time_scale:float, rolling_step:float, avoid_duplicate=False) -> list:
    """
    Identify extractable time interval windows in original data.
    Each window starts at time point given in interval_series.
    :authors: Jie Tang, Yonglin Wang
    :param interval_series: series of time points between 2 crash events
    :param time_scale: time scale length of data used for prediction in seconds, i.e. size of window
    :param rolling_step: length of rolling step in seconds
    :param avoid_duplicate: whether to check for duplicate windows before appending (time consuming)
    :return: list of series of windows, each containing seconds timestamp from original data
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

        # check for duplicate window to strictly avoid duplicate (time consuming)
        if avoid_duplicate:
            if len(window_series) > 1:
                # only keep unique windows with more than 1 data points
                if not valid_windows:
                    valid_windows.append(window_series)
                elif not any([window_series.equals(existing_window) for existing_window in valid_windows]):
                    valid_windows.append(window_series)

        # increment boundary
        left_bound += rolling_step
        right_bound += rolling_step

    # # iterate over each point interval
    # for i in range(len(interval_series)):
    #     # define time boundary for window
    #     left_bound = interval_series.iloc[i] + i * rolling_step
    #     right_bound = left_bound + time_scale
    #     window_series = interval_series[interval_series.between(left_bound, right_bound)]
    #     ## if the series length is less than 2, we need to pass it
    #     if len(window_series) < 2:
    #         continue
    #     ## need to determine the boundary that last time series and left boundary is enough to do sliding ( greater than buck_size)
    #     ## or we need to break the loop, since the data points are not enough to do next sliding.
    #     if interval_series.iloc[-1] - left_bound < time_scale:
    #         break
    #     valid_windows.append(window_series)

    return valid_windows


def interpolate_entries(entries,
                        sampling_rate=50,
                        cols_to_interpolate=None,
                        x_col="seconds"):
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

    # record interpolation result for each column
    output = {}

    # interpolate each column
    for col_name in cols_to_interpolate:
        output["new_{}".format(col_name)] = sp.interpolate.interp1d(
            x, entries[col_name], kind='linear')(x_sample)

    return output


def generate_features(time_scale_train,
                      time_ahead,
                      sampling_rate,
                      time_gap,
                      time_step):
    '''
    author: Jie Tang
    inputs:
        time_scale_train: time length of data used for training, in seconds
        time_ahead: time in advance to predict, in seconds
        sampling_rate: sampling rate in each time_scale_train
        time_gap: minimal length of time allowed between two crash events for sliding windows extraction

        time_step: the step we will move when we do sliding window
    '''
    # initial settings (to be moved)
    np.set_printoptions(suppress=True)

    # convert time unit and get output paths
    ahead_ms_str = str(int(time_ahead * 1000))
    scale_ms_str = str(int(time_scale_train * 1000))

    crash_pickle_path = os.path.join(OUT_DIR_FORMAT.format(scale_ms_str), CRASH_FILE_FORMAT.format(ahead_ms_str, scale_ms_str))
    noncrash_pickle_path = os.path.join(OUT_DIR_FORMAT.format(scale_ms_str), NONCRASH_FILE_FORMAT.format(ahead_ms_str, scale_ms_str))

    # ensure output folder exists
    if not os.path.exists(OUT_DIR_FORMAT.format(scale_ms_str)):
        os.makedirs(OUT_DIR_FORMAT.format(scale_ms_str))

    # getting data
    # data_all = pd.read_csv('data/data_all.csv')

    ##### extract all needed columns ######
    cols_used = ['seconds', 'trialPhase', 'currentPosRoll', 'currentVelRoll', 'calculated_vel',
                  'joystickX', 'peopleName', 'trialName', 'peopleTrialKey', 'datetimeNew']

    raw_data = pd.read_csv('data/data_all.csv', usecols=cols_used)
    raw_data['datetimeNew'] = pd.to_datetime(raw_data['datetimeNew'])
    # filter out non-human controls in data
    raw_data = raw_data[raw_data.trialPhase != 1]
    # raw_data.set_index('datetimeNew', inplace=True)

    # find all crash controls by trial phase
    crash_data = raw_data[raw_data.trialPhase == 4]

    # get unique peopleTrialKeys that have crashes
    crash_keys_all = set(crash_data.peopleTrialKey.unique())

    ####### give a threshold of a time interval between two consecutive crashes within one trial #######

    # extract crash events that are not too short from the previous crash
    valid_crashes = pd.DataFrame()
    # TODO record # only?
    excluded_crashes = pd.DataFrame()

    # keep track of valid crash events for validation
    num_valid_crash = 0

    # generate feature and save
    crash_feature_label_df = pd.DataFrame()


    # extract crash events features from each trial
    for crash_key, trial_raw_data in raw_data.groupby("peopleTrialKey"):
        # only process keys that has crashes
        if crash_key in crash_keys_all:
            # find all crash data points in this trial
            trial_crash_data = trial_raw_data[trial_raw_data.trialPhase == 4]

            # Calculate each crash event's elapsed time since last crash (defined as difference since 0 for first crash)
            trial_crash_data["preceding_crash_seconds"] = trial_crash_data.seconds.shift(1, fill_value=0)
            times_since_last = trial_crash_data.seconds - trial_crash_data.preceding_crash_seconds

            # Keep only crash events longer than given time gap away since last
            valid_trial_data = trial_crash_data[times_since_last > time_gap]
            # record crash events for later use
            valid_crashes = pd.concat([valid_crashes, valid_trial_data])
            num_valid_crash += len(valid_trial_data)

            # For validation, include excluded crash events
            invalid_trial_data = trial_crash_data[times_since_last <= time_gap]
            times_since_last.name = "seconds_since_last_crash"
            excluded_crashes = pd.concat([excluded_crashes, invalid_trial_data.join(times_since_last)])

            # iterate through each valid crash to create data entry
            for crash_time in valid_trial_data.seconds:

                # (1/2) Extract Crash Events in each group
                # find corresponding data points between time scale start and crash event to generate training data
                entries_for_train = trial_raw_data[trial_raw_data.seconds.between(
                    crash_time - time_scale_train - time_ahead, crash_time - time_ahead)]

                # keep only useful data TODO Don't need these...?
                # entries_for_train = entries_for_train[
                #     ['seconds', 'currentVelRoll', 'currentPosRoll', 'calculated_vel', 'joystickX', 'peopleTrialKey']]


                # only process entries with more than one data points
                if len(entries_for_train) > 1:
                    # resample & interpolate
                    int_results = interpolate_entries(entries_for_train)

                    new_y_calculated_vel = int_results["new_calculated_vel"]
                    new_y_original_vel = int_results["new_currentVelRoll"]
                    new_y_currentPosRoll = int_results["new_currentPosRoll"]
                    new_y_joystickX = int_results["new_joystickX"]

                    arr1 = np.dstack([new_y_calculated_vel, new_y_currentPosRoll, new_y_joystickX]).reshape(sampling_rate,
                                                                                                            3)
                    arr2 = np.dstack([new_y_original_vel, new_y_currentPosRoll, new_y_joystickX]).reshape(sampling_rate, 3)
                    arr3 = 1
                    arr4 = entries_for_train['peopleTrialKey'].iloc[0]
                    arr5 = entries_for_train['seconds'].iloc[0]
                    arr6 = entries_for_train['seconds'].iloc[-1]



                    crash_feature_label_df = pd.concat(
                        [crash_feature_label_df, pd.DataFrame([[arr1, arr2, arr3, arr4, arr5, arr6]],
                                                              columns=["features_cal_vel", "features_org_vel", 'label',
                                                                       'peopleTrialKey', 'start_seconds', 'end_seconds'])])
                else:
                    pass #TODO record the excluded criteria
                # (2/2) Extract Noncrash Events in each group
            # crash_events_valid = pd.concat([crash_events_valid, valid_trial_data])

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

    print("{} out of total {} examined crash events excluded due to following last crash in less than {}s.".format(
        len(excluded_crashes), num_valid_crash, time_gap))

    # output excluded and included crashes
    excluded_crashes["included"] = "yes"

    # debug only
    # valid_crashes.to_csv("valids_new.csv")

            ## save it as pickle for training
    crash_feature_label_df.to_pickle(crash_pickle_path)

    ####### noncrash event data info  ######

    peopleTrialHasCrash_ex = valid_crashes.peopleTrialKey.unique()

    noncrash_feature_label_df = pd.DataFrame()
    for num in range(len(peopleTrialHasCrash_ex)):
        curr_crash_key = peopleTrialHasCrash_ex[num]
        # print(num)
        trial_data = valid_crashes[valid_crashes.peopleTrialKey == curr_crash_key]
        trial_data['seconds_shift'] = trial_data['seconds'].shift(1)
        trial_data.fillna(0, inplace=True)
        trial_data['time_gap'] = trial_data['seconds'] - trial_data['seconds_shift']

        df_trial = raw_data[(raw_data['peopleTrialKey'] == curr_crash_key)]

        for crash_time in (valid_crashes.loc[
            (valid_crashes['peopleTrialKey'] == curr_crash_key), 'seconds']):
            # left bound: last crash time of current valid crash, right bound: crash interval ahead of current crash
            left = trial_data.seconds_shift[trial_data.seconds == crash_time].iloc[0]
            right = crash_time - time_scale_train - time_ahead
            # noncrash_time_range = [left, right]

            # crucially not include left boundary (last crash entry)
            temp_serie = df_trial.loc[(df_trial.seconds > left) & (df_trial.seconds <= right), 'seconds']

            ## run sliding window on noncrash event
            list_all = sliding_window(temp_serie, time_scale_train, time_step)

            if len(list_all) < 1:
                break
            for x in range(len(list_all)):
                entries_for_train = df_trial[(df_trial.seconds >= list_all[x].iloc[0]) \
                                   & (df_trial.seconds <= list_all[x].iloc[-1])]

                ##### resample & interpolate
                entries_for_train = entries_for_train[
                    ['seconds', 'currentVelRoll', 'currentPosRoll', 'calculated_vel', 'joystickX', 'peopleTrialKey']]
                x = entries_for_train.seconds
                y_calculated_vel = entries_for_train.calculated_vel
                y_org_vel = entries_for_train.currentVelRoll
                y_currentPosRoll = entries_for_train.currentPosRoll
                y_joystickX = entries_for_train.joystickX

                new_x = np.linspace(x.min(), x.max(), sampling_rate)
                new_y_calculated_vel = sp.interpolate.interp1d(x, y_calculated_vel, kind='linear')(new_x)
                new_y_original_vel = sp.interpolate.interp1d(x, y_org_vel, kind='linear')(new_x)
                new_y_currentPosRoll = sp.interpolate.interp1d(x, y_currentPosRoll, kind='linear')(new_x)
                new_y_joystickX = sp.interpolate.interp1d(x, y_joystickX, kind='linear')(new_x)

                arr11 = np.dstack([new_y_calculated_vel, new_y_currentPosRoll, new_y_joystickX]).reshape(sampling_rate,
                                                                                                         3)
                arr22 = np.dstack([new_y_original_vel, new_y_currentPosRoll, new_y_joystickX]).reshape(sampling_rate, 3)
                arr33 = 0
                arr44 = entries_for_train['peopleTrialKey'].iloc[0]
                arr55 = entries_for_train['seconds'].iloc[0]
                arr66 = entries_for_train['seconds'].iloc[-1]

                noncrash_feature_label_df = pd.concat(
                    [noncrash_feature_label_df, pd.DataFrame([[arr11, arr22, arr33, arr44, arr55, arr66]],
                                                             columns=["features_cal_vel", "features_org_vel", 'label',
                                                                      'peopleTrialKey', 'start_seconds',
                                                                      'end_seconds'])])

                ## save it as pickle for training
    noncrash_feature_label_df.to_pickle(noncrash_pickle_path)


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
    generate_features(time_scale_train=args.scale,
                      time_ahead=args.ahead,
                      sampling_rate=args.rate,
                      time_gap=args.gap,
                      time_step=args.rolling)
