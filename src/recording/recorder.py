#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yonglin Wang
# Date: 2021/1/31 9:47 PM
"""
Lightweight class to record training and dataset file info for later retrieval.
Experiment ID is generated from the exp_ID_config.csv file; 1 if file not exist.
Also includes data FalsePositiveCategorizer class for categorizing false negatives and false positives
"""

import os
from datetime import date
import pickle
from typing import Union, Dict, Tuple, List

import pandas as pd
from tensorflow.keras import Sequential
import numpy as np

from processing.marsdataloader import MARSDataLoader, generate_all_feat_df
from processing.extract_features import extract_destabilize
import consts as C


class Recorder():
    def __init__(self,
                 loader: MARSDataLoader,
                 train_args: dict,
                 seq_y: bool,
                 verbose=True):

        self.loader = loader
        self.verbose = verbose
        self.train_args = train_args
        self.configID = self.train_args["configID"]
        self.exp_date = date.today().strftime("%B %d, %Y")
        self.using_seq_label = seq_y
        self.dataset_name = C.DATA_SUBDIR

        # get unique experiment ID for current project folder
        self.exp_ID = int(_find_next_exp_ID())

        # unique experiment folder path
        # i.e. fill in exp{}_{}win_{}ahead_conf{}_{}
        self.exp_dir = C.EXP_FORMAT.format(self.exp_ID,
                                           self.train_args["window"],
                                           self.train_args["ahead"],
                                           self.train_args["configID"],
                                           self.train_args["model"])

        # get prediction path
        self.pred_path = C.PRED_PATH.format(self.exp_ID,
                                            self.train_args["window"],
                                            self.train_args["ahead"],
                                            self.train_args["configID"],
                                            self.train_args["model"])


        self.model_path = os.path.join(self.exp_dir, C.MODEL_PATH)  # path to model
        self.recorder_path = os.path.join(self.exp_dir, C.REC_BASENAME)
        self.norm_stats_path = os.path.join(self.exp_dir, C.NORM_STATS_PATH)

        # to be recorded on record_experiment
        self.history: dict = {}  # hisotry dict from keras history object, if any passed
        self.time_taken: str = ""     # string of time taken in this experiment
        self.average_epochs: float = 0
        self.std_epochs: float = 0
        self.best_split: int = -1     # index of the best performing split, 0-based

        if self.verbose:
            print("Now recording experiment #{}".format(self.exp_ID))

    def record_experiment(self,
                          test_results: dict,
                          time_taken: str,
                          epoch_list: list,
                          best_split: int,
                          model: Sequential = None,
                          norm_stats: dict = None,
                          train_history: list = None,
                          save_model: bool = False):
        """record experiment configuration and statistics"""
        # link references
        if train_history:
            self.history = train_history
        self.average_epochs = float(np.mean(epoch_list))
        self.std_epochs = float(np.std(epoch_list))
        self.best_split = best_split
        self.time_taken = time_taken

        # create new path in results and experiment folders
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)

        if model is not None and save_model:
            self.__save_model(model)

        if norm_stats is not None:
            self.__save_norm_stats(norm_stats)

        # append test set metrics to results/exp_results_all.csv
        self.__save_results(test_results)

        # once all of the above done, append experiment info to results/exp_ID_config.csv
        self.__save_exp_config()

        # pickle this recorder to its path
        pickle.dump(self, open(self.recorder_path, "wb"))

        if self.verbose:
            print("Experiment {} recorded successfully!".format(self.exp_ID))

    def save_predictions(self,
                         test_inds: Union[list, np.ndarray],
                         y_pred: Union[list, np.ndarray],
                         true_preds_path: str="",
                         false_preds_path: str="",
                         custom_ahead: float=None,
                         save_lookahead_windows=False) -> None:
        """save prediction for specified rows; separate files will be generated if no sequence label used and true and
        false pred paths are given."""
        # generate test DataFrame
        test_df = generate_all_feat_df(self.loader, self.configID, inds=test_inds)

        # append predictions
        if y_pred.ndim <=2:
            test_df[C.PRED_COL] = y_pred
        else:
            # squeeze sequence labels to (num_samples, sampling_rate)
            y_pred = y_pred.squeeze(-1)
            # convert to list of arrays for DataFrame to correctly append new column
            test_df[C.PRED_COL] = [y_pred[i, :] for i in range(y_pred.shape[0])]

        # reorder so that false predictions come up first and label true and false predictions
        if self.using_seq_label:
            # compare seq predictions by row
            test_df["pred_seq_is_correct"] = test_df.apply(lambda row: np.array_equal(row.seq_label, row[C.PRED_COL]), axis=1)
            test_df.sort_values("pred_seq_is_correct", inplace=True)
        else:
            # show false negatives first
            test_df.sort_values(["label", C.PRED_COL], ascending=[False, True], inplace=True)
            # pop seq_label column since not needed
            test_df.drop(["seq_label"], axis=1, inplace=True)

        # save correct and incorrect predictions separately if both paths are given; otherwise, save in one file
        if true_preds_path and false_preds_path and not self.using_seq_label:
            pred_label_is_correct = test_df.apply(lambda row: np.array_equal(row.label, row[C.PRED_COL]), axis=1)

            # categorize false negatives for non-sequential labels
            if not self.using_seq_label:
                print("now processing destab joystick in lookahead windows...")
                test_df = append_lookahead_stats(test_df, self, custom_ahead, save_lookahead_windows=save_lookahead_windows)

            grouped = test_df.groupby(pred_label_is_correct)

            # find respective rows and save separately
            true_df = grouped.get_group(True)
            false_df = grouped.get_group(False)

            true_df.to_csv(true_preds_path, index=False)
            print(f"saved {len(true_df)} true/correct predictions to {true_preds_path}")
            false_df.to_csv(false_preds_path, index=False)
            print(f"saved {len(false_df)} true/correct predictions to {false_preds_path}")
            print(f"accuracy (for debugging): {len(true_df)/(len(true_df) + len(false_df))}")
        else:
            test_df.to_csv(self.pred_path, index=False)

        if self.verbose:
            print("Model test set input and prediction saved successfully!")

    def list_training_columns(self) -> list:
        return C.CONFIG_SPECS[self.configID][C.COLS_USED]

    def __save_model(self, model) -> None:
        """helper to save models"""
        assert type(model) == Sequential, "Only Keras Sequential models are supported! " \
                                          "Consider adding new code and updating model saving methods."
        # append number to avoid collision, if needed
        collision_n = 0
        if os.path.exists(self.model_path):
            while os.path.exists(self.model_path + "_" + str(collision_n)):
                collision_n += 1
            self.model_path = self.model_path + "_" + str(collision_n)
        if collision_n:
            print("Model path has been revised to {} to avoid collision. \n"
                  "In principal, this shouldn't happen since model path has unique experiment ID.".format(
                self.model_path))
        model.save(self.model_path)

    def __save_norm_stats(self, norm_stats: dict):
        """helper to save normalization stats"""
        pickle.dump(norm_stats, open(self.norm_stats_path, "wb"))

    def __save_results(self, cv_results: Dict[str, list]) -> None:
        """calculate and append CV test results to results/exp_results_all.csv"""
        # compute mean and std of CV results
        calculated_results = {}
        for metric_name in cv_results:
            calculated_results[metric_name + C.MEAN_SUFFIX] = np.nanmean(cv_results[metric_name])
            calculated_results[metric_name + C.STD_SUFFIX] = np.nanstd(cv_results[metric_name])

        # add ID to current results
        calculated_results[C.EXP_COL_CONV[C.EXP_ID_COL]] = self.exp_ID

        # retrieve previous results
        try:
            results_df = pd.read_csv(C.ALL_RES_CSV_PATH)
        except IOError:
            results_df = pd.read_csv(C.TEMPLATE_ALL_RES)

        # save current results
        results_df = results_df.append(calculated_results, ignore_index=True)
        results_df.to_csv(C.ALL_RES_CSV_PATH, index=False)

    def __save_exp_config(self) -> None:
        """save current configuration to exp_ID_config.csv for easy retrieval"""
        # load configuration file
        if os.path.exists(C.EXP_ID_LOG):
            config_df = pd.read_csv(C.EXP_ID_LOG, dtype={C.EXP_ID_COL: int})
        else:
            config_df = pd.read_csv(C.TEMPLATE_ID_LOG, dtype={C.EXP_ID_COL: int})

        config_df = config_df.append(self.__compile_exp_dict(), ignore_index=True)
        config_df.to_csv(C.EXP_ID_LOG, index=False)

    def __compile_exp_dict(self) -> dict:
        """compile experiment configuration dictionary"""
        # put together attributes for extraction
        all_atts = {**vars(self), **vars(self.loader), **self.train_args}

        # keep only savable atts--filter out lists, dicts, etc.
        savable_atts = _filter_values(all_atts)

        # convert the convertable columns, if possible, for output
        output = {}
        for (column, value) in savable_atts.items():
            if column in C.EXP_COL_CONV:
                output[C.EXP_COL_CONV[column]] = value
            else:
                output[column] = value

        # Lastly, add info not included in class fields.
        # text description of dataset configuration (e.g. basic triple)
        output[C.CONFIG_DESC_COL_NAME] = C.CONFIG_SPECS[self.configID][C.CONFIG_OVERVIEW]

        return output


def _find_next_exp_ID() -> int:
    """helper to find the next unique exp ID in given exp dir, fast operation to avoid collision"""
    # find ID based on ID record file
    try:
        with open(C.EXP_ID_RECORD, "r") as id_file:
            next_id = int(id_file.read())
    except IOError:
        next_id = 1

    # save ID to record
    with open(C.EXP_ID_RECORD, 'w') as count_file:
        count_file.write(str(next_id + 1))

    return next_id


def _filter_values(vars_dict: dict)->dict:
    """helper function to filter out dictionary entries whose values are not str, num or bool; called before converting args to column names"""
    output = {key: value for key, value in vars_dict.items() if type(value) in C.ACCEPTABLE_TYPES}
    # ad-hoc popping duplicate keys
    output.pop("seq_label")     # same as using_seq_label in Recorder
    # ad-hoc for adding layer sizes
    if vars_dict["model"] in {C.CNN, C.MLP}:
        output["layer_sizes"] = vars_dict["layer_sizes"]
    else:
        output["layer_sizes"] = "NA"

    # ad-hoc change filter_number to NA for non-CNN models
    if vars_dict["model"] != C.CNN:
        output["filter_number"] = "NA"

    return output


class TestSetProcessor:
    def __init__(self,
                 recorder: Recorder,
                 current_ahead: float
                 ):
        if not os.path.exists(C.RAW_DATA_PATH):
            raise FileNotFoundError("Raw data file cannot be found at {}".format(C.RAW_DATA_PATH))
        # extract all needed columns
        self.raw_data = pd.read_csv(C.RAW_DATA_PATH, usecols=C.ESSENTIAL_RAW_COLS)
        # filter out non-human controls in data for faster processing
        self.raw_data = self.raw_data[self.raw_data.trialPhase != 1]
        # group by trials for easy locating
        self.grouped = self.raw_data.groupby('peopleTrialKey')
        # get data from recorder
        self.window_size = recorder.loader.window
        self.lookahead = current_ahead
        self.velocity_col = "calculated_vel" if "velocity_cal" in recorder.list_training_columns() else "currentVelRoll"

    def generate_categories(self, data_df: pd.DataFrame) -> Tuple[List[float], List[float], List[int], List[int], List[float], List[float]]:
        """append a new column containing entry stats"""
        # apply categorization function to each data point to assign error type.
        # ico = including carryover destabilizing joystick from input window (ie seen by machine); eco = exclude such
        lookahead_avg_destab_mag_ico, lookahead_avg_destab_mag_eco = [], []
        lookahead_total_destab_steps_ico, lookahead_total_destab_steps_eco = [], []
        lookahead_destab_sustained_ico, lookahead_destab_sustained_eco = [], []
        for _, row in data_df.iterrows():
            avg_destab_mag_ico, avg_destab_mag_eco, \
            total_destab_steps_ico, total_destab_steps_eco, \
            destab_sustained_ico, destab_sustained_eco = self._extract_lookahead_stats(float(row.end_seconds),
                                                                                 self.grouped.get_group(row.trial_key))
            lookahead_avg_destab_mag_ico.append(avg_destab_mag_ico)
            lookahead_avg_destab_mag_eco.append(avg_destab_mag_eco)
            lookahead_total_destab_steps_ico.append(total_destab_steps_ico)
            lookahead_total_destab_steps_eco.append(total_destab_steps_eco)
            lookahead_destab_sustained_ico.append(destab_sustained_ico)
            lookahead_destab_sustained_eco.append(destab_sustained_eco)
        return lookahead_avg_destab_mag_ico, lookahead_avg_destab_mag_eco, \
               lookahead_total_destab_steps_ico, lookahead_total_destab_steps_eco, \
                lookahead_destab_sustained_ico, lookahead_destab_sustained_eco

    def save_lookahead_windows(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """save lookahead windows of each entry in given DataFrame, effective if specified in predict.py args"""
        # output: [trial key, window_end], vel, pos, joystick,
        # locate lookahead sequences, note that first entry is last time step in input window
        lookahead_df_dict = {key:[] for key in [# "trial_key", "window_end",
                                                "lookahead_vel", "lookahead_pos", "lookahead_joy", "lookahead_times"]}
        for _, row in data_df.iterrows():
            end_sec = float(row.end_seconds)
            trial_entries = self.grouped.get_group(row.trial_key)
            lookahead_readings = trial_entries[
                trial_entries.seconds.between(end_sec, end_sec + self.lookahead, inclusive="neither")]
            # record data into df
            # lookahead_df_dict["trial_key"].append(row.trial_key)
            # lookahead_df_dict["window_end"].append(end_sec)
            lookahead_df_dict["lookahead_vel"].append(lookahead_readings[self.velocity_col].to_numpy())
            lookahead_df_dict["lookahead_pos"].append(lookahead_readings['currentPosRoll'].to_numpy())
            lookahead_df_dict["lookahead_joy"].append(lookahead_readings['joystickX'].to_numpy())
            lookahead_df_dict["lookahead_times"].append(lookahead_readings.seconds.to_numpy())

        return data_df.assign(**lookahead_df_dict)


    def _extract_lookahead_stats(self, end_sec: float,
                                 trial_entries: pd.DataFrame) -> Tuple[float, float, int, int, float, float]:
        """for a single entry, return its avg destabilizing joystick magnitude, w/ or w/o carryover destabilizing
        joystick, ie destab carried over from input window (i.e. "seen by machine", such as ...111 -> 1100);
        if no such destab, return NaN. If no lookahead window (i.e. for end-of-trial neg samples), return NaN """

        # locate lookahead sequences, note that first entry is last time step in input window
        lookahead_readings = trial_entries[trial_entries.seconds.between(end_sec, end_sec + self.lookahead, inclusive="left")]
        # see if deflection occurs
        base_triples = lookahead_readings[[self.velocity_col, 'currentPosRoll', 'joystickX']].to_numpy()
        # get an array of whether each reading in lookahead is destabilizing, (sampling_rate,)
        has_deflections = extract_destabilize(base_triples, single_entry=True)
        last_in_window_is_destab = has_deflections[0]
        destab_ico = has_deflections[1:]
        joystick_ico = lookahead_readings.joystickX.to_numpy()[1:]
        assert destab_ico.shape == joystick_ico.shape, f"shape diff: {destab_ico.shape} vs {joystick_ico.shape}"

        # time points for calculating length sustained
        timepoints_ico = lookahead_readings.seconds.to_numpy()[1:]

        # split lookahead destab into bool subarray chunks: 1110011000 -> 111 00 11 000
        lookahead_destab_cutpoints = np.where(np.diff(destab_ico))[0] + 1
        destab_chunks_ico = np.split(destab_ico, lookahead_destab_cutpoints)      # destab chunks incl potential carryover destab
        joystick_chunks_ico = np.split(joystick_ico, lookahead_destab_cutpoints)  # corresponding joystick chunks incl potential carryover destab
        timepoints_chunks_ico = np.split(timepoints_ico, lookahead_destab_cutpoints)

        # take out carryover if any
        destab_chunks_eco = destab_chunks_ico.copy()    # destab chunks excl potential carryover destab
        joystick_chunks_eco = joystick_chunks_ico.copy()
        timepoints_chunks_eco = timepoints_chunks_ico.copy()

        # note: end-of-trial neg samples do not have lookahead window
        if destab_chunks_ico[0].shape[0] != 0 and destab_chunks_ico[0][0] == True:
            if last_in_window_is_destab:
                # if the first chunk is a carry over destab, pop it
                destab_chunks_eco.pop(0)
                joystick_chunks_eco.pop(0)
                timepoints_chunks_eco.pop(0)

        # piece the chunks back into one vector
        assert len(destab_chunks_eco) == len(joystick_chunks_eco)
        if not destab_chunks_eco:
            # if chunk lists become empty after popping, assign empty arrays
            destab_eco, joystick_eco = np.empty(0), np.empty(0)
        else:
            destab_eco, joystick_eco = np.hstack(destab_chunks_eco), np.hstack(joystick_chunks_eco)
        lookahead_has_destab_ico = np.any(destab_ico)
        lookahead_has_destab_eco = np.any(destab_eco)   # returns False if empty

        # Average Absolute magnitude of destabilizing joystick deflections: (dot product)/(# of destab, ie sum)
        # avg is NaN iff lookahead has no such destab segment
        avg_destab_magnitude_ico = destab_ico.dot(np.abs(joystick_ico)) / np.sum(destab_ico) if lookahead_has_destab_ico else np.nan
        avg_destab_magnitude_eco = destab_eco.dot(np.abs(joystick_eco)) / np.sum(destab_eco) if lookahead_has_destab_eco else np.nan

        # add up time diffs
        destab_sustained_ico, destab_sustained_eco = 0, 0
        for destab_chunk, time_chunk in zip(destab_chunks_ico, timepoints_chunks_ico):
            if destab_chunk.size > 0 and destab_chunk[0] == True:
                destab_sustained_ico += time_chunk[-1] - time_chunk[0]
        for destab_chunk, time_chunk in zip(destab_chunks_eco, timepoints_chunks_eco):
            if destab_chunk.size > 0 and destab_chunk[0] == True:
                destab_sustained_eco += time_chunk[-1] - time_chunk[0]

        return avg_destab_magnitude_ico, avg_destab_magnitude_eco, \
               int(np.sum(destab_ico)), int(np.sum(destab_eco)), \
               destab_sustained_ico, destab_sustained_eco


def append_lookahead_stats(dataset_df: pd.DataFrame,
                           recorder: Recorder,
                           current_ahead: float=None,
                           save_lookahead_windows=False) -> pd.DataFrame:
    """append prediction categories to given test set DataFrame"""
    if not current_ahead:
        current_ahead = recorder.loader.ahead

    prc = TestSetProcessor(recorder, current_ahead)
    lookahead_avg_destab_mag_ico,  lookahead_avg_destab_mag_eco, \
    lookahead_total_destab_steps_ico, lookahead_total_destab_steps_eco,\
        lookahead_destab_sustained_ico, lookahead_destab_sustained_eco = prc.generate_categories(dataset_df)

    new_appended_df = dataset_df.assign(lookahead_avg_destab_mag_ico=lookahead_avg_destab_mag_ico,
                                        lookahead_avg_destab_mag_eco=lookahead_avg_destab_mag_eco,
                                        lookahead_total_destab_steps_ico=lookahead_total_destab_steps_ico,
                                        lookahead_total_destab_steps_eco=lookahead_total_destab_steps_eco,
                                        lookahead_destab_sustained_ico=lookahead_destab_sustained_ico,
                                        lookahead_destab_sustained_eco=lookahead_destab_sustained_eco)

    if save_lookahead_windows:
        new_appended_df = prc.save_lookahead_windows(new_appended_df)
    return new_appended_df


if __name__ == "__main__":
    # debugging categorization function
    test_curr_ahead = 1.0
    test_false_df = pd.read_csv("local/test_false_df.csv")
    test_recorder = pickle.load(open("local/test_recorder.pkl", "rb"))
    new_df = append_lookahead_stats(test_false_df, test_recorder, test_curr_ahead)
    print("Done!")
