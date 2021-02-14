#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yonglin Wang
# Date: 2021/1/31 9:47 PM
"""
Lightweight class to record training and dataset file info for later retrieval
Experiment ID is generated from the exp_ID_config.csv file; 1 if file not exist
"""

import math
import os
import pickle
from datetime import date
from typing import Union, Dict

import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import History
import numpy as np

from processing.marsdataloader import MARSDataLoader, generate_all_feat_df
import consts as C


class Recorder():
    def __init__(self,
                 loader: MARSDataLoader,
                 train_args: dict,
                 verbose=True):

        self.loader = loader
        self.verbose = verbose
        self.train_args = train_args
        self.configID = self.train_args["configID"]
        self.exp_date = date.today().strftime("%B %d, %Y")

        # get unique experiment ID
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
        self.recorder_path = os.path.join(self.exp_dir, C.REC_PATH)

        # to be recorded on record_experiment
        self.history = None  # hisotry dict from keras history object, if any passed
        self.time_taken = None
        self.total_epochs = None

        if self.verbose:
            print("Now recording experiment #{}".format(self.exp_ID))

    def record_experiment(self,
                          test_results: dict,
                          time_taken: str,
                          epoch_list: list,
                          model: Sequential = None,
                          train_history: list = None):
        """record experiment configuration and statistics"""
        # link references
        if train_history:
            self.history = train_history
        self.average_epochs = float(np.mean(epoch_list))
        self.std_epochs = float(np.std(epoch_list))
        self.time_taken = time_taken

        # create new path in results and experiment folders
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)

        if model:
            self.__save_model(model)

        # append test set metrics to results/exp_results_all.csv
        self.__save_results(test_results)

        # once all of the above done, append experiment info to results/exp_ID_config.csv
        self.__save_exp_config()

        if self.verbose:
            print("Experiment {} recorded successfully!".format(self.exp_ID))

    def save_predictions(self,
                         test_inds: Union[list, np.ndarray],
                         y_pred: Union[list, np.ndarray]) -> None:
        # generate test DataFrame
        test_df = generate_all_feat_df(self.loader, self.configID, inds=test_inds)
        test_df[C.PRED_COL] = y_pred

        # reorder so that false negatives come up first
        test_df.sort_values(["label", C.PRED_COL], ascending=[False, True], inplace=True)

        # save prediction file
        test_df.to_csv(self.pred_path, index=False)

        if self.verbose:
            print("Model test set input and prediction saved successfully!")

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

    def __save_results(self, cv_results: Dict[list]) -> None:
        """calculate and append CV test results to results/exp_results_all.csv"""
        # compute mean and std of CV results
        calculated_results = []
        for metric_name in cv_results:
            calculated_results[metric_name + C.MEAN_SUFFIX] = np.mean(cv_results[metric_name])
            calculated_results[metric_name + C.STD_SUFFIX] = np.std(cv_results[metric_name])

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
        config_df = pd.read_csv(C.EXP_ID_LOG, dtype={C.EXP_ID_COL: int})
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

        return output

    def predict_with_model(self, X_input: np.ndarray) -> np.ndarray:
        """return y probabilities given data with model saved at model_path"""
        assert os.path.exists(self.model_path), "No model saved at {}".format(self.model_path)

        model = load_model(self.model_path)

        # compare only non sample size shapes
        assert model.input_shape[1:] == X_input.shape[1:], \
            "2nd & 3rd dimensions of input ({}) do not align with model input shape ({}).".format(str(model.input_shape[1:]),
                                                                                     str(X_input.shape[1:]))

        return model.predict(X_input)


def _find_next_exp_ID() -> int:
    """helper to find the next unique exp ID in given exp dir"""
    # find ID based on exp_ID_config.csv file
    exp_id_col = C.EXP_COL_CONV[C.EXP_ID_COL]
    try:
        df = pd.read_csv(C.EXP_ID_LOG, usecols=[exp_id_col])
        max_ID = df[exp_id_col].max()
        if math.isnan(max_ID):
            # return 1 if no experiment recorded yet
            return 1
        else:
            return max_ID + 1
    except IOError:
        # if file not exist, create one based on template
        pd.read_csv(C.TEMPLATE_ID_LOG).to_csv(C.EXP_ID_LOG, index=False)
        return 1


def _filter_values(vars_dict: dict)->dict:
    """helper function to filter out dictionary entries whose values are not str, num or bool"""
    return {key: value for key, value in vars_dict.items() if type(value) in C.ACCEPTABLE_TYPES}


if __name__ == "__main__":
    # for debugging only
    train_dict = {'window': 2.0,
                  'ahead': 1.0,
                  'rolling': 0.7,
                  'rate': 50,
                  'gap': 5,
                  'configID': 1,
                  'cal_vel': False,
                  'early_stop': False,
                  'patience': 3,
                  'conv_crit': 'loss',
                  'silent': False,
                  'pbar': False,
                  'no_preds': True,
                  'model': 'lstm',
                  'crash_ratio': 1,
                  'notes': 'n/a'}

    loader = MARSDataLoader(window_size=2.0, time_ahead=1.0)
    recorder = Recorder(loader, train_dict)
    print("recroder stats: ")
    print(vars(recorder))
