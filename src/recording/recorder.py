# !/usr/bin/env python3
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
from typing import Union

import pandas as pd
from tensorflow.keras import Sequential
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

        # to be recorded on record_experiment
        self.model_path = os.path.join(self.exp_dir, C.MODEL_PATH)  # path to model
        self.recorder_path = os.path.join(self.exp_dir, C.REC_PATH)
        self.history = None  # hisotry dict from keras history object, if any passed
        self.time_taken = None
        self.test_inds = None
        self.test_preds = None

        if self.verbose:
            print("Now recording experiment #{}".format(self.exp_ID))

    def record_experiment(self,
                          test_results: dict,
                          time_taken: str,
                          model: Sequential = None,
                          train_history: History = None,
                          test_inds: Union[list, np.ndarray] = None,
                          test_preds: Union[list, np.ndarray] = None):
        """record training process, train history currently optional"""
        # link references
        if train_history:
            self.history = train_history.history
        self.time_taken = time_taken
        self.test_inds = test_inds
        self.test_preds = test_preds

        # create new path in results and experiment folders
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)

        # save model, if any, current syntax only supports Keras model
        if model:
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

        # ### append test_dict (test set results) values to results/exp_results_all.csv
        # add to current results
        test_results[C.EXP_ID_COL] = self.exp_ID
        # retrieve results
        try:
            results_df = pd.read_csv(C.ALL_RES_CSV_PATH)
        except IOError:
            results_df = pd.read_csv(C.TEMPLATE_ALL_RES)
        # append results and save
        results_df = results_df.append(test_results, ignore_index=True)
        results_df.to_csv(C.ALL_RES_CSV_PATH, index=False)

        # ### pickle this object to its path
        pickle.dump(self, open(self.recorder_path, "wb"))

        # once all of the above done, append experiment info to results/exp_ID_config.csv
        # add recorder path and model path to dict for append
        # df.append({"a":3, "c":5}, ignore_index=True)
        config_df = pd.read_csv(C.EXP_ID_LOG)
        config_df = config_df.append(self.__compile_exp_dict(), ignore_index=True)
        config_df.to_csv(C.EXP_ID_LOG, index=False)

        if self.verbose:
            print("Experiment {} recorded successfully!".format(self.exp_ID))

    def plot_history(self, save=True):
        """generate plot for history, can save at experiment path"""
        # TODO

    def save_predictions(self,
                         test_inds: Union[list, np.ndarray] = None,
                         y_pred: Union[list, np.ndarray] = None):
        """output model predictions on test set as CSV file"""
        if not test_inds:
            assert self.test_inds is not None, "Provide test indices or run record_experiment " \
                                               "before calling this fucntion"
            test_inds = self.test_inds

        if not y_pred:
            assert self.test_preds is not None, "Provide test predictions or run record_experiment " \
                                                "before calling this function"
            y_pred = self.test_preds

        # generate test DataFrame
        test_df = generate_all_feat_df(self.loader, self.configID, inds=test_inds)
        test_df[C.PRED_COL] = y_pred

        # reorder so that false negatives come up first
        test_df.sort_values(["label", C.PRED_COL], ascending=[False, True], inplace=True)

        # save prediction file
        test_df.to_csv(self.pred_path, index=False)

        if self.verbose:
            print("Model test set input and prediction saved succesfully!")

    def __compile_exp_dict(self) -> dict:
        """compile experiment configuration dictionary"""
        # put together attributes for extraction
        all_atts = {**vars(self), **vars(self.loader), **self.train_args}
        # loop over ordered columns
        return {C.EXP_COL_CONV[column]: all_atts[column] for column in C.EXP_COL_CONV}


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