#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yonglin Wang
# Date: 2021/1/29
"""Constants used in scripts. Crucially, to prevent circular import, this script does not import relative modules."""
import os
from collections import OrderedDict

RANDOM_SEED = 2020

# -----
# Feature extraction Constants
# -----
# names to save each column under, original data if no "calculated" or "normalized" specified in file path
# In principle,if the directory is not tampered, the following arrays should all have shape (n, [sampling_rate])
COL_PATHS = OrderedDict([
                         ('velocity', 'velocity.npy'),  # training, (n_sample, sampling_rate)
                         ('velocity_cal', 'velocity_calculated.npy'),  # training, (n_sample, sampling_rate)
                         ('position', 'position.npy'),  # training, (n_sample, sampling_rate)
                         ('joystick', 'joystick_deflection.npy'),  # training, (n_sample, sampling_rate)
                         ('destabilizing', 'destabilizing_deflection.npy'),  # training, (n_sample, sampling_rate)
                         ('trial_key', 'corresponding_peopleTrialKey.npy'),  # for output reference only (n_samples,)
                         ('start_seconds', 'entry_start_seconds.npy'),  # can be added as feature?  (n_samples,)
                         ('end_seconds', 'entry_end_seconds.npy'),  # can be added as feature?  (n_samples,)
                         ('label', 'label.npy'), # truth label. 1=crash, 0=noncrash (n_sample,)
                         ]
                        )

# lists of initial feature columns generated by generate_feature_files, key names same as COL_PATHS keys
INIT_FEATURES = {"velocity", "velocity_cal", "position", "joystick", "trial_key", "start_seconds", "end_seconds",
                 "label"}

# argparser value checker
MIN_STEP = 0.04

# crash event criteria
MIN_ENTRIES_IN_WINDOW = 2  # minimum # of entries between two crash events (i.e. in a window)

# paths for saving output
DEBUG_FORMAT = "debug_{}ahead_{}window.csv"

# columns we will use for interpolation
COLS_TO_INTERPOLATE = ('currentVelRoll', 'currentPosRoll', 'calculated_vel', 'joystickX')

# -----
# DataLoader Constants
# -----

# velocity mode tag
CALCULATED = "calc"
ORIGINAL = "orig"

# paths for saving output
DATA_OUT_DIR_FORMAT = "data/{}window_{}ahead_{}rolling/"

# path to pickle dataloader, saved under expriment
LOADER_BASENAME = "dataloader.pkl"

# Unique IDs for X Y data preprocessing configuration
CONFIG_IDS = {1, 2, 3}

# train test split config constants
TEST_SIZE = 0.2
USED_COL_FORMAT = "{}_InTrain"

# configuration specific details, if seperate prediction in future
COLS_USED = "cols_used"
CONFIG_OVERVIEW = "config_overview"
CONFIG_SPECS = {
    1: {
        # columns used in configuration; name strings identical to COL_PATHS
        COLS_USED: ['position', 'velocity', 'joystick'],
        # concise description of this configuration
        CONFIG_OVERVIEW: "basic triple (orig vel)"},
    2: {COLS_USED: ['position', 'velocity_cal', 'joystick'],
        CONFIG_OVERVIEW: "basic triple (calc vel)"},
    3: {COLS_USED: ['position', 'velocity', 'joystick', 'destabilizing'],
        CONFIG_OVERVIEW: "basic triple (orig vel)"}
}

# -----
# Model Constants
# -----

# path to save model (to be joined with exp)


# list of available RNN models, all lower-cased
LSTM = "lstm"
GRU = "gru"
MLP = "mlp"
AVAILABLE_MODELS = [LSTM, GRU]
RNN_MODELS = [LSTM, GRU]

# names for metrics
AUC = "auc"
PRECISION = "precision"
RECALL = "recall"
ACC = "accuracy"

# model file name under path
MODEL_PATH = "model"

# default training notes
DEFAULT_NOTES = "None Entered."

# -----
# Recorder Constants
# -----
# general experiment path
EXP_PATH = "exp/"

# experiment path
EXP_FORMAT = os.path.join(EXP_PATH, "exp{}_{}win_{}ahead_conf{}_{}")  # e.g.exp1_1000win_500scale_conf1_lstm"

# experiment directory path for each experiment
RESULT_DIR = "results"

# path for saving predictions
PRED_PATH = os.path.join(RESULT_DIR, "TestSetPred_exp{}_{}win_{}ahead_conf{}_{}.csv")
PRED_COL = "predicted"

# experiment configuration .csv values
EXP_ID_LOG = os.path.join(RESULT_DIR, "exp_ID_config.csv")
TEMPLATE_ID_LOG = os.path.join(RESULT_DIR, "template", "exp_ID_config.csv")
EXP_ID_COL = "exp_ID"
EXP_COL_CONV = OrderedDict([('exp_ID', 'experiment ID'),
                            ('configID', 'dataset config ID'),
                            ('window_ms', 'window size (ms)'),
                            ('ahead_ms', 'time ahead (ms)'),
                            ('rolling_ms', 'rolling step (ms)'),
                            ('sampling_rate', 'sampling rate'),
                            ('time_gap', 'time gap'),
                            ('model', 'model type'),
                            ('crash_ratio', 'crash ratio'),
                            ('notes', 'experiment notes'),
                            ('exp_date', 'date of experiment'),
                            ('time_taken', 'training time'),
                            ('total_sample_size', 'total samples'),
                            ('crash_sample_size', 'crash sample size'),
                            ('noncrash_sample_size', 'noncrash sample size'),
                            ('cal_vel', 'using calculated velocity'),
                            ('early_stop', 'early stopping'),
                            ('conv_crit', 'convergence criteria'),
                            ('data_dir', 'data directory'),
                            ('exp_dir', 'experiment directory'),
                            ('pred_path', 'prediction path'),
                            ('model_path', 'model save path'),
                            ('recorder_path', 'recorder save path')])

# results .csv values
RES_COLS = ['accuracy', 'precision', 'recall', 'auc', 'f1', 'tn', 'fp', 'fn', 'tp']
ALL_RES_CSV_PATH = os.path.join(RESULT_DIR, "exp_results_all.csv")
TEMPLATE_ALL_RES = os.path.join(RESULT_DIR, "template", "exp_results_all.csv")

# name for recorder file
REC_PATH = "recorder.pkl"
