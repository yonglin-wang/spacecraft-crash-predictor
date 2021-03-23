# Author: Yonglin Wang
# Date: 2021/1/29
"""Constants used in scripts. Crucially, to prevent circular import, this script does not import relative modules."""
import os
from collections import OrderedDict

import numpy as np

RANDOM_SEED = 2020


# -----
# Raw data segmentation constants
# -----
SEGMENT_COLS = ['seconds', 'trialPhase', 'peopleName', 'peopleTrialKey']
ID = "id"
OUTPUT_COLS = ["duration", "reading_num", "phase", "crash_ind", "subject", "trial_key"]
SEGMENT_DICT_PATH = os.path.join("data", "segment_dict.pkl")
SEGMENT_STATS_PATH = os.path.join("data", "segment_stats.csv")

# -----
# Feature extraction Constants
# -----
RAW_DATA_PATH = os.path.join("data", "data_all.csv")
ESSENTIAL_RAW_COLS = ['seconds', 'trialPhase', 'currentPosRoll', 'currentVelRoll', 'calculated_vel', 'joystickX',
                      'peopleName', 'peopleTrialKey']


# names to save each column under, original data if no "calculated" or "normalized" specified in file path
# In principle,if the directory is not tampered, the following arrays should all have shape (n, [sampling_rate])
COL_PATHS = OrderedDict([
                         ('velocity', 'velocity.npy'),  # training, (n_sample, sampling_rate)
                         ('velocity_cal', 'velocity_calculated.npy'),  # training, (n_sample, sampling_rate)
                         ('position', 'position.npy'),  # training, (n_sample, sampling_rate)
                         ('joystick', 'joystick_deflection.npy'),  # training, (n_sample, sampling_rate)
                         ('destabilizing', 'destabilizing_deflection.npy'),  # training, (n_sample, sampling_rate)
                         ('trial_key', 'corresponding_peopleTrialKey.npy'),  # for output reference only (n_samples,)
                         ('person', 'subject_name.npy'),
                         ('start_seconds', 'entry_start_seconds.npy'),  # can be added as feature?  (n_samples,)
                         ('end_seconds', 'entry_end_seconds.npy'),  # can be added as feature?  (n_samples,)
                         ('label', 'label.npy'), # truth label. 1=crash, 0=noncrash (n_sample,)
                         ]
                        )

INDS_PATH = {"train": "train_inds.npy", "test": "test_inds.npy"}

# lists of initial feature columns generated by generate_feature_files, key names same as COL_PATHS keys
INIT_FEATURES = {"velocity", "velocity_cal", "position", "joystick", "trial_key", "start_seconds", "end_seconds",
                 "label", "person"}

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
VAL_SIZE = 0.1

# for marking columns used in training in the output prediction file
USED_COL_FORMAT = "{}_InTrain"

# configuration specific details
COLS_USED = "cols_used"
CONFIG_OVERVIEW = "config_overview"
CONFIG_SPECS = {
    1: {
        # columns used in configuration; Requirements:
        # 1) name strings identical to keys in COL_PATHS
        # 2) order identical to output in dataset_config.py
        COLS_USED: ['velocity', 'position', 'joystick'],
        # concise description of this configuration
        CONFIG_OVERVIEW: "basic triple (orig vel)"},
    2: {COLS_USED: ['velocity_cal', 'position', 'joystick'],
        CONFIG_OVERVIEW: "basic triple (calc vel)"},
    3: {COLS_USED: ['velocity', 'position', 'joystick', 'destabilizing'],
        CONFIG_OVERVIEW: "basic triple (orig vel) + destab."}
}

# parameters related to feature normalization
NO_NORM = 'disable'
NORM_LARGE = 'large'
NORM_ALL = 'all'
NORMALIZATION_MODES = [NO_NORM, NORM_LARGE, NORM_ALL]
# features with larger values for partial normalization
LARGE_VAL_FEATS = {'position', 'velocity', 'velocity_cal'}
CATEGORICAL_FEATS = {'destabilizing'}

# -----
# Model Training Constants
# -----

# convergence criteria, must be consistent with what EarlyStopping supports
VAL_LOSS = "val_loss"
VAL_AUC = "val_auc"
VAL_RECALL = "val_recall"
CONV_CRIT = [VAL_AUC, VAL_LOSS, VAL_RECALL]
CONV_MODE = {VAL_LOSS: "min",
             VAL_AUC: "max",
             VAL_RECALL: "max"}

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
DEFAULT_NOTES = "none entered."

# -----
# Recorder Constants
# -----
# CV Strategies
NO_CV = "disable"
KFOLD = "kfold"
LEAVE_OUT = "leave_out"
CV_OPTIONS = [NO_CV, KFOLD, LEAVE_OUT]
# general experiment path
EXP_PATH = "exp/"

# experiment path
EXP_FORMAT = os.path.join(EXP_PATH, "exp{}_{}win_{}ahead_conf{}_{}")  # e.g.exp1_1000win_500scale_conf1_lstm"

# experiment directory path for each experiment
RESULT_DIR = "results"

# path for saving predictions
PRED_PATH = os.path.join(RESULT_DIR, "TestSetPred_exp{}_{}win_{}ahead_conf{}_{}.csv")
PRED_COL = "predicted"

# path for normalization stats
NORM_STATS_PATH = "normalization_mean_std.pkl"

# experiment configuration .csv values
ACCEPTABLE_TYPES = {int, str, bool, float, np.float}
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
                            ('early_stop', 'early stopping'),
                            ('conv_crit', 'convergence criteria'),
                            ('data_dir', 'data directory'),
                            ('exp_dir', 'experiment directory'),
                            ('pred_path', 'prediction path'),
                            ('model_path', 'model save path'),
                            ('recorder_path', 'recorder save path')])
# name of stray columns not from namespace
CONFIG_DESC_COL_NAME = "dataset config desc"

# results .csv values
PERF_METRIC = 'auc_sklearn'
RES_COLS = ['auc_tf', 'auc_sklearn', 'precision', 'recall', 'f1', 'accuracy', 'tn', 'fp', 'fn', 'tp', 'total']
ALL_RES_CSV_PATH = os.path.join(RESULT_DIR, "exp_results_all.csv")
TEMPLATE_ALL_RES = os.path.join(RESULT_DIR, "template", "exp_results_all.csv")
MEAN_SUFFIX = "_mean"
STD_SUFFIX = "_std"

# name for recorder file
REC_BASENAME = "recorder.pkl"

# file for recording unique IDs
EXP_ID_RECORD = os.path.join(RESULT_DIR, ".expIDs")

# -----
# Splitter Constants
# -----
SPLIT_MODES = [KFOLD, LEAVE_OUT, NO_CV]

# -----
# Summary Constants
# -----
COMBINED_FILE_FORMAT = os.path.join(RESULT_DIR, "hpcc_results{}.csv")