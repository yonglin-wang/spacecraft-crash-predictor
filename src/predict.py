#!/usr/bin/env python3
# Author: Yonglin Wang
# Date: 2021/3/21 10:14 PM
# Run prediction on test sets, support running on different test sets

import argparse
import os
import time
import re
import pickle
from typing import Union, Tuple

import numpy as np
import tensorflow.keras as keras
import pandas as pd

from tqdm import tqdm

import argparse
import random

from utils import calculate_exec_time
import consts as C
from processing.marsdataloader import MARSDataLoader
from recording.recorder import Recorder
from experiment import find_col_inds_for_normalization, compute_test_results
from processing.dataset_config import load_dataset

# control for randomness in case of any
np.random.seed(C.RANDOM_SEED)
random.seed(C.RANDOM_SEED)


EXP_PREF_PATTERN = "exp{}_"      # insert exp id here

def generate_model_prediction(X,
                              model: keras.Sequential,
                              threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """predict data given model; return crash probabilities, predicted labels"""

    # get crash probabilities
    y_proba = model.predict(X)
    y_pred = y_proba.copy()
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    return y_proba, y_pred


def get_datasets(loader: MARSDataLoader,
                 normalize_mode: str,
                 config_id: int,
                 norm_stats_path: str=None) -> Tuple[np.ndarray, np.ndarray]:
    """load test X and y, normalized if specified"""
    # Load original test set X and y
    X, y = load_dataset(loader, config_id, test_split=True)

    # get cols normalized, if specified
    if normalize_mode != C.NO_NORM:
        assert norm_stats_path is not None
        X = _normalize_X(X, normalize_mode, config_id, norm_stats_path)

    return X, y


def _normalize_X(X: np.ndarray, norm_mode: str, config_id: int, norm_stats_path: str):
    """normalize given X based on mode"""
    # Get normalization index
    norm_inds = find_col_inds_for_normalization(config_id, norm_mode)
    # get norm stats
    norm_stats = pickle.load(open(norm_stats_path, "rb"))
    # normalize each index
    for col_ind in norm_inds:
        col_mean, col_std = norm_stats[col_ind]["mean"], norm_stats[col_ind]["std"]
        X[:, :, col_ind:col_ind + 1] = (X[:, :, col_ind:col_ind + 1] - col_mean) / col_std

    return X


def _get_recorder(exp_id: int, exp_parent_dir: str) -> Recorder:
    """search and return pickled recorder based on exp id in the given exp directory"""
    # get corresponding folder path for given id, raise error if none or multiple found
    pat = re.compile(EXP_PREF_PATTERN.format(exp_id))
    exp_dir_match = [dir_name for dir_name in os.listdir(exp_parent_dir) if pat.match(dir_name)]
    if not exp_dir_match:
        raise ValueError(f"Cannot find experiment folder for id {exp_id}")
    if len(exp_dir_match) > 1:
        raise ValueError(f"More than 1 experiment folders found for id {exp_id}")
    rec_path = os.path.join(exp_parent_dir, exp_dir_match[0], C.REC_BASENAME)

    # load recorder
    if not os.path.exists(rec_path):
        raise ValueError(f"Recorder object not found at {rec_path}")

    return pickle.load(open(rec_path, "rb"))


def _load_model(model_path: str) -> keras.Sequential:
    """load model from given path"""
    # load model
    model: keras.Sequential = keras.models.load_model(model_path)

    return model


def _find_next_pred_ID() -> int:
    """helper to find the next unique exp ID in given exp dir, fast operation to avoid collision"""
    # find ID based on ID record file
    try:
        with open(C.PRED_ID_RECORD, "r") as id_file:
            next_id = int(id_file.read())
    except IOError:
        next_id = 1

    # save ID to record
    with open(C.PRED_ID_RECORD, 'w') as count_file:
        count_file.write(str(next_id + 1))

    return next_id


def _save_prediction_results(res: dict, recorder: Recorder, curr_loader: MARSDataLoader):
    """use result metrics, recorder, test dataset loader to save results to pred file"""
    # add additional info
    res.update({C.EXP_COL_CONV[C.EXP_ID_COL]: recorder.exp_ID,
                C.PRED_ID_COL: _find_next_pred_ID(),
                "window": curr_loader.window_ms,
                "train ahead": recorder.loader.ahead_ms,
                "pred ahead": curr_loader.ahead_ms,
                "rolling": curr_loader.rolling_ms,
                "gap": curr_loader.time_gap,
                "config id": recorder.configID,
                })

    # open results df and save results
    try:
        results_df = pd.read_csv(C.PRED_RES_CSV_PATH)
    except IOError:
        results_df = pd.read_csv(C.TEMPLATE_PRED_RES)

    results_df = results_df.append(res, ignore_index=True)
    results_df.to_csv(C.PRED_RES_CSV_PATH, index=False)


def _save_test_predicitons(X: np.ndarray,
                           y_true: np.ndarray,
                           y_proba: np.ndarray,
                           y_pred: np.ndarray,
                           recorder: Recorder,
                           test_ahead: int):
    """Save test set predictions (inputs, probabilities, predicted labels, true labels)"""
    pred_path = C.HELDOUT_PRED_PATH.format(recorder.exp_ID,
                                           recorder.train_args["window"] * 1000,
                                           recorder.train_args["ahead"] * 1000,
                                           test_ahead * 1000)
    assert X.shape[0] == y_true.shape[0] == y_proba.shape[0] == y_pred.shape[0], f"Data lengths (first dim) must match! Instead got: X: {X.shape}, y_true: {y_true.shape}, y_proba: {y_proba.shape}, y_pred: {y_pred.shape}"

    # save four matrices into one .npz file
    np.savez_compressed(pred_path,
                         X=X,
                         y_true=y_true,
                         y_proba=y_proba,
                         y_pred=y_pred)



def run_prediction(args: argparse.Namespace):
    """main function of predict.py, loads dataset, runs normalization, and gets predictions"""
    exp_recorder = _get_recorder(args.exp_id, C.EXP_PATH)
    exp_loader = exp_recorder.loader
    verbose = not args.silent

    if args.ahead != 0:
        # if ahead specified, use ahead and other params in recorder to create a placeholder loader
        current_dataset_loader = MARSDataLoader(exp_loader.window, args.ahead, exp_loader.sampling_rate,
                                                time_gap=0, rolling_step=exp_loader.rolling,
                                                verbose=verbose, show_pbar=verbose)
    else:
        # if ahead is not specified, default to ahead in recorder (i.e. use dataset the model is trained on)
        current_dataset_loader = MARSDataLoader(exp_loader.window, exp_loader.ahead, exp_loader.sampling_rate,
                                                exp_loader.time_gap, exp_loader.rolling,
                                                verbose=verbose, show_pbar=verbose)

    # get normalization recorder
    X_test, y_test = get_datasets(current_dataset_loader, exp_recorder.train_args["normalize"], exp_recorder.configID, exp_recorder.norm_stats_path)

    model = _load_model(exp_recorder.model_path)

    # get results
    eval_res = model.evaluate(X_test, y_test, return_dict=True, verbose=verbose)
    y_proba, y_preds = generate_model_prediction(X_test, model, exp_recorder.train_args["threshold"])
    all_res = compute_test_results(y_preds, y_test, y_proba, eval_res)
    all_res["crash_size"] = all_res["tp"] + all_res["fn"]
    all_res["noncrash_size"] = all_res["tn"] + all_res["fp"]

    # Save prediction at results <exp_id>_<win>win_<ahead>ahead.npz
    _save_prediction_results(all_res, exp_recorder, current_dataset_loader)

    if args.save_preds:
        # save the actual predictions for each data, if needed
        _save_test_predicitons(X_test, y_test, y_proba, y_preds, exp_recorder, args.ahead)


def main():
    # command line parser
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(prog="predict.py",
                                     description="Run saved model on the held out test set.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("exp_id",
                        type=int,
                        help="ID of the experiment to load saved model from")
    # Preprocessing flags
    parser.add_argument(
        '--ahead', type=float, default=0, help='prediction timing ahead of event, in seconds')
    parser.add_argument(
        '--silent', action='store_true',
        help='whether to silence custom print statements for training information')
    parser.add_argument(
        '--save_preds', action='store_true',
        help='whether to save model predictions (inputs, probabilities, predicted labels, true labels) on the test set')
    args = parser.parse_args()

    run_prediction(args)



if __name__ == "__main__":
    main()