#!/usr/bin/env python3
# Author: Yonglin Wang
# Date: 2021/3/21 10:14 PM
# Run prediction on held-out test sets, supports 1) running on test sets with different look ahead times
# and 2) running on new threshold at a specific recall

import os
import re
import pickle
from typing import Union, Tuple

import numpy as np
import tensorflow.keras as keras
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, \
    precision_score, recall_score, accuracy_score, precision_recall_curve

import argparse
import random

import consts as C
from processing.marsdataloader import MARSDataLoader
from recording.recorder import Recorder
from experiment import find_col_inds_for_normalization
from processing.dataset_config import load_dataset

# control for randomness in case of any
np.random.seed(C.RANDOM_SEED)
random.seed(C.RANDOM_SEED)


def get_datasets(loader: MARSDataLoader,
                 normalize_mode: str,
                 config_id: int,
                 seq_label: bool,
                 norm_stats_path: str=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """load test X, y, normalized if specified, and sample indices in entire dataset"""
    # Load original test set X and y
    X, y, _, sample_inds = load_dataset(loader, config_id, test_split=True, seq_label=seq_label, return_indices=True)

    # get cols normalized, if specified
    if normalize_mode != C.NO_NORM:
        assert norm_stats_path is not None
        X = _normalize_X(X, normalize_mode, config_id, norm_stats_path)

    return X, y, sample_inds


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
    pat = re.compile(C.EXP_PREF_PATTERN.format(exp_id))
    exp_dir_match = [dir_name for dir_name in os.listdir(exp_parent_dir) if pat.match(dir_name)]
    if not exp_dir_match:
        raise ValueError(f"Cannot find experiment folder for id {exp_id}")
    if len(exp_dir_match) > 1:
        raise ValueError(f"More than 1 experiment folders found for id {exp_id}; regex pattern: {pat.pattern}")
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


def _save_prediction_results(res: dict, recorder: Recorder,
                             curr_loader: MARSDataLoader, threshold: float, recall_at: float=None) -> int:
    """use result metrics, recorder, test dataset loader to save results to pred file"""
    pred_ID = _find_next_pred_ID()
    # add additional info
    res.update({C.EXP_COL_CONV[C.EXP_ID_COL]: recorder.exp_ID,
                C.PRED_ID_COL: pred_ID,
                "window": curr_loader.window_ms,
                "train ahead": recorder.loader.ahead_ms,
                "pred ahead": curr_loader.ahead_ms,
                "rolling": curr_loader.rolling_ms,
                "gap": curr_loader.time_gap,
                "config id": recorder.configID,
                "decision threshold": threshold,
                "recall for inferring threshold": recall_at if recall_at else "not specified",
                "test set name": C.DATA_SUBDIR,
                "train set name": recorder.dataset_name
                })

    # open results df and save results
    try:
        results_df = pd.read_csv(C.PRED_RES_CSV_PATH)
    except IOError:
        results_df = pd.read_csv(C.TEMPLATE_PRED_RES)

    results_df = results_df.append(res, ignore_index=True)
    results_df.to_csv(C.PRED_RES_CSV_PATH, index=False)
    return pred_ID


def _save_test_predictions(pred_id: int,
                           X: np.ndarray,
                           y_true: np.ndarray,
                           y_proba: np.ndarray,
                           y_pred: np.ndarray,
                           recorder: Recorder,
                           test_ahead: float,
                           threshold: float,
                           test_inds: np.ndarray,
                           save_npz: bool=False,
                           save_csv: bool=True,
                           save_lookahead_windows: bool=False
                           ):
    """Save test set predictions (inputs, probabilities, predicted labels, true labels)"""

    assert X.shape[0] == y_true.shape[0] == y_proba.shape[0] == y_pred.shape[
        0], f"Data lengths (first dim) must match! Instead got: X: {X.shape}, y_true: {y_true.shape}, " \
            f"y_proba: {y_proba.shape}, y_pred: {y_pred.shape}"

    # ### save output as NPZ
    if save_npz:
        pred_path = C.HELDOUT_PRED_NPZ_PATH.format(pred_id, recorder.exp_ID,
                                                   threshold * 100,
                                                   recorder.train_args["window"] * 1000,
                                                   recorder.train_args["ahead"] * 1000,
                                                   test_ahead * 1000)

        # save four matrices into one .npz file
        np.savez_compressed(pred_path,
                             X=X,
                             y_true=y_true,
                             y_proba=y_proba,
                             y_pred=y_pred)

    # ### save output as csv
    # note for sequence labels, all in one file, no sequence labeling, etc.
    true_path = C.HELDOUT_TRUE_PRED_PATH.format(pred_id, recorder.exp_ID,
                                               threshold * 100,
                                               recorder.train_args["window"] * 1000,
                                               recorder.train_args["ahead"] * 1000,
                                               test_ahead * 1000)
    false_path = C.HELDOUT_FALSE_PRED_PATH.format(pred_id, recorder.exp_ID,
                                                   threshold * 100,
                                                   recorder.train_args["window"] * 1000,
                                                   recorder.train_args["ahead"] * 1000,
                                                   test_ahead * 1000)
    if save_csv:
        recorder.save_predictions(test_inds, y_pred, true_preds_path=true_path,
                                  false_preds_path=false_path, custom_ahead=test_ahead,
                                  save_lookahead_windows=save_lookahead_windows)


def _infer_decision_threshold(y_proba: np.ndarray,
                              y_true: np.ndarray,
                              default_dt: float,
                              results_at_recall: float=None) -> float:
    """infer decision threshold based on given recall value, """
    if results_at_recall:
        assert 0 < results_at_recall < 1, "Prediction at Recall value must be a float between 0 and 1 (exclusive)!"
        # ### find threshold corresponding to this recall value
        # generate prec recall pairs at when example proba < (not <=) threshold
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        # find the first closest recall and its position, leave out the last placeholder value
        _, recall_idx = _find_nearest_value_idx(recalls[:-1], results_at_recall)
        # infer threshold from recall position
        threshold = thresholds[recall_idx]
    else:
        threshold = default_dt

    return float(threshold)


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
    print(f"Loading test set from {current_dataset_loader.data_dir}...")
    X_test, y_test, sample_inds = get_datasets(current_dataset_loader,
                                  exp_recorder.train_args["normalize"],
                                  exp_recorder.configID,
                                  seq_label=exp_recorder.using_seq_label,
                                  norm_stats_path=exp_recorder.norm_stats_path)

    print(f"Predicting with model from {exp_recorder.model_path}...")
    model = _load_model(exp_recorder.model_path)

    # get results
    eval_res = model.evaluate(X_test, y_test, return_dict=True, verbose=verbose)
    y_proba = model.predict(X_test)

    # determine decision threshold for converting probability to labels
    default_threshold = exp_recorder.train_args["threshold"]
    threshold = _infer_decision_threshold(y_proba, y_test, default_threshold, results_at_recall=args.at_recall)
    y_pred = y_proba.copy()
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    all_res = compute_test_eval_results(y_pred, y_test, y_proba, eval_res)

    print(f"Evaluation results at {threshold} decision threshold: {all_res}")

    # Save prediction at results <exp_id>_<win>win_<ahead>ahead.npz
    pred_ID = _save_prediction_results(all_res, exp_recorder, current_dataset_loader, threshold, recall_at=args.at_recall)

    if args.save_preds_csv or args.save_preds_npz:
        # save the actual predictions for each data, if needed
        _save_test_predictions(pred_ID, X_test, y_test, y_proba, y_pred, exp_recorder, args.ahead, threshold, sample_inds,
                               save_npz=args.save_preds_npz, save_csv=args.save_preds_csv, save_lookahead_windows=args.save_lookahead_windows)


def compute_test_eval_results(y_pred, y_true, y_proba, eval_res):
    """helper function to compute and return dictionary containing all results"""
    tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()

    # eval_res may be different if we have a different threshold!
    output = {
                "total": int(sum([tn, fp, fn, tp])),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "auc_sklearn": roc_auc_score(y_true.flatten(), y_proba.flatten()),
                C.PAT85R: eval_res[C.PAT85R],
                C.PAT90R: eval_res[C.PAT90R],
                C.PAT95R: eval_res[C.PAT95R],
                C.PAT99R: eval_res[C.PAT99R],
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred)
            }

    try:
        f1 = (2 * output["precision"] * output["recall"])/(output["precision"] + output["recall"])
    except ZeroDivisionError:
        f1 = np.nan
    output["f1"] = f1

    output["crash_size"] = output["tp"] + output["fn"]
    output["noncrash_size"] = output["tn"] + output["fp"]

    return output

def _find_nearest_value_idx(array: np.ndarray, value: float) -> Tuple[float, int]:
    """find the first closest value in array to the given value and its index in array"""
    array = np.asarray(array)
    idx = int((np.abs(array - value)).argmin())
    nearest_recall = float(array[idx])
    return nearest_recall, idx


def main():
    # command line parser
    # noinspection PyTypeChecker
    parser = C.create_template_argparser("Test-set Prediction Program",
                                         description="Run saved model on the specified held out test set")

    parser.add_argument("exp_id",
                        type=int,
                        help="ID of the experiment to load saved model from")
    # Preprocessing flags
    parser.add_argument(
        '--ahead', type=float, default=0, help='prediction timing ahead of event, in seconds')

    # general flags
    parser.add_argument(
        '--silent', action='store_true',
        help='whether to silence custom print statements for training information')
    parser.add_argument(
        '--save_preds_csv', action='store_true',
        help='whether to save model predictions and input as csv files')
    parser.add_argument(
        '--save_lookahead_windows', action='store_true',
        help='whether to save test set lookahead windows in preds output files; only effective if --save_preds_csv is selected.')
    parser.add_argument(
        '--save_preds_npz', action='store_true',
        help='whether to save model predictions (inputs, probabilities, predicted labels, true labels) on the test set as a npz file')
    parser.add_argument(
        '--at_recall', type=float, default=None,
        help='Generate prediction output at decision threshold that yields the specified recall.')

    args = parser.parse_args()

    run_prediction(args)



if __name__ == "__main__":
    main()