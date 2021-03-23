#!/usr/bin/env python3
# Author: Yonglin Wang
# Date: 2021/1/29
"""main class for parsing commandline input, creating DataLoader and training a model"""

import os
import argparse
import time
from typing import Tuple
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score

import consts as C
from utils import calculate_exec_time
from processing.marsdataloader import MARSDataLoader
from processing.dataset_config import load_dataset
from processing.split_data import Splitter
from modeling.rnn import build_keras_rnn
from recording.recorder import Recorder

# Control for random states as much as possible, though on GPU the outputs are unavoidably non-deterministic [1]
# [1]: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import random

os.environ['PYTHONHASHSEED'] = str(C.RANDOM_SEED)
random.seed(C.RANDOM_SEED)  # python's built-in pseudo-random generator
np.random.seed(C.RANDOM_SEED)  # numpy pseudo-random generator
tf.random.set_seed(C.RANDOM_SEED)  # tensorflow pseudo-random generator


def train(args: argparse.Namespace):
    # ### Begin script
    # confirm TensorFlow sees the GPU
    # assert 'GPU' in str(device_lib.list_local_devices()), "TensorFlow cannot find GPU"
    #
    # # confirm Keras sees the GPU (for TensorFlow 1.X + Keras)
    # assert len(tensorflow_backend._get_available_gpus()) > 0, "Keras cannot find GPU"

    # ### time the script
    begin = time.time()

    # ### prepare data and other args
    loader = MARSDataLoader(args.window, args.ahead, args.rate, args.gap, args.rolling,
                            verbose=not args.silent, show_pbar=args.pbar)
    recorder = Recorder(loader, vars(args), verbose=not args.silent)


    # calculate class weight:
    if args.crash_ratio >= 1:
        class_weight = {0: 1, 1: args.crash_ratio}
    elif args.crash_ratio <= 0:
        raise ValueError("Crash ratio must be positive!")
    else:
        class_weight = {0: 1 - args.crash_ratio, 1: args.crash_ratio}

    # prepare args
    if args.pbar:
        fit_ver = 1
    else:
        fit_ver = 2  # only one line per epoch in output file

    X_all, y_all = load_dataset(loader, args.configID)

    # get generator for splitting
    splitter = Splitter(args.cv_mode, args.cv_splits, verbose=loader.verbose)
    if args.cv_mode == C.NO_CV or args.cv_mode == C.KFOLD:
        split_gen = splitter.split_ind_generator(y_all)
    elif args.cv_mode == C.LEAVE_OUT:
        split_gen = splitter.split_ind_generator(y_all, loader.retrieve_col("person"))
    else:
        raise NotImplementedError

    # create results dictionary
    all_split_results = OrderedDict([(metric_name, []) for metric_name in C.RES_COLS])

    # record best performing stats and model for saving results
    best_test_inds = None
    best_y_preds = None
    best_model = None
    best_norm_stat = None
    best_split_num = -1
    best_performance_metric = 0
    all_epochs = []
    all_training_history = []

    for split_number, (train_inds, test_inds) in enumerate(split_gen):
        # validate data size
        assert loader.total_sample_size == train_inds.shape[0] + test_inds.shape[0], "train test numbers don't add up"

        if loader.verbose:
            print(f"Now training split #{split_number + 1} out of total {args.cv_splits}...")

        # train split
        train_hist, model, val_results, y_preds, normalization_stats = train_one_split(args, loader,
                                                                          X_all, y_all,
                                                                          train_inds, test_inds,
                                                                          fit_ver, class_weight)
        # append split results
        for metric_name in all_split_results:
            all_split_results[metric_name].append(val_results[metric_name])

        # record stats
        all_training_history.append(train_hist.history)
        all_epochs.append(len(train_hist.epoch))

        # compare results and save best performing set of objects for recording
        if val_results[C.PERF_METRIC] > best_performance_metric:
            best_performance_metric = val_results[C.PERF_METRIC]
            best_test_inds, best_y_preds, best_model, best_split_num, best_norm_stat = \
                test_inds, y_preds, model, split_number, normalization_stats

    # record experiment outputs
    assert best_split_num >= 0 \
           and best_test_inds is not None \
           and best_y_preds is not None \
           and best_performance_metric > 0, "update best failed"
    time_str = calculate_exec_time(begin, scr_name=__file__, verbose=loader.verbose)
    recorder.record_experiment(all_split_results,
                               time_str,
                               all_epochs,
                               best_split_num,
                               model=best_model,
                               norm_stats=best_norm_stat,
                               train_history=all_training_history,
                               save_model=args.save_model)

    # output predictions and input to csv if needed
    if args.save_preds:
        recorder.save_predictions(best_test_inds, best_y_preds)


def train_one_split(args: argparse.Namespace,
                    loader: MARSDataLoader,
                    X_all: np.ndarray,
                    y_all: np.ndarray,
                    train_inds: np.ndarray,
                    test_inds: np.ndarray,
                    fit_ver: int,
                    class_weight: dict
                    ) -> Tuple[tf.keras.callbacks.History, tf.keras.Sequential, dict, np.ndarray, dict]:
    """function that trains a model and returns train history, the model, and test set results dictionary"""
    # get train and test data from indices
    X_train = X_all[train_inds]
    X_test = X_all[test_inds]
    y_train = y_all[train_inds]
    y_test = y_all[test_inds]

    # normalize features if needed
    norm_stats = None
    if args.normalize != C.NO_NORM:
        norm_feat_inds = find_col_inds_for_normalization(args.configID, args.normalize)
        X_train, X_test, norm_stats = normalize_col(norm_feat_inds, X_train, X_test)

    if loader.verbose:
        # print split info
        print_split_info(train_inds, test_inds, X_train, X_test, y_train, y_test)

    if args.early_stop:
        callback = [tf.keras.callbacks.EarlyStopping(monitor=args.conv_crit,
                                                     patience=args.patience,
                                                     mode=C.CONV_MODE[args.conv_crit])]
    else:
        callback = None

    # ### start training
    # build model
    model = match_and_build_model(args, X_train)

    # train model
    history = model.fit(
        X_train, y_train,
        epochs=args.max_epoch,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=fit_ver,
        shuffle=False,
        class_weight=class_weight,
        callbacks=callback
    )

    # Evaluate model
    if not args.silent:
        print("Now evaluating, metrics used: {}".format(model.metrics_names))

    # generate result stats
    eval_res = model.evaluate(X_test, y_test, return_dict=True, verbose=int(args.pbar))
    y_pred_proba = model.predict(X_test)
    y_pred = y_pred_proba.copy()
    y_pred[y_pred >= args.threshold] = 1
    y_pred[y_pred < args.threshold] = 0
    results = compute_test_results(y_pred, y_test, y_pred_proba, eval_res)

    # print test set results:
    if not args.silent:
        print("Evaluation done, results: {}".format(eval_res))

    return history, model, results, y_pred, norm_stats


def normalize_col(col_inds: list,
                  train_data: np.ndarray,
                  test_data: np.ndarray,
                  inplace=False) -> Tuple[np.ndarray, np.ndarray, dict]:
    assert train_data.ndim == 3 and test_data.ndim == 3, "train test arrays passed not 3D"

    if not inplace:
        train_data = train_data.copy()
        test_data = test_data.copy()

    # save to {col_ind: {"mean": float, "std": float}}
    norm_stats = dict()
    for col_ind in col_inds:
        # find mean and std using training set
        train_mean = np.mean(train_data[:, :, col_ind:col_ind + 1])
        train_std = np.std(train_data[:, :, col_ind:col_ind + 1])

        norm_stats = {col_ind: {"mean": train_mean, "std": train_std}}

        # use mean and std to normalize both train and test data
        train_data[:, :, col_ind:col_ind + 1] = (train_data[:, :, col_ind:col_ind + 1] - train_mean) / train_std
        test_data[:, :, col_ind:col_ind + 1] = (test_data[:, :, col_ind:col_ind + 1] - train_mean) / train_std

        # assertion to make sure col normalized correctly
        assert np.isclose(np.std(train_data[:, :, col_ind:col_ind + 1]), 1) and \
               np.isclose(np.mean(train_data[:, :, col_ind:col_ind + 1]), 0)

    return train_data, test_data, norm_stats


def find_col_inds_for_normalization(config_ID: int, mode: str) -> list:
    assert config_ID in C.CONFIG_SPECS, f"Cannot recognize configuration ID {config_ID}"
    assert mode in C.NORMALIZATION_MODES

    feat_names = C.CONFIG_SPECS[config_ID][C.COLS_USED]

    if mode == C.NORM_ALL:
        # return indices of all non categorical features
        return [ind for ind, feat in enumerate(feat_names) if feat not in C.CATEGORICAL_FEATS]
    elif mode == C.NORM_LARGE:
        # return indices of all present large-value features
        return [ind for ind, feat in enumerate(feat_names) if feat in C.LARGE_VAL_FEATS]
    else:
        raise NotImplementedError()


def match_and_build_model(args, X_train: np.ndarray)->tf.keras.Sequential:
    """helper function to return correct model based on given input"""
    if args.model in C.RNN_MODELS:
        model = build_keras_rnn(X_train.shape[1],
                                X_train.shape[2],
                                rnn_out_dim=args.hidden_dim,
                                dropout_rate=args.dropout,
                                rnn_type=args.model)
    else:
        raise NotImplementedError("Model {} has not been implemented!".format(args.model))

    return model


def compute_test_results(y_pred, y_true, y_proba, eval_res):
    """helper function to compute and return dictionary containing all results"""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    try:
        f1 = (2 * eval_res[C.PRECISION] * eval_res[C.RECALL])/(eval_res[C.PRECISION] + eval_res[C.RECALL])
    except ZeroDivisionError:
        f1 = np.nan

    output = {
                "total": int(y_pred.shape[0]),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "accuracy": eval_res[C.ACC],
                "precision": eval_res[C.PRECISION],
                "recall": eval_res[C.RECALL],
                "auc_tf": eval_res[C.AUC],
                "auc_sklearn": roc_auc_score(y_true, y_proba),
            }

    try:
        f1 = (2 * eval_res[C.PRECISION] * eval_res[C.RECALL])/(eval_res[C.PRECISION] + eval_res[C.RECALL])
    except ZeroDivisionError:
        f1 = np.nan

    output["f1"] = f1

    # assert no difference between this and predefined output columns
    assert not set(output.keys()).difference(set(C.RES_COLS)), "output eval metrics don't match existing columns"

    return output


def print_training_info(args: argparse.Namespace):
    """helper function to print training information"""
    print("\n" + "*=*" * 20)
    print("Training information:")
    print(f"Now training model with {int(args.window * 1000)}ms scale, {int(args.ahead * 1000)}ms ahead.\n"
          f"Early Stopping? {args.early_stop}\n"
          f"Using Dataset Configuration #{args.configID}")

    if args.early_stop:
        print(f"Stopping early if no {args.conv_crit} improvement in {args.patience} epochs.")

    print("Note to this experiment: {}".format(args.notes))

    if args.cv_mode == C.NO_CV or args.cv_splits == 1:
        print(f"No cross validation, using a default {C.VAL_SIZE} test size split.")
    elif args.cv_mode == C.KFOLD:
        print(f"Currently training with {args.cv_splits}-fold cross validation.")
    elif args.cv_mode == C.LEAVE_OUT:
        print(f"Currently training with leave n out with total of {args.cv_splits} splits.")
    else:
        raise NotImplementedError(f"cannot recognize {args.cv_mode}")

    print("")


def print_split_info(inds_train, inds_test, X_train, X_test, y_train, y_test):
    """print shapes of split"""
    print(
        "Train-test split Information\n"
        "Total sample size: {:>16}\n"
        "Train sample size: {:>16}\n"
        "Train crash number: {:>15}\n"
        "Test sample size: {:>17}\n"
        "Test crash number: {:>16}\n"
        "Input shapes:\n"
        "X_train shape: {:>20}\n"
        "X_test shape: {:>21}\n"
        "y_train shape: {:>20}\n"
        "y_test shape: {:>21}\n".
            format(inds_train.shape[0] + inds_test.shape[0],
                   inds_train.shape[0],
                   int(sum(y_train==1)),
                   inds_test.shape[0],
                   int(sum(y_test == 1)),
                   str(X_train.shape),
                   str(X_test.shape),
                   str(y_train.shape),
                   str(y_test.shape))
    )


def main():

    # Argparser
    # noinspection PyTypeChecker
    argparser = argparse.ArgumentParser(prog="Experiment Argparser",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Preprocessing flags
    argparser.add_argument(
        '--window', type=float, default=2.0, help='size of sliding window used in training data, in seconds')
    argparser.add_argument(
        '--ahead', type=float, default=1.0, help='prediction timing ahead of event, in seconds')
    argparser.add_argument(
        '--rolling', type=float, default=0.7, help='length of rolling step between two sliding windows, in seconds')
    argparser.add_argument(
        '--rate', type=int, default=50, help='sampling rate used in window interpolation')
    argparser.add_argument(
        '--gap', type=int, default=0, help='minimal time gap allowed between two crash events for data extraction. '
                                           'Will be recalculated to be at least [2 * window_size + time_ahead].'
                                           'Set to 0 to get auto calculation. ')

    # General training flags
    argparser.add_argument(
        '--configID', type=int, default=1,
        help='dataset configuration ID for model training')
    argparser.add_argument(
        '--early_stop', action='store_true',
        help='whether to stop early training when converged')
    argparser.add_argument(
        '--patience', type=int, default=3,
        help='max number of epochs allowed with no improvement, if early stopping')
    argparser.add_argument(
        '--conv_crit', type=str.lower, default=C.VAL_AUC, choices=C.CONV_CRIT,
        help='type of convergence criteria, if early stopping')
    argparser.add_argument(
        '--silent', action='store_true',
        help='whether to silence custom print statements for training information')

    # Model specific flags
    argparser.add_argument(
        '--normalize', type=str.lower, default=C.NO_NORM, choices=C.NORMALIZATION_MODES,
        help=f'Normalization modes: '
             f'{C.NO_NORM}: no normalization; '
             f'{C.NORM_ALL}: normalize all non-categorical features; '
             f'{C.NORM_LARGE}: normalize the following large-value features, if present: {", ".join(C.LARGE_VAL_FEATS)}')
    argparser.add_argument(
        '--model', type=str.lower, default=C.LSTM, choices=C.AVAILABLE_MODELS,
        help='type of model used')
    argparser.add_argument(
        '--crash_ratio', type=float, default=1,
        help='if >= 1, processed as 1:ratio noncrash-crash ratio; '
             'if < 1 and > 0, processed as (1-ratio):ratio noncrash-crash ratio')
    argparser.add_argument(
        '--dropout', type=float, default=0.5,
        help='dropout rate in RNNs')
    argparser.add_argument(
        '--hidden_dim', type=int, default=100,
        help='hidden dimensions in RNNs')
    argparser.add_argument(
        '--batch_size', type=int, default=256,
        help='batch size for training keras model')
    argparser.add_argument(
        '--threshold', type=float, default=0.5,
        help='decision boundary for binary labels')
    argparser.add_argument(
        '--max_epoch', type=int, default=50,
        help='highest number of epochs allowed in experiment')
    argparser.add_argument(
        '--cv_mode', type=str.lower, default=C.KFOLD, choices=C.CV_OPTIONS,
        help=f'cv mode to use. {C.NO_CV}: no CV; {C.KFOLD}: stratified K-fold; {C.LEAVE_OUT}: leave N subject(s) out')
    argparser.add_argument(
        '--cv_splits', type=int, default=10,
        help='total number of splits in CV strategy. A split number of 1 is the same as disable CV.')

    # Experiment annotation
    argparser.add_argument(
        '--notes', type=str.lower, default=C.DEFAULT_NOTES,
        help='Notes for this experiment')
    argparser.add_argument(
        '--save_preds', action='store_true',
        help='whether to save model test input and output as .csv data')
    argparser.add_argument(
        '--save_model', action='store_true',
        help='whether to save model in no CV or best performing model in CV')
    argparser.add_argument(
        '--pbar', action='store_true',
        help='whether to display progress bar during training. If not selected, outputs one line per epoch.')

    args = argparser.parse_args()

    if args.silent:
        print("Currently in silent mode. Only Tensorflow progess will be printed.")
    else:
        print_training_info(args)

    train(args)

if __name__=="__main__":
    main()
