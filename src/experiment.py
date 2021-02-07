#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yonglin Wang
# Date: 2021/1/29
"""main class for parsing commandline input, creating DataLoader and training a model"""

import os
import argparse
import time
from typing import Union

import numpy as np
import keras

import tensorflow as tf
from sklearn.metrics import confusion_matrix

import consts as C
from utils import display_exec_time
from processing.marsdataloader import MARSDataLoader
from processing.dataset_config import load_splits
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
    train_inds, test_inds, X_train, X_test, y_train, y_test = load_splits(loader, args.configID)

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

    # early stopping to save up GPU, stops training when there's no loss reduction in given epochs
    if args.early_stop:
        callback = [tf.keras.callbacks.EarlyStopping(monitor=args.conv_crit, patience=args.patience)]
    else:
        callback = None

    # ### start training
    # build model
    if args.model in C.RNN_MODELS:
        model = build_keras_rnn(X_train.shape[1],
                                X_train.shape[2],
                                rnn_out_dim=args.hidden_dim,
                                dropout_rate=args.dropout,
                                rnn_type=args.model)
    else:
        raise NotImplementedError("Model {} has not been implemented!".format(args.model))

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

    time_str = display_exec_time(begin, scr_name="model.py")

    # ### Evaluate model
    if not args.silent:
        print("Now evaluating, metrics used: {}".format(model.metrics_names))

    # generate result stats
    eval_res = model.evaluate(X_test, y_test, return_dict=True, verbose=int(args.pbar))
    y_pred = model.predict(X_test)
    y_pred[y_pred >= args.threshold] = 1
    y_pred[y_pred < args.threshold] = 0
    results = compute_test_results(y_pred, y_test, eval_res)

    # print test set results:
    if not args.silent:
        print("Evaluation done, results: {}".format(eval_res))

    # record experiment
    recorder.record_experiment(results, time_str, model, train_history=history, test_inds=test_inds, test_preds=y_pred)

    # output experiment if needed, same as recorder.save_predictions(test_inds=test_inds, y_pred=y_pred)
    if args.save_output:
        recorder.save_predictions()


def print_training_info(args: argparse.Namespace):
    """helper function to print training information"""
    print("Training information:")
    print(f"Now training model with {int(args.window * 1000)}ms scale, {int(args.ahead * 1000)}ms ahead.\n"
          f"Early Stopping? {args.early_stop}\n"
          f"Using Dataset Configuration #{args.configID}\n")

    if args.early_stop:
        print(f"Stopping early if no {args.conv_crit} improvement in {args.patience} epochs.\n")


def compute_test_results(y_pred, y_true, eval_res):
    """helper function to compute and return dictionary with following keys:
    accuracy,precision,recall,auc,f1,tn,fp,fn,tp
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "total": int(y_pred.shape[0]),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": eval_res[C.ACC],
        "precision": eval_res[C.PRECISION],
        "recall": eval_res[C.RECALL],
        "auc": eval_res[C.AUC],
        "f1": (2 * eval_res[C.PRECISION] * eval_res[C.RECALL])/(eval_res[C.PRECISION] + eval_res[C.RECALL])
    }


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
        '--gap', type=int, default=5, help='minimal time gap allowed between two crash events for data extraction')

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
        '--conv_crit', type=str.lower, default='loss', choices=['loss'],
        help='type of convergence criteria, if early stopping')
    argparser.add_argument(
        '--silent', action='store_true',
        help='whether to silence custom print statements for training information')

    # Model specific flags
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

    # Experiment annotation
    argparser.add_argument(
        '--notes', type=str.lower, default=C.DEFAULT_NOTES,
        help='Notes for this experiment')
    argparser.add_argument(
        '--save_output', action='store_true',
        help='whether to save model test input and output as .csv data')
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
