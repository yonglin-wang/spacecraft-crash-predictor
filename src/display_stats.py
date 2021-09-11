#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date: 2021/1/31 11:04 PM
"""Displays history and model summary of a given experiment ID"""

import argparse

from recording.recorder import Recorder

def print_history(trainer: Recorder):
    """retrieves and prints model history in console"""
    # TODO print all histories
    # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    pass


def validate_experiment_ID(id: int)->bool:
    """validate whether the trainer file for given exp ID still exists and trainer has necessary info"""
    # TODO validate if Trainer path exists

    # TODO validate if history exists

    # model doesn't have to exist

    pass


def plot_history():
    """"""
    # TODO validate if model path still exists


def print_model_summary():
    """print model.summary only if model file exists"""


def find_diff_and_compare(ids: list):
    """show different str/number attributes in given lists of ids"""
    # ensure all id exists


def validate_id(id: int):
    """ensure given experiment ID can be found in .csv file"""


def summarize_datasets():
    """summarize dataset stats of data sets under data directory, save as .csv"""


def main():
    # command line parser
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(prog="Name of Program",
                                     description="Program Description",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("positional",
                        type=int,
                        help="a positional argument")
    parser.add_argument('--float',
                        type=float,
                        default=0.5,
                        help='optional float with default of 0.5')
    parser.add_argument("-o", "--optional_argument",
                        type=str,
                        default=None,
                        help="optional argument; shorthand o")
    parser.add_argument("-t", "--now_true",
                        action="store_true",
                        help="boolean argument, stores true if specified, false otherwise; shorthand t")

    args = parser.parse_args()


if __name__ == "__main__":
    pass