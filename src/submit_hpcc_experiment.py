#!/usr/bin/env python3
# Author: Yonglin Wang, Selena Lin, Jian Yu
# Date: 1/4/22 1:55 PM
# Automation script for submitting multiple HPCC jobs. Does NOT depend on any other scripts.


from typing import Set
import os
from string import Formatter
from collections import OrderedDict
import time

import pandas as pd


SH_TEMPLATE_PATH = "tmp/sbatch_template.txt"
SH_FILE_PATH = "tmp/sbatch_script.sh"
EXP_PARAM_KEY = "exp_param_string"
SHELL_SEP = "\n"
SLEEP_SEC = 2


def main():
    template = __load_file_template()
    # assumes that any names not in template is considered to be passed to the .py script
    names_in_template = set(
        [field_name for _, field_name, _, _ in Formatter().parse(template) if field_name is not None])

    # print sleep information
    print(f"To prevent concurrent access to results/.expID, "
          f"a sleep period of {SLEEP_SEC} seconds will be added between initiating each process.")

    # process and execute exp config
    configs = pd.read_csv("hpcc_exp_configs.csv")

    # give a rough estimate of when program will end
    print(f"Total {len(configs)} experiments to be run, taking approximately {SLEEP_SEC * len(configs)} seconds.")

    for i, row in configs.iterrows():
        print("\n" + "=*-+" * 5 + f"Now processing experiment at row {i + 1}/{len(configs)}..." + "=*-+" * 5)

        # process bool args and add exp_param str
        raw_exp_info_dict: OrderedDict = row.to_dict(into=OrderedDict)
        exp_info_dict = __add_exp_param_string_to_dict(__process_boolean_options(raw_exp_info_dict), names_in_template)

        # test with following line, to be deleted when running experiments
        # exit_code = execute_echo_test(exp_info_dict)
        # exit_code = os.system("sbatch nonexistent.sh")
        # submit a sbatch job using the following line
        exit_code = execute_sbatch_shell_script(template, exp_info_dict)
        # implement the if statement for exit_code as described in instructions
        if exit_code != 0:
            # raise ValueError, giving exit code
            raise ValueError(f"exit code {exit_code} is not 0! exp_info_dict content: {exp_info_dict}")
        else:
            print(f"Command above finished with exit code {exit_code}")

        # space out the processes to avoid different processes accessing results/.expIDs at the same time
        time.sleep(SLEEP_SEC)

    print(f"\n[Final Message] Congrats! Successfully submitted {len(configs)} jobs!")


def execute_echo_test(processed_exp_dict: dict) -> int:
    """execute an echo test that prints the exp_dict to stdout"""
    # use f-string to format the string for echo command
    assert EXP_PARAM_KEY in processed_exp_dict, f"code assumes dict is processed and has " \
                                                f"{EXP_PARAM_KEY} key, which isn't found."

    echo_exit_code = os.system(f"echo echoing formatted args... {processed_exp_dict[EXP_PARAM_KEY]}")

    return echo_exit_code


def __load_file_template() -> str:
    """load file template from default path"""
    if not os.path.exists(SH_TEMPLATE_PATH):
        raise FileNotFoundError(f"Cannot find shell script template at {os.path.abspath(SH_TEMPLATE_PATH)}. "
                                f"Check instructions for retrieving the file.")

    template = open(SH_TEMPLATE_PATH, "r").read()
    return template


def __add_exp_param_string_to_dict(exp_dict: OrderedDict, template_names: Set, verbose: bool=True) -> OrderedDict:
    """add exp_param_string key to a copy of experiment dict and return it;
    throws error if a key called exp_param_string already exists"""
    if EXP_PARAM_KEY in exp_dict:
        raise ValueError(f"{EXP_PARAM_KEY} found in input dictionary! Avoid using this name in the config file.")

    # generate formatted argument values, while preserving order
    arguments = [f"--{arg_name} {value}".strip() for arg_name, value in exp_dict.items()
               if arg_name not in template_names]

    exp_param_string = " ".join(arguments)

    if verbose:
        print(f"formatted arguments: {exp_param_string}")

    # doesn't change the original dict inplace
    exp_dict_copy = exp_dict.copy()
    exp_dict_copy[EXP_PARAM_KEY] = exp_param_string

    return exp_dict_copy


def __process_boolean_options(exp_dict: OrderedDict) -> OrderedDict:
    """change True values to empty string and pop any keys that has False as its value;
    so that it will either show up in the arguments, e.g. --save_model, or doesn't"""
    for key, value in exp_dict.items():
        # bool only, don't change 0 or 1 values
        if type(value) == bool:
            if value == True:
                # overwrite True with empty string
                exp_dict[key] = ""
            else:
                # delete key from dict
                exp_dict.pop(key)
        else:
            # do nothing to the rest of args
            pass
    return exp_dict


def __generate_temporary_shell_script(template_txt: str, processed_exp_dict: OrderedDict) -> str:
    """generate and save the shell script to tmp folder. returns filled template """
    assert EXP_PARAM_KEY in processed_exp_dict, f"code assumes dict is processed and has the " \
                                                f"'{EXP_PARAM_KEY}' key, which isn't found."

    shell_content = template_txt.format(**processed_exp_dict)
    with open(SH_FILE_PATH, "w") as file:
        file.write(shell_content)

    return shell_content


def execute_sbatch_shell_script(template_txt: str, processed_exp_dict: OrderedDict) -> int:
    """executes the sbatch command using temp sh script saved at SH_FILE_PATH"""

    script_content = __generate_temporary_shell_script(template_txt, processed_exp_dict)

    # print argument line (last line) for preview
    print(f"Last line of shell script (ie command): \n{script_content.split(sep=SHELL_SEP)[-1]}")

    assert os.path.exists(SH_FILE_PATH), f"can't find shell script to execute at {os.path.abspath(SH_FILE_PATH)}"

    sbatch_command = f"sbatch {SH_FILE_PATH}"
    exit_code = os.system(sbatch_command)

    return exit_code


if __name__ == "__main__":
    main()

