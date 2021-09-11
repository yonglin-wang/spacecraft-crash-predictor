# Date: 2021/1/29
"""Non-experiment helper functions"""

import time
import argparse
from typing import Union, List


def calculate_exec_time(begin: float, scr_name: str = "", verbose=True) -> str:
    """Displays the execution time"""
    exec_time = time.time() - begin

    if exec_time > 60:
        min_str, sec_str = int(exec_time / 60), int(exec_time % 60)
        time_str = "{}m {}s".format(min_str, sec_str)

    else:
        time_str = "{:.2f}s".format(exec_time)

    if verbose:
        msg_header = "Execution Time:"
        if scr_name:
            msg_header = scr_name.rstrip() + " " + msg_header
        print("{} {}".format(msg_header, time_str))

    return time_str

def parse_layer_size(arg_input: str) -> Union[List[int], None]:
    """convert layer size input to a list of integer sizes. Raises errors if string is not the correct form.
    Returns None if input is None."""
    if arg_input is None:
        return None

    nums = arg_input.split(",")
    try:
        return [int(num) for num in nums]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{arg_input} does not have the correct format of integers separated by commas. Acceptable forms includes: 3,4,5 or 3")
