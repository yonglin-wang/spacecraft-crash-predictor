# Author: Yonglin Wang
# Date: 2021/1/29
"""Non-experiment helper functions"""

import time


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
