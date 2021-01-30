#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Yonglin Wang
# Date: 2021/1/29
"""Non-experiment helper functions"""

import time


def display_exec_time(begin: float, scr_name: str = ""):
    """Displays the execution time"""
    exec_time = time.time() - begin

    msg_header = "Execution Time:"
    if scr_name:
        msg_header = scr_name.rstrip() + " " + msg_header

    if exec_time > 60:
        et_m, et_s = int(exec_time / 60), int(exec_time % 60)
        print("\n%s %dm %ds" % (msg_header, et_m, et_s))
    else:
        print("\n%s %.2fs" % (msg_header, exec_time))
