#! /usr/bin/python3
# -*- coding: utf-8 -*-

import time

def display_exec_time(begin: float, scr_name: str = ""):
  """
  Displays the execution time
  :param begin:
  :param scr_name:
  """
  exec_time = time.time() - begin

  msg_header = "Execution Time:"
  if scr_name:
    msg_header = scr_name.rstrip() + " " + msg_header

  if exec_time > 60:
    et_m, et_s = int(exec_time / 60), int(exec_time % 60)
    print("\n%s %dm %ds" % (msg_header, et_m, et_s))
  else:
    print("\n%s %.2fs" % (msg_header, exec_time))