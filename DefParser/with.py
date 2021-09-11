# -*- coding:utf-8 -*-
"""
User：Ranki Wang
Date: 2021/9/3
"""


class Sample:

  def __enter__(self):
    print("In __enter__()")
    return self

  def __exit__(self, exit_type, exit_value, exit_trace):
    if not exit_type is None: print("type:", exit_type)
    if not exit_value is None: print("value:", exit_value)
    if not exit_trace is None: print("trace:", exit_trace)
    print("In __exit__()")

  def do_something(self):
    bar = 1 / 2
    return bar + 10


def get_sample():
  return Sample()


with Sample() as sample:
  sample.do_something()
