# -*- coding:utf-8 -*-
"""
User：Ranki Wang
Date: 2021/9/6
"""

from collections import defaultdict


class Top(object):
  def __init__(self):
    self.name = 'chip'
    self.instanceList = defaultdict(dict)

  def parsedef(self):
    a0 = tile('test1')
    self.instanceList[a0.name] = a0


class tile(object):
  def __init__(self, name):
    self.name = name
    self.full_name = 'ce_c/' + self.name


b = Top()
b.parsedef()
print(b.instanceList['test1'].name)
print(b.instanceList['test1'].full_name)
for obj in b.instanceList:
  print(obj)