# -*- coding:utf-8 -*-
"""
User：Ranki Wang
Date: 2021/8/25
"""


# class Base(object):
#  def __init__(self):
#    print('Create Base')
#
# lass A(Base):
#  def __init__(self):
#    # Base.__init__(self)
#    # super(A, self).__init__()
#    super().__init__()      # python3
#    print('Create A')
import glob
from metrics import gzopen


class Base(object):
  """ base """
  def __init__(self):
    """ init """
    print("enter Base")
    print("leave Base")

  def parsedef(self):
    """
    parsedef
    """
    defFile= glob.glob('./*.py')
    for file in defFile:
      if 'super' in file:
        print(file)
        f = gzopen
    #self.__params__(self.parsedef.__doc__)


class A(Base):
  def __init__(self):
    print("enter A")
    super(A, self).__init__()
    print("leave A")


class B(Base):
  def __init__(self):
    print("enter B")
    # super(B, self).__init__()
    print("leave B")


class C(A, B):
  def __init__(self):
    print("enter C")
    super(C, self).__init__()
    print("leave C")


base = Base()
base.parsedef()
# print(base.__init__.__doc__)
# print(base.parsedef.__doc__)
