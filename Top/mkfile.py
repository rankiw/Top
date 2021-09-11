# -*- coding:utf-8 -*-
"""
User：Ranki Wang
Date: 2021/9/8
"""
import os, subprocess
import gzip
import sys


def MkFile_open(file_name, mode='w+', type='rpt', username='--'):
  if username == '--':
    username = os.popen("whoami").readlines()[0].strip()
    if os.name == "nt": username = username.split("\\")[-1]
  file_name_only = file_name
  file_name = '/'.join([username, type, file_name])
  file_dir = '/'.join(file_name.split('/')[0:-1])
  if not os.path.exists(file_dir):
    os.makedirs(file_dir)
  if 'gz' in file_name_only:
    file_handle = gzip.open(file_name, 'wb').read()
  else:
    file_handle = open(file_name, mode)
  return file_handle


class MkFile(object):
  """
      Class support kinds of file operations:
      1. Automatically return file handle as username/type/filename
      2. Support open gz file for read
      3. use param gzip=True to gzip output file

      All files will be created under username/(rpt|log|data)/file

      Usage/Example:
          1. with MkFile(file_name, mode = 'w+', type = 'rpt', username = 'USER') as f:
          2. with MkFile(file_name, mode = 'rb', type = 'date', username = 'USER' ) as f:
          2. with MkFile(file_name, mode = 'w', type = 'date', username = 'USER',gzip=True ) as f:
  """

  def __init__(self, file_name, mode='w+', type='rpt', gzip=False, username='--'):
    if username == '--':
      self.username = os.popen("whoami").readlines()[0].strip()
      if os.name == "nt": self.username = self.username.split("\\")[-1]
    self.filename = '/'.join([self.username, type, file_name])
    self.file_name = file_name
    self.mode, self.gzip = mode, gzip

  def __enter__(self):
    file_dir = '/'.join(self.filename.split('/')[0:-1])
    if not os.path.exists(file_dir):
      os.makedirs(file_dir)
    if 'r' in self.mode and not os.path.exists(self.filename):
      os.system('touch %s' % self.filename)
    if 'gz' in self.file_name:
      self.file_handle = gzip.open(self.filename, self.mode)
    else:
      self.file_handle = open(self.filename, self.mode)
    return self.file_handle

  def __exit__(self, *para):
    self.file_handle.close()
    if self.gzip:
      #  print('gzip %s' % self.filename)
      os.system('gzip %s -f' % self.filename)


if __name__ == '__main__':
  with MkFile(file_name='chip.def.gz', type='data', mode='r') as f:
    for line in f:
      print(line)
    #f.write('test')
