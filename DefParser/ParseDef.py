# -*- coding:utf-8 -*-
"""
User：Ranki Wang
Date: 2021/9/3
"""

import re, glob
import gzip

debug_mode = True


class ParseDate(object):
  """
  **Parse Date**
  """
  def __init__(self, name):
    self.name = name
    self.reportLimit = 3

    regx_diearea = re.compile(r'DIEAREA (\(.*\)) ;')
    regx_tuple = re.compile(r'\( (\S+ \S+) \)')
    regx_track = re.compile(r'TRACKS (Y|X) (\S+) DO (\S+) STEP (\S+) LAYER (M5|M6|M7|M8|M9|M10|M12|M13)')
    regx_inst = re.compile(r'- (\S+) (\S+).*\( (.*) \) (\S+) ?;?')

  def defIn(self, defPath='./chip.def.gz', unit=2000):
    """
    **read in def**
    """
    print(self.defIn.__doc__)
    [print("%-10s ==  %s" % (x[0],x[1])) for x in locals().items() if not x[0] == 'self']

    regx_design = re.compile(r'DESIGN (\S+) ;')

    db = {}
    defFiles = glob.glob(defPath)
    for file in defFiles:
      if self.reportLimit > 0:
        print(file)
        self.reportLimit -= 1
      elif self.reportLimit == 0:
        print("........")
        self.reportLimit -= 1
      f = gzip.open(file, 'r')
      for line in f:
        mt_design = regx_design.search(line.decode())
        if mt_design:
          tile = mt_design.groups()[0]
          print('Got design name ' + tile)
          break
      f.close()


if __name__ == "__main__":
  tmp = ParseDate('ranki')
  tmp.defIn('D:\FCFP\DefParser\*.def.gz')
  print(tmp.__doc__)