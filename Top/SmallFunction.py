# -*- coding:utf-8 -*-
"""
User：Ranki Wang
Date: 2021/9/10
"""


def printDict(my_dict):
  for dict_key in my_dict.keys():
    if isinstance(my_dict[dict_key], str):
      print(dict_key + ' : ' + my_dict[dict_key])
    elif isinstance(dict_key[dict_key], list):
      dict_value = dict_key[dict_key]
      dict_value_list = [str(i) for i in dict_value]
      dict_value_list2str = ''.join(dict_value_list)
      print(dict_key + ' : ' + dict_value_list2str)


if __name__ == "__main__":
  my_dict_name = {'a': ['a', 'b', 'c'], 'c': 'd', 'e': 'f'}
  printDict(my_dict_name)
