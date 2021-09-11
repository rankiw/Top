# -*- coding:utf-8 -*-
"""
User：Ranki Wang
Date: 2021/8/26
"""

import _pickle as pickle

# obj = 123, "abcdef", ["ac", 123], {"key": "value", "key1": "value1"}
# print(obj)
#
# with open("a.txt", "wb") as f:
#   pickle.dump(obj, f)

obj = {'a': 123, 'b': 'ads', 'c': [[1, 2], [3, 4]]}
path = 'a.txt'
f = open(path, 'wb')
pickle.dump(obj, f)
# f.close()

# f1 = open(path, 'rb')
# data1 = pickle.load(f1)
# print(data1)
