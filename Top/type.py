# -*- coding:utf-8 -*-
"""
User：Ranki Wang
Date: 2021/9/13
"""


# 实例方法
def instancetest(self):
  print("this is a instance method")


# 类方法
def classtest(cls):
  print("this is a class method")


# 静态方法
def statictest(self):
  print("this is a static method")


# 创建类
test_property = {"name": "tom", "instancetest": instancetest, "classtest": classtest, "statictest": statictest}
Test = type("Test", (), test_property)

# 创建对象
test = Test()
# 调用方法
print(test.name)
test.instancetest()
test.classtest()
test.statictest()
