# -*- coding:utf-8 -*-
"""
User：Ranki Wang
Date: 2021/9/15
"""
import matplotlib
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

# # 录入身高与体重数据
# height = ['170', '179', '159', '160', '180', '164', '168', '174', '160', '183']
# weight = ['57', '62', '47', '67', '59', '49', '54', '63', '66', '80']
#
# plt.scatter(height, weight, s=500, c='r', marker='.')  # 散点图生成
# plt.xlabel('height(cm)')                              # plt.xlabel 设置x轴标签
# plt.ylabel('weight(kg)')                              # plt.ylabel 设置y轴标签
# plt.title('Test')                                     # plt.title  设置图像标题
# plt.grid()
# xmajorLocator = MultipleLocator(120)
# xminorLocator = MultipleLocator(60)

#plt.show()  # 显示图片

# x = np.arange(0, 3 * np.pi, 0.1)
# y = np.sin(x+0.1)
# y_sin = np.sin(x)
# y_cos = np.cos(x)
#
# # Plot the points using matplotlib
# plt.plot(x, y)
# plt.plot(x, y_sin)
# plt.plot(x, y_cos)
# plt.xlabel('x axis label')
# plt.ylabel('y axis label')
# plt.title('Sine and Cosine')
# plt.legend(['Ori','Sine', 'Cosine'])
# plt.show()
# plt.subplot(2, 1, 1)

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# 我要绘制2行1列的子图，现在声明，我要绘制第1个子图啦~
plt.subplot(2, 2, 3)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# 现在声明我要绘制第2个子图啦~
plt.subplot(2, 2, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()