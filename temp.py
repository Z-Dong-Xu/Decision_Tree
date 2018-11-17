#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time:2018/10/11.


import numpy as np
import pandas as pd

iris = pd.read_csv("E:\PyCharm code\Data\iris.data.txt", header=None)
X = iris.iloc[:, [2, 3]].values
y = iris.iloc[:, [4]].values

y = np.array(y)
y = y.flatten()
print(X.shape, y.shape)
print(y)