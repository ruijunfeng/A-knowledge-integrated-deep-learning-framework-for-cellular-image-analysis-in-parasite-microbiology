# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 09:58:52 2022

@author: Android12138
"""
import numpy as np
import math

# geometric feature
def feature(ex):
    # init coordinates from extremepoints
    feature = []
    xl = ex[0]
    xr = ex[2]
    xt = ex[1]
    xb = ex[3]
    
    # edge calculation
    x1 = xl - xt
    x2 = xt - xr
    x3 = xr - xb
    x4 = xb - xl
    x5 = xr - xl
    x6 = xb - xt
    
    # hypotenuse calculation
    x1_ = math.hypot(x1[0], x1[1])
    x2_ = math.hypot(x2[0], x2[1])
    x3_ = math.hypot(x3[0], x3[1])
    x4_ = math.hypot(x4[0], x4[1])
    x5_ = math.hypot(x5[0], x5[1])
    x6_ = math.hypot(x6[0], x6[1])
    
    # adjacency matrix
    Y = np.array([[0  , x1_, x5_, x4_],
                  [x1_, 0  , x2_, x6_],
                  [x5_, x2_, 0,   x3_],
                  [x4_, x6_, x3_, 0]])
    
    # eigenvalue calculation
    feature, b = np.linalg.eig(Y)
    feature_val = np.sqrt(np.sum(np.square(feature[0]) + np.square(feature[1]) + np.square(feature[2]) + np.square(feature[3])))
    
    return feature_val