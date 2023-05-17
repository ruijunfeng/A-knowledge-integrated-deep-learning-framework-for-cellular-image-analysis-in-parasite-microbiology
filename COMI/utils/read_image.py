# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:44:39 2022

@author: Android12138
"""
import os
import cv2
import numpy as np

def load_confocal(input_shape=None, cell_type=None, z_depth=None, ratio=0.8):
    dir = 'dataset/BPAEC/' + cell_type

    # training dataset 80%
    # init list
    hrhq_train = []
    lrhq_train = []
    hrlq_train = []
    lrlq_train = []
    # reading images
    for _, _, files in os.walk(dir+'/'+z_depth):
        for file in files:
            if int(file.split('_')[-1].split('.')[0]) < len(files) * ratio:
                # training set reading
                img_lq = cv2.imread(dir+'/'+z_depth + '/' + file)
                img = cv2.resize(img_lq, (input_shape[0], input_shape[1]))
                lrlq_train.append(img)
                img = cv2.resize(img_lq, (input_shape[0]*4, input_shape[1]*4))
                hrlq_train.append(img)
                # read the corresponding ground truth
                file = 'Z7_' + file.split('_')[1]
                img_hq = cv2.imread(dir+'/Z007' + '/' + file)
                img = cv2.resize(img_hq, (input_shape[0]*4, input_shape[1]*4))
                hrhq_train.append(img)
                img = cv2.resize(img_hq, (input_shape[0], input_shape[1]))
                lrhq_train.append(img)
    
    # testing dataset 20%
    # init list
    lrlq_test = []
    hrlq_test = []
    lrhq_test = []
    hrhq_test = []
    # reading images
    for _, _, files in os.walk(dir+'/'+z_depth):
        for file in files:
            if int(file.split('_')[-1].split('.')[0]) >= len(files) * ratio:
                # test set reading
                img_lq = cv2.imread(dir+'/'+z_depth + '/' + file)
                img = cv2.resize(img_lq, (input_shape[0], input_shape[1]))
                lrlq_test.append(img)
                img = cv2.resize(img_lq, (input_shape[0]*4, input_shape[1]*4))
                hrlq_test.append(img)
                # read the corresponding ground truth
                file = 'Z7_' + file.split('_')[1]
                img_hq = cv2.imread(dir+'/Z007' + '/' + file)
                img = cv2.resize(img_hq, (input_shape[0]*4, input_shape[1]*4))
                hrhq_test.append(img)
                img = cv2.resize(img_hq, (input_shape[0], input_shape[1]))
                lrhq_test.append(img)

    # normalization
    hrhq_train = np.array(hrhq_train)
    hrhq_train = hrhq_train.astype('float32') /127.5 - 1.
    hrhq_test = np.array(hrhq_test)
    hrhq_test = hrhq_test.astype('float32')  /127.5 - 1.

    lrhq_train = np.array(lrhq_train)
    lrhq_train = lrhq_train.astype('float32') /127.5 - 1.
    lrhq_test = np.array(lrhq_test)
    lrhq_test = lrhq_test.astype('float32') /127.5 - 1.

    hrlq_train = np.array(hrlq_train)
    hrlq_train = hrlq_train.astype('float32') /127.5 - 1.
    hrlq_test = np.array(hrlq_test)
    hrlq_test = hrlq_test.astype('float32') /127.5 - 1.

    lrlq_train = np.array(lrlq_train)
    lrlq_train = lrlq_train.astype('float32') /127.5 - 1.
    lrlq_test = np.array(lrlq_test)
    lrlq_test = lrlq_test.astype('float32') /127.5 - 1.

    print(hrhq_train.shape)
    print(hrhq_test.shape)
    
    return hrhq_train, hrhq_test, lrhq_train, lrhq_test, hrlq_train, hrlq_test, lrlq_train, lrlq_test