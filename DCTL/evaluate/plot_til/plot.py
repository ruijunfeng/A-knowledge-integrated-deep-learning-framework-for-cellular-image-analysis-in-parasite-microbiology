
from __future__ import print_function
from lib import utils
import numpy as np
import os

import cv2
import matplotlib.cm as cm
from PIL import Image
from sklearn import preprocessing



def plot_conv_output(conv_img, i,id_y_dir):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    #ori_dir = os.path.join(id_y_dir, 'original_map')

    #print(conv_img)
    if i in [0,1,2,10,11,12,13,14,15,23,24,25]:
        for j in range(np.shape(conv_img)[3]):
            conv_dir=os.path.join(id_y_dir,'layer_'+str(i))
            utils.prepare_dir(conv_dir)
            file_name=os.path.join(conv_dir,'_conv_'+str(j)+'.png')
            gray_img=np.array(conv_img[0,:,:,j])
            v_min=np.min(gray_img)
            v_max=np.max(gray_img)
            img=(gray_img-v_min)/(v_max-v_min)*255
            cv2.imwrite(file_name,img)



def draw_heatmap(occ_map_path, ori_img, save_dir):
    # occ_map_path = '/home/root123/data/FCGAN/FCGAN_CODE/result/occ_test/occlusion_map.txt'
    occ_map = np.loadtxt(occ_map_path, dtype=np.float64)

    min_max_scaler = preprocessing.MinMaxScaler()
    occ_map = min_max_scaler.fit_transform((occ_map))
    # print(occ_map)
    # plt.imshow(abs(occ_map), cmap=cm.hot_r)
    # plt.colorbar()
    # plt.show()

    occ_map_img = Image.fromarray(np.uint8(cm.hot_r(abs(occ_map)) * 255))
    occ_map_img = cv2.cvtColor(np.asarray(occ_map_img), cv2.COLOR_RGB2BGR)
    frame = ori_img
    overlay = frame.copy()
    alpha = 0.5
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.addWeighted(occ_map_img, alpha, frame, 1 - alpha, 0, frame)
    cv2.imwrite(os.path.join(save_dir), frame)
