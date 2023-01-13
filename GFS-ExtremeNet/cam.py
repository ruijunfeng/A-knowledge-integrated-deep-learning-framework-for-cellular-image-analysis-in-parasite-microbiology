# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 18:17:10 2022

@author: Android12138
"""

# Define your model
import os
import json
import torch
import importlib
import argparse
from nnet.py_factory import DummyModule, Network
from config import system_configs
from db.datasets import datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")
    parser.add_argument("--cfg_file", 
                        help="config file", 
                        default="GFS-ExtremeNet",
                        type=str)
    parser.add_argument("--weights_dir", 
                        help="test the model saved at weights_dir",
                        default="checkpoints/GFS-ExtremeNet_0.pkl", 
                        type=str)
    parser.add_argument("--split", 
                        help="which split to use (training, validation, and testing)",
                        default="validation",
                        type=str)
    parser.add_argument("--debug",
                        help="visualization of heatmaps",
                        default=False,
                        action="store_true")
    parser.add_argument("--visualize",
                        help="visualize the final detections and write into the 'result' directory",
                        default=True,
                        action="store_true")
    args = parser.parse_args()
    return args
# init model
args = parse_args()

cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
print("read cfg_file: {}".format(cfg_file))
with open(cfg_file, "r") as f:
    configs = json.load(f)

configs["system"]["snapshot_name"] = args.cfg_file
system_configs.update_config(configs["system"])

train_split = system_configs.train_split
val_split   = system_configs.val_split
test_split  = system_configs.test_split
split = {"training": train_split,
         "validation": val_split,
         "testing": test_split}[args.split]

dataset = system_configs.dataset
testing_db = datasets[dataset](configs["db"], split)
module_file = "models.{}".format(system_configs.snapshot_name)
nnet_module = importlib.import_module(module_file)
params = torch.load("checkpoints/20221127-165855/GFS-ExtremeNet_best.pkl")
model = DummyModule(nnet_module.model(testing_db))
model.load_state_dict(params)
model = model.eval()

from torchcam.cams.cam import CAM
cam = CAM(model, 'cnvs', '1')
import cv2
import numpy as np
from utils import normalize_
from torchcam.cams.gradcam import SmoothGradCAMpp

# pre-process input image
image = cv2.imread("dataset/Babesia/images/train2017/Babesiidae1.png")
resized_image = cv2.resize(image, (511, 511))
resized_image =  resized_image.astype(np.float32) / 255.0
normalize_(resized_image, testing_db.mean, testing_db.std)
resized_image = resized_image.transpose((2, 0, 1))
input_tensor = torch.from_numpy(resized_image)
input_tensor = input_tensor.unsqueeze(0)

with SmoothGradCAMpp(model) as cam_extractor:
  # Preprocess your data and feed it to the model
  out = model(input_tensor, debug=False)
  # Retrieve the CAM by passing the class index and the model output
  activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
  
