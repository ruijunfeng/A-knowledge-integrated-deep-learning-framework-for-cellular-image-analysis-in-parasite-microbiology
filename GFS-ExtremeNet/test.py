import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import json
import torch
import pprint
import argparse
import importlib

import matplotlib
matplotlib.use("Agg")

from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.benchmark = False

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

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def test(db, split, weights_dir, debug=False, visualize=True):
    # init result directory
    result_dir = system_configs.result_dir
    result_dir = os.path.join(result_dir, split)
    make_dirs([result_dir])
    
    # load parameters
    print("building neural network...")
    nnet = NetworkFactory(db)
    print("loading parameters at {}".format(weights_dir))
    nnet.load_params(weights_dir)
    
    # testing data reading
    test_file = "testing.{}".format(db.data) # import testing.coco_extreme
    testing = importlib.import_module(test_file).testing # import testing.coco_extreme.testing
    
    # model eval
    nnet.cuda()
    nnet.eval_mode()
    testing(db, nnet, result_dir, debug=debug, visualize=visualize)

if __name__ == "__main__":
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

    print("loading all datasets...")
    dataset = system_configs.dataset
    print("split: {}".format(split))
    testing_db = datasets[dataset](configs["db"], split)

    print("system config...")
    pprint.pprint(system_configs.full)

    print("db config...")
    pprint.pprint(testing_db.configs)
    
    args.weights_dir = "checkpoints/20221127-1705/GFS-ExtremeNet_best.pkl"
    args.split = "testing"
    test(db=testing_db, split=args.split, weights_dir=args.weights_dir, debug=args.debug, visualize=args.visualize)