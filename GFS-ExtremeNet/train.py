import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import json
import torch
import numpy as np
import queue
import pprint
import random
import argparse
import importlib
import threading
import traceback

from tqdm import tqdm
from utils import stdout_to_tqdm
from db.datasets import datasets
from config import system_configs
from nnet.py_factory import NetworkFactory
from torch.multiprocessing import Process, Queue, Pool
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train CornerNet")
    parser.add_argument("--cfg_file", 
                        help="config file", 
                        default="GFS-ExtremeNet",
                        type=str)
    parser.add_argument("--start_iter", 
                        help="train at iteration i, correspond to the read pretraind model",
                        default=0, 
                        type=int)
    parser.add_argument("--threads", 
                        help="number of image reading threads",
                        default=4, 
                        type=int)
    parser.add_argument("--debug", 
                        default=False, 
                        help="whethter to visualize the ground truths during training",
                        action="store_true")

    args = parser.parse_args()
    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

def prefetch_data(db, queue, sample_data, system_configs, data_aug, debug=False):
    ind = 0 # the read image id, 0 means every iteration the images is read randomly from the datasets
    print("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            # read data and put them into queue
            data, ind = sample_data(db, ind, system_configs, data_aug=data_aug, debug=debug)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e
        
def init_parallel_jobs(dbs, queue, fn, system_configs, data_aug, debug=False):
    tasks = [Process(target=prefetch_data, args=(db, queue, fn, system_configs, data_aug, debug)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()
        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]
        pinned_data_queue.put(data)
        if sema.acquire(blocking=False):
            return
        
def train(training_dbs, validation_db, start_iter=0, debug=False):
    # init settings
    learning_rate     = system_configs.learning_rate
    max_iteration     = system_configs.max_iter
    pretrained_model  = system_configs.pretrain
    snapshot_interval = system_configs.snapshot_interval
    val_iter          = system_configs.val_iter
    display_interval  = system_configs.display_interval
    decay_rate        = system_configs.decay_rate
    stepsize          = system_configs.stepsize
    training_size     = len(training_dbs[0].db_inds)
    # loss curve
    loss_indicator   = float('inf')
    loss_values      = {}
    loss_values['avg_loss'] = []

    # init data queue
    training_queue   = Queue(system_configs.prefetch_size)
    validation_queue = Queue(5)
    pinned_training_queue = queue.Queue(system_configs.prefetch_size)
    pinned_validation_queue = queue.Queue(5)
    
    # import function
    data_file   = "sample.{}".format(training_dbs[0].data) # equal to sample.coco_extreme
    sample_data = importlib.import_module(data_file).sample_data # equal to sample.coco_extreme.sample_data
    
    # training data
    training_tasks = init_parallel_jobs(dbs=training_dbs, 
                                        queue=training_queue, 
                                        fn=sample_data, 
                                        data_aug=True,
                                        debug=debug,
                                        system_configs=system_configs)
    
    training_pin_semaphore   = threading.Semaphore()
    training_pin_semaphore.acquire()

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()
    
    # validation data
    if val_iter:
        validation_tasks = init_parallel_jobs(dbs=[validation_db], 
                                              queue=validation_queue,
                                              fn=sample_data,
                                              data_aug=False,
                                              debug=False,
                                              system_configs=system_configs)
    
        validation_pin_semaphore = threading.Semaphore()
        validation_pin_semaphore.acquire()
        
        validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
        validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
        validation_pin_thread.daemon = True
        validation_pin_thread.start()
    
    print("building model...")
    nnet = NetworkFactory(training_dbs[0])
    
    # start the model training at the specified iteration
    if start_iter and pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        # load the pretrained model
        print("loading from pretrained model")
        nnet.load_params(pretrained_model)
        # set learning rate of the pretrained model
        learning_rate /= (decay_rate ** (start_iter // stepsize))
        nnet.set_lr(learning_rate)
        print("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)

    print("training start...")
    nnet.cuda()
    nnet.train_mode()
    avg_loss = AverageMeter()
    
    with stdout_to_tqdm() as save_stdout:
        # training loop
        for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):
            # forward and backward
            training = pinned_training_queue.get(block=True)
            training_loss = nnet.train(**training)
            avg_loss.update(training_loss.item())
            
            # output training information
            if display_interval and iteration % display_interval == 0:
                print("Iteration:%d/%d" %(iteration + 1, max_iteration))
                print("Training loss: %.6f" %(training_loss.item()))
                print("Average training loss within %d iterations: %.6f" %(display_interval, avg_loss.avg))
                print('-'*60)
                loss_values['avg_loss'].append(avg_loss.avg)
                # update the best model
                if loss_indicator > avg_loss.avg:
                    loss_indicator = avg_loss.avg
                    nnet.save_params("best")
                # reset avg_loss
                avg_loss.reset()
            del training_loss
            
            # validation
            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                nnet.eval_mode()
                validation = pinned_validation_queue.get(block=True)
                validation_loss = nnet.validate(**validation)
                print("validation loss at iteration {}: {}".format(iteration, validation_loss.item()))
                nnet.train_mode()
                
            # save the model
            if iteration % snapshot_interval == 0:
                nnet.save_params(iteration)
            
            # reduce the learning rate
            if iteration % stepsize == 0:
                learning_rate /= decay_rate
                nnet.set_lr(learning_rate)
    
    # plot loss curve
    iteration_num = [i for i in range(0, max_iteration, display_interval)]
    plt.plot(iteration_num, loss_values['avg_loss'], label='avg_loss')
    plt.xlabel('Number of Iteration')
    plt.ylabel('Loss Value')
    plt.title('Loss_Curve')
    plt.legend()
    plt.show()
    
    # sending signal to kill the thread
    training_pin_semaphore.release()

    # terminating data fetching processes
    for training_task in training_tasks:
        training_task.terminate()
    
    if val_iter:
        validation_pin_semaphore.release()
        for validation_task in validation_tasks:
            validation_task.terminate()

if __name__ == "__main__":
    args = parse_args()
    args.debug = False
    
    # read cfg file
    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    # update system_config
    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])
    if not os.path.exists(system_configs.snapshot_dir):
        os.makedirs(system_configs.snapshot_dir)
    # read datasets
    train_split = system_configs.train_split
    val_split   = system_configs.val_split
    print("loading all datasets...")
    dataset = system_configs.dataset
    # threads = max(torch.cuda.device_count() * 2, 4)
    threads = args.threads
    print("using {} threads".format(threads))
    training_dbs  = [datasets[dataset](configs["db"], train_split) for _ in range(threads)]
    if system_configs.val_iter: # You may remove validation to save GPU resources (val_iter)
        validation_db = datasets[dataset](configs["db"], val_split) 
    else:
        validation_db = None
    
    print("system config...")
    pprint.pprint(system_configs.full)
    print("db config...")
    pprint.pprint(training_dbs[0].configs)
    print("len of db: {}".format(len(training_dbs[0].db_inds)))
    
    train(training_dbs, validation_db, args.start_iter, args.debug)