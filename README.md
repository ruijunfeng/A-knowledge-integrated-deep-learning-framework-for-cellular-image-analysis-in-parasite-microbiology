# A protocol for integrating human expert knowledge into deep learning models for microscopy image analysis
This repository includes Python codes and datasets used in our paper. A detailed description of how to use this repository is provided in our paper, for review purposes, the preview version of this paper is not available right now. These codes are based on the previous publications of our group, including three different tasks: classification ([DCTL](https://github.com/senli2018/DCTL)), detection ([GFS-ExtremeNet](https://github.com/jiangdat/GFS-ExtremeNet)), and reconstruction ([COMI](https://github.com/jiangdat/COMI)). All the codes have been refactored and abundant code comments for better readability, it is recommended to use this repository instead of the original one. The corresponding datasets for these codes are available at https://www.scidb.cn/anonymous/bWUyMm15.

## Dependencies
To run the codes, install the [Anaconda](https://repo.anaconda.com/archive/Anaconda3-2022.10-Windows-x86_64.exe) and create the virtual environment using the following command, if you are using a GPU then use the command 1 to create a GPU-compatible virtual environment, otherwise, use command 2:
1. conda create -n microscopy_image_analysis cudatoolkit=10.0.130 cudnn=7.6.5 python=3.7.13
2. conda create -n microscopy_image_analysis python=3.7.13

After installing the needed virtual environment, the next step is to set up the package dependencies. Simply open the command prompt under the previously installed virtual environment and use the "pip install packagename==version" to install the needed package. For example, if you want to install pandas with a version of 1.3.5, you should use this command: "pip install pandas==1.3.5". For pytorch and torchvision, it is recommended to use the command provided by the official website. The following are the needed packages:
+ [Python](https://www.python.org/downloads/) 3.7.13 or below
+ [tensorflow 1.15.0](https://www.tensorflow.org/install/) 
+ [pytorch 1.2.0](https://pytorch.org/get-started/previous-versions/#v120)
+ [torchvision 0.4.0](https://pytorch.org/get-started/previous-versions/#v120)
+ [keras 2.2.4](https://keras.io)
+ keras-contrib 2.0.8
+ h5py 2.10.0
+ [scikit-learn 1.0.2](https://scikit-learn.org/stable/install.html)
+ opencv-python 4.6.0.66
+ pycocotools 2.0.5
+ tqdm 4.64.1
+ [pandas 1.3.5](https://pandas.pydata.org/pandas-docs/stable/install.html)
+ [numpy 1.21.5](https://numpy.org/)
+ [matplotlib 3.5.3](https://matplotlib.org/)

## File Structure
To run this repository, it is important to keep an identical file structure for each template as follows, the pretrained models used in COMI are available at (https://drive.google.com/drive/folders/13R9fZ45IyPdJrq-ATHatPc_j_977qsT3?usp=sharing):
```
DCTL          GFS-ExtremeNet    COMI
├─checkpoints ├─checkpoints     ├─dataset
├─dataset     ├─config          │  └─BPAEC
│  ├─X        ├─dataset         │      ├─actin
│  └─Y        │  ├─Babesia      │      ├─mitochondria
├─evaluate    │  ├─Toxoplasma   │      └─nucleus
│  └─plot_til │  └─Trypanosoma  ├─pretrained
├─lib         ├─db              ├─results
├─models      ├─external        └─utils
└─results     ├─models          train.py
    ├─auc_roc │  └─py_utils     test.py
    └─tsne    ├─nnet
train.py      ├─results
test.py       ├─sample
              ├─testing
              ├─tools
              └─utils
              config.py
              train.py
              test.py
```

## Code Running
When running these codes, open the Spyder IDE (use Spyder 5.2.2 for best compatibility) under the installed virtual environment and switch the file directory to the needed one. 
Open the "train.py" file and click the run button. If all the settings are correct, the IPython console will keep scrolling up with the training information. The trained model weights will be saved under the "models" folder. 
To test the trained model, open the "test.py" and configure the settings.

## Contact Us
If you have any problems in running this repository, do not hesitate to contact us with this email ruijun.feng@anu.edu.au.
