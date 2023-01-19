# Human-expert-knowledge-integrated-deep-learning-for-microscopy-image-analysis
This repository includes Python codes and datasets used in our npj Digital Medicine protocol paper "Human expert knowledge integrated deep learning for microscopy image analysis". A detailed description of how to use this repository is provided in our protocol, for review purposes, the preview version of this protocol is not available right now. These codes are based on the previous publications of our group, including three different tasks: classification (DCTL), detection (GFS-ExtremeNet), and reconstruction (COMI). All the codes are refactored for better readability. The corresponding datasets for these codes are available at https://www.scidb.cn/en/s/me22my.

## Dependencies
To run the codes, install the Anaconda(https://repo.anaconda.com/archive/Anaconda3-2022.10-Windows-x86_64.exe) and create the virtual environment using the following command, if you are using a GPU then use the command 1 to create a GPU-compatible virtual environment, otherwise, use command 2:
1. conda create -n microscopy_image_analysis cudatoolkit=10.0.130 cudnn=7.6.5 python=3.7.13
2. conda create -n microscopy_image_analysis python=3.7.13

To run the codes, the following dependencies are required:
+ [Python](https://www.python.org/downloads/) 3.7.13 or below
+ [tensorflow 1.15.0](https://www.tensorflow.org/install/) 
+ [pytorch 1.2.0](https://pytorch.org/get-started/previous-versions/#v120)
+ [torchvision 0.4.0](https://pytorch.org/get-started/previous-versions/#v120)
+ [keras 2.2.4](https://keras.io)
+ [scikit-learn 1.0.2](https://scikit-learn.org/stable/install.html)
+ [numpy 1.21.5](https://numpy.org/)
+ [pandas 0.25.0](https://pandas.pydata.org/pandas-docs/stable/install.html)
+ [matplotlib 3.5.3](https://matplotlib.org/)
