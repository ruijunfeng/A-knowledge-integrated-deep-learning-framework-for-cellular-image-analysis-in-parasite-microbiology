# A knowledge-integrated deep learning framework for cellular image analysis in parasite microbiology: specialized for parasite classification, detection, and out-of-focus reconstruction

This repository includes code and datasets for a knowledge-integrated deep learning framework for cellular image analysis in parasite microbiology. A detailed description of how to run this repository is provided at this protocol paper: <https://doi.org/10.1016/j.xpro.2023.102452>.

The code is based on previous publications from our group and includes three different tasks: classification ([DCTL](https://github.com/senli2018/DCTL)), detection ([GFS-ExtremeNet](https://github.com/jiangdat/GFS-ExtremeNet)), and reconstruction ([COMI](https://github.com/jiangdat/COMI)).

All the code has been refactored and includes detailed comments for improved readability. We recommend using this repository instead of the original ones. The corresponding datasets are available at <https://www.scidb.cn/anonymous/bWUyMm15>.

## Dependencies

The following tutorial is tested on Windows platform, for Linux platform, you need to download a differernt Anaconda. To run the codes, install the [Anaconda](https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe) and create a virtual environment using the following command. If you are using a GPU, use the following command instead to create a GPU-compatible virtual environment:

```bash
conda create -n microscopy_image_analysis cudatoolkit=10.0.130 cudnn=7.6.5 python=3.7.16
```

Otherwise, use this command to create a regular virtual enviroment:

```bash
conda create -n microscopy_image_analysis python=3.7.16
```

After setting up the required virtual environment, the next step is to install the necessary package dependencies. Open the command prompt within the activated virtual environment and use the command below to install each package:

```bash
pip install packagename==version
```

For example, to install pandas version 1.3.5, you would use:

```bash
pip install pandas==1.3.5
```

For installing pytorch and torchvision, it is recommended to use the command provided by the official website. The following are the needed packages:

+ [python](https://www.python.org/downloads/) 3.7.16 or below
+ [tensorflow 1.15.0](https://www.tensorflow.org/install/)
+ [pytorch 1.2.0](https://pytorch.org/get-started/previous-versions/#v120)
+ [torchvision 0.4.0](https://pytorch.org/get-started/previous-versions/#v120)
+ [keras 2.2.4](https://keras.io)
+ [keras-contrib 2.0.8](https://github.com/keras-team/keras-contrib)
+ h5py 2.10.0
+ protobuf 3.19.0
+ [scikit-learn 1.0.2](https://scikit-learn.org/stable/install.html)
+ opencv-python 4.6.0.66
+ pycocotools 2.0.5
+ tqdm 4.64.1
+ [pandas 1.3.5](https://pandas.pydata.org/pandas-docs/stable/install.html)
+ [numpy 1.21.5](https://numpy.org/)
+ [matplotlib 3.5.3](https://matplotlib.org/)

We have also provided a .txt file for quick setup. If you don't have a GPU, simply run:

```bash
pip install -r packages_cpu.txt
```

If you have a compatible GPU, use the following instead:

```bash
pip install -r packages_gpu.txt
```

## File Structure

To run this repository, it is important to maintain an identical file structure for each project, as shown below:

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

The pretrained models used in COMI are available at: <https://drive.google.com/drive/folders/13R9fZ45IyPdJrq-ATHatPc_j_977qsT3?usp=sharing>.

## Code Running

If you are running this code on a Windows platform, you can open Anaconda Navigator and install the Spyder IDE (use Spyder 5.3.3 for best compatibility) within the virtual environment named ``microscopy_image_analysis``.

Switch to the appropriate file directory, open the ``train.py`` file, and click the Run button. If all settings are correct, the IPython console will continuously scroll with training information. The trained model weights will be saved in the ``models\`` folder.

To test the trained model, open ``test.py`` and configure the necessary settings.

If you are using Linux, you can use VSCode, switch to the virtual environment in the terminal, and perform the same operations.

## Contact Us

If you have any problems running this repository, do not hesitate to contact us at this email address <fengruijuan558@gmail.com>.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{feng2023knowledge,
 title = {A knowledge-integrated deep learning framework for cellular image analysis in parasite microbiology},
 journal = {STAR Protocols},
 volume = {4},
 number = {3},
 pages = {102452},
 year = {2023},
 issn = {2666-1667},
 doi = {https://doi.org/10.1016/j.xpro.2023.102452},
 url = {https://www.sciencedirect.com/science/article/pii/S2666166723004197},
 author = {Ruijun Feng and Sen Li and Yang Zhang},
 keywords = {Bioinformatics, Computer sciences, Microscopy},
 abstract = {Cellular image analysis is an important method for microbiologists to identify and study microbes. Here, we present a knowledge-integrated deep learning framework for cellular image analysis, using three tasks as examples: classification, detection, and reconstruction. Alongside thorough descriptions of different models and datasets, we describe steps for computing environment setup, knowledge representation, data pre-processing, and training and tuning. We then detail evaluation and visualization. For complete details on the use and execution of this protocol, please refer to Li et al. (2021),1 Jiang et al. (2020),2 and Zhang et al. (2022).3}
}
```
