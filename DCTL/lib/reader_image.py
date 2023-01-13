import os
import cv2
import random
import numpy as np

def read_a_image_by_name(name, image_width, image_height, data_path):
    """
    :param name: 图片文件名
    :param image_width: 期望图片宽度
    :param image_height: 期望图片高度
    :param data_path:
    :return: 图片矩阵, 图片标签 /home/hit/codes/new_dataset/Y/0_malaria
    """
    image = cv2.imread(os.path.join(data_path, name))
    if image is None:
        print("Can't find image of %s\n" %(os.path.join(data_path, name)))
        exit()
    # resize
    image = cv2.resize(image, (image_width, image_height))
    oimage = image.copy()
    # normalization
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image) / 127.5 - 1.
    return image, oimage

def get_training_data(xy, image_width, image_height, data_path):
    # init directory
    data_path = data_path + xy + '/'
    images = []
    labels = []
    subFold = "train/"
    label_folds = os.listdir(data_path)
    # read image and label
    for label_fold in label_folds:
        prefix = data_path + label_fold + '/' + subFold
        x_files = os.listdir(prefix)
        for x_file in x_files:
            labels.append(label_fold)
            image = cv2.imread(prefix + x_file)
            # resize
            image = cv2.resize(image, (image_width, image_height))
            # normalization
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image) / 127.5 - 1.0
            images.append(image)
    return images, labels

def get_train_batch(xy, batch_size, image_width, image_height, data_path):
    data_path = data_path + xy + '/'
    images = []
    labels = []
    x_folds = []
    x_files = []
    subFold = "train/"
    label_folds = os.listdir(data_path)
    # prefix
    for i in range(len(label_folds)):
        x_folds.append(data_path + label_folds[i] + '/' + subFold)
    # image names
    for x_fold in x_folds:
        x_files.append(os.listdir(x_fold))
    # read image and label
    for index in range(0, batch_size):
        i = random.randint(0, len(x_folds)-1)
        labels.append(label_folds[i])
        img_file = random.choice(x_files[i])
        image, _ = read_a_image_by_name(img_file, image_width, image_height, x_folds[i])
        images.append(image)
    return images, labels

def get_test_batch(xy, batch_size, image_width, image_height, data_path):
    # init directory
    data_path = data_path + xy + '/'
    images = []
    oimages = []
    labels = []
    x_folds = []
    x_files = []
    subFold = "test/"
    label_folds = os.listdir(data_path)
    # prefix
    for i in range(len(label_folds)):
        x_folds.append(data_path + label_folds[i] + '/' + subFold)
    # image names
    for x_fold in x_folds:
        x_files.append(os.listdir(x_fold))
    # read image and label
    for index in range(0, batch_size):
        for i in range(0, len(x_folds)):
            labels.append(label_folds[i])
            img_file = random.choice(x_files[i])
            image, oimage = read_a_image_by_name(img_file, image_width, image_height, x_folds[i])
            images.append(image)
            oimages.append(oimage)
    return images, labels, oimages

def get_roc_batch(image_width, image_height, data_path):
    # init directory
    data_path = data_path + '/'
    images = []
    labels = []
    x_folds = []
    x_files = []
    subFold = "test/"
    label_folds = os.listdir(data_path)
    # prefix
    for i in range(0, len(label_folds)):
        x_folds.append(data_path + label_folds[i] + '/' + subFold)
    # image names
    for x_fold in x_folds:
        x_files.append(os.listdir(x_fold))
    # read every image and label
    for i in range(0, len(x_folds)):
        for j in range(0, len(x_files[i])):
            labels.append(label_folds[i])
            img_file = x_files[i][j]
            image, _ = read_a_image_by_name(img_file, image_width, image_height, x_folds[i])
            images.append(image)
    return images, labels

if __name__  ==  '__main__':
    x_image, x_label = get_training_data("X", 224, 225, "./dataset/")