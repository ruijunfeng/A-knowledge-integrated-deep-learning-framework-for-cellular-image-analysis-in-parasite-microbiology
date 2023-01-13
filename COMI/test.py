from __future__ import print_function, division
import warnings
warnings.simplefilter("ignore", UserWarning)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import numpy as np

from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from scipy.stats import pearsonr
from skimage.measure import compare_psnr, compare_ssim
from utils.read_image import load_confocal

def test(prefix, weights_dir, input_shape, cell_type=None, z_depth=None):
    # Load the dataset
    hrhq_train, hrhq_test, lrhq_train, lrhq_test,\
        hrlq_train, hrlq_test, lrlq_train, lrlq_test\
            = load_confocal(input_shape=input_shape,
                            cell_type=cell_type, 
                            z_depth=z_depth)
    
    # load the model
    lq2hq = load_model(weights_dir + '/' + 'generator_l2h.h5', custom_objects={'InstanceNormalization': InstanceNormalization})
    hq2lq = load_model(weights_dir + '/' + 'generator_h2l.h5', custom_objects={'InstanceNormalization': InstanceNormalization})
    
    # calculate metrics
    print('original hrlq')
    compute(hrhq_test, hrlq_test)

    gen_hrhq = lq2hq.predict(hrlq_test, batch_size=1)
    print('save_model generate hrhq: ')
    compute(hrhq_test, gen_hrhq)

    gen_hrlq = hq2lq.predict(hrhq_test, batch_size=1)
    print('save_model generate hrlq: ')
    compute(hrlq_test, gen_hrlq)

    reconstr_hrhq = lq2hq.predict(gen_hrlq, batch_size=1)
    print('save_model reconstr hrhq: ')
    compute(hrhq_test, reconstr_hrhq)
    
    # visualization
    visualize_dir = prefix + '_deblur_imgs'
    if not os.path.exists(visualize_dir):
        os.makedirs(visualize_dir)
    img_dir = 'dataset/BPAEC/' + cell_type + '/' + z_depth
    for _, _, files in os.walk(img_dir):
        for file in files:
            img = []
            img_lq = cv2.imread(img_dir + '/' + file)
            img_lq = cv2.resize(img_lq, (input_shape[0] * 4, input_shape[1] * 4))
            img.append(img_lq)
            img = np.array(img)
            img = img.astype('float32') / 127.5 - 1.
            img = lq2hq.predict(img, batch_size=1)
            cv2.imwrite(visualize_dir + '/' + file, (0.5 * img[0] + 0.5) * 255)

# metric computation
def compute(set1, set2):
    b, h, w, c = set1.shape
    psnr_value = 0
    ssim_value = 0
    pearson_value = 0
    for i in range(0, b):
        psnr_value += pearsonr(set1[0].flatten(), set2[0].flatten())[0]
        ssim_value += PSNR(0.5 * set1[i] + 0.5, 0.5 * set2[i] + 0.5) # rescale to 1
        pearson_value += SSIM(0.5 * set1[i] + 0.5, 0.5 * set2[i] + 0.5)
    print('PSNR : ' + str(psnr_value / b))
    print('SSIM : ' + str(ssim_value / b))
    print('Pearson : ' + str(pearson_value / b))

def PSNR(img1, img2):
    return compare_psnr(img1, img2)

def SSIM(img1, img2):
    return compare_ssim(img1, img2, data_range=1, multichannel=True)

if __name__ == '__main__':
    # acgan + mnist dataset
    save_num = 500
    input_shape = (128, 128, 3)
    model_type = 'deblursrgan4'
    cell_type = 'actin'
    z_depth = 'Z005'
    prefix = 'results/%s_%s_%s' %(model_type, cell_type, z_depth)
    weights_dir = 'results/deblursrgan4_actin_Z005_weights'
    batch_size = 1
    test(prefix=prefix,
         weights_dir=weights_dir,
         input_shape=input_shape,
         cell_type=cell_type,
         z_depth=z_depth)
