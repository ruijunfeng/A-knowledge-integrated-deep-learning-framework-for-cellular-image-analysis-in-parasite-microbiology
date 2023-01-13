from __future__ import print_function, division
import warnings
warnings.simplefilter("ignore", UserWarning)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import tensorflow as tf
from keras.models import Model
from keras.layers.merge import _Merge
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Conv2D, BatchNormalization, UpSampling2D, Activation
from keras.layers import Reshape, Dropout, Concatenate, Lambda, Multiply, Add, Flatten, Dense
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.applications.vgg19 import VGG19

from skimage.measure import compare_psnr, compare_ssim
from utils.read_image import load_confocal

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
          
class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""

    def define_batch_size(self, bs):
        self.bs = bs

    def _merge_function(self, inputs):
        alpha = K.random_uniform((self.bs, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class StarGAN(object):
    def __init__(self, input_shape, learning_rate):
        # Model configuration
        self.input_shape = input_shape
        self.lr_height = input_shape[0]      # Low resolution height (128)
        self.lr_width = input_shape[1]       # Low resolution width (128)
        self.channels = input_shape[2]
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height * 4  # High resolution height
        self.hr_width = self.lr_width * 4    # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        self.n_residual_blocks = 9
        # Hyperparameter setting
        self.learning_rate = learning_rate
        optimizer = Adam(self.learning_rate, 0.5, 0.99)
        # Number of filters in the first layer of Generator and Discriminator
        self.gf = 64
        self.df = 64
        # Calculate output shape of Discriminator
        patch_hr_h = int(self.hr_height / 2 ** 4)
        patch_hr_w = int(self.hr_width / 2 ** 4)
        self.disc_patch_hr = (patch_hr_h, patch_hr_w, 1)
        # Number of GPU
        gpu_num = len(tf.config.experimental.list_physical_devices('GPU'))
        
        
        ''' Initiate models (Generator, Discriminator, and VGG19 image feature extractor)
        '''
        # Generator for out-of-focus to in-focus (lq2hq) and in-focus to out-of-focus (hq2lq) (cycle loss)
        self.generator_lq2hq = self.build_generator(name='gen_lq2hq')
        self.generator_hq2lq = self.build_generator(name='gen_hq2lq')        
        
        # Discriminator for in-focus (hq) image and out-of-focus (lq) image (GAN loss)
        self.discriminator_hq = self.build_discriminator(name='dis_hq')
        if gpu_num == 1:
            self.discriminator_hq_m = self.discriminator_hq 
        else:
            multi_gpu_model(self.discriminator_hq, gpus=gpu_num)
        self.discriminator_hq_m.compile(loss='mse',
                                        optimizer=optimizer,
                                        metrics=['accuracy'])
        self.discriminator_lq = self.build_discriminator(name='dis_lq')
        if gpu_num == 1:
            self.discriminator_lq_m = self.discriminator_lq 
        else:    
            multi_gpu_model(self.discriminator_lq, gpus=gpu_num)
        self.discriminator_lq_m.compile(loss='mse',
                                        optimizer=optimizer,
                                        metrics=['accuracy'])
        
        # Pre-trained VGG19 for image features extraction (content loss)
        self.vgg_hq = self.build_vgg_hr(name='vgg_hq')
        self.vgg_hq.trainable = False # froze the model weights
        self.vgg_hq_m = self.vgg_hq # multi_gpu_model(self.vgg_hq, gpus=1)
        self.vgg_hq_m.compile(loss='mse',
                              optimizer=optimizer,
                              metrics=['accuracy'])
        self.vgg_lq = self.build_vgg_hr(name='vgg_lq')
        self.vgg_lq.trainable = False
        self.vgg_lq_m = self.vgg_lq # multi_gpu_model(self.vgg_lq, gpus=3)
        self.vgg_lq_m.compile(loss='mse',
                              optimizer=optimizer,
                              metrics=['accuracy'])
        
        '''Simulate forward process and define the compared features using Model class
        '''
        # Init input for simulation
        img_lq = Input(shape=self.hr_shape)
        img_hq = Input(shape=self.hr_shape)
        
        img_lq_id = self.generator_hq2lq(img_lq)
        img_hq_id = self.generator_lq2hq(img_hq)
        
        # Cycle loss
        # Generate in-focus immage and out-of-focus image
        fake_hq = self.generator_lq2hq(img_lq)
        fake_lq = self.generator_hq2lq(img_hq)
        # Reconstruction to the input images
        reconstr_lq = self.generator_hq2lq(fake_hq)
        reconstr_hq = self.generator_lq2hq(fake_lq)
        
        # GAN loss
        # classify restored images
        validity_hq = self.discriminator_hq(fake_hq)
        validity_lq = self.discriminator_lq(fake_lq)
        # classify reconstruct images
        validity_reconstr_hq = self.discriminator_hq(reconstr_hq)
        validity_reconstr_lq = self.discriminator_lq(reconstr_lq)
        
        # Content loss
        # generate image features of restored images
        fake_hq_features = self.vgg_hq(fake_hq)
        fake_lq_features = self.vgg_lq(fake_lq)
        # generate image features of reconstruct images
        reconstr_hq_features = self.vgg_hq(reconstr_hq)
        reconstr_lq_features = self.vgg_lq(reconstr_lq)
        
        # Seal the inputs and outputs for loss calculation in the Model class
        self.combined_hq = Model([img_lq,
                                  img_hq], 
                                 [validity_hq, # GAN loss
                                  validity_reconstr_lq, 
                                  fake_hq_features, # Content loss
                                  reconstr_lq_features, 
                                  img_lq_id]) # Cycle loss
        if gpu_num == 1:
            self.combined_hq_m = self.combined_hq
        else:
            multi_gpu_model(self.combined_hq, gpus=gpu_num)
            
        self.combined_hq_m.compile(loss=['mse', 'mse', 'mse', 'mse', 'mse'],
                                   loss_weights=[1e-3, 1e-3, 1, 1, 1],
                                   optimizer=optimizer)
        self.combined_lq = Model([img_lq, 
                                  img_hq], 
                                 [validity_lq, 
                                  validity_reconstr_hq, 
                                  fake_lq_features, 
                                  reconstr_hq_features, 
                                  img_hq_id])
        if gpu_num == 1:
            self.combined_lq_m = self.combined_lq
        else:
            multi_gpu_model(self.combined_lq, gpus=gpu_num)
        self.combined_lq_m.compile(loss=['mse', 'mse', 'mse', 'mse', 'mse'],
                                   loss_weights=[1e-3, 1e-3, 1, 1, 1],
                                   optimizer=optimizer)
        
    def build_generator(self, name=None):
        # residual block
        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = InstanceNormalization()(d)
            d = Activation('relu')(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = InstanceNormalization()(d)
            d = Add()([d, layer_input])
            return d

        # High resolution image input
        img_hr = Input(shape=self.hr_shape)

        # Encoder
        c1 = Conv2D(64, kernel_size=7, strides=1, padding='same')(img_hr)
        c1 = InstanceNormalization()(c1)
        c1 = Activation('relu')(c1)

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            c1 = Conv2D(filters=64 * mult * 2, kernel_size=(3, 3), strides=2, padding='same')(c1)
            c1 = InstanceNormalization()(c1)
            c1 = Activation('relu')(c1)

        # Residual blocks
        r = residual_block(c1, self.gf * (n_downsampling ** 2))
        for _ in range(0, self.n_residual_blocks-1):
            r = residual_block(r, self.gf * (n_downsampling ** 2))

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            r = UpSampling2D()(r)
            r = Conv2D(filters=int(64 * mult / 2), kernel_size=(3, 3), padding='same')(r)
            r = InstanceNormalization()(r)
            r = Activation('relu')(r)

        # Decoder
        c2 = Conv2D(self.channels, kernel_size=7, strides=1, padding='same')(r)
        c2 = Activation('tanh')(c2)
        c2 = Add()([c2, img_hr])
        model = Model(img_hr, [c2], name=name)

        model.summary()
        return model

    def build_discriminator(self, name=None):
        n_layers, use_sigmoid = 3, False
        inputs = Input(shape=self.hr_shape)
        ndf=64
        
        # Input block
        x = Conv2D(filters=ndf, kernel_size=(4, 4), strides=2, padding='same')(inputs)
        x = LeakyReLU(0.2)(x)
        
        # Hidden block
        nf_mult, nf_mult_prev = 1, 1
        for n in range(n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            x = Conv2D(filters=ndf * nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        x = Conv2D(filters=ndf * nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        
        # Output block
        x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
        if use_sigmoid:
            x = Activation('sigmoid')(x)
        
        # Dense block
        x = Dense(1024, activation='tanh')(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=x, name=name)
        
        model.summary()
        return model

    def build_discriminator_lr(self):
        # Discriminator block
        def d_block(layer_input, filters, strides=1, bn=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            if bn:
                d = BatchNormalization()(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d
        
        d0 = Input(shape=self.lr_shape)
        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d4 = d_block(d2, self.df * 2, strides=2)
        d9 = Dense(self.df * 4)(d4)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)
        model = Model(d0, validity)
        model.summary()
        return Model(d0, validity)
    
    def build_vgg_hr(self, name=None):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        # redefine VGG19
        vgg = VGG19(include_top=False, weights="pretrained/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")
        vgg.outputs = [vgg.layers[9].output] # use the features of layer 9 as image features
        img = Input(shape=self.hr_shape)
        img_features = vgg(img)
        model = Model(img, img_features, name=name)
        model.summary()
        return model

    def PSNR(self, img1, img2):
        return compare_psnr(img1, img2, 1)

    def SSIM(self, img1, img2):
        return compare_ssim(img1, img2, data_range=1, multichannel=True)

    def save_imgs(self, img_dir, iteration, imgs_hrhq, imgs_hrlq):
        # number of images
        sample_num = imgs_hrhq.shape[0]
        
        # predict
        fake_hq = self.generator_lq2hq.predict(imgs_hrlq, batch_size=1)
        reconstr_lq = self.generator_hq2lq.predict(fake_hq, batch_size=1)
        fake_lq = self.generator_hq2lq.predict(imgs_hrhq, batch_size=1)
        reconstr_hq = self.generator_lq2hq.predict(fake_lq, batch_size=1)
        
        # rescale to 255
        imgs_lr = (0.5 * imgs_hrlq + 0.5) * 255
        fake_hr = (0.5 * fake_hq + 0.5) * 255
        fake_lr = (0.5 * fake_lq + 0.5) * 255
        imgs_hr = (0.5 * imgs_hrhq + 0.5) * 255
        reconstr_lr = (0.5 * reconstr_lq + 0.5) * 255
        reconstr_hr = (0.5 * reconstr_hq + 0.5) * 255
        
        # saving images
        for i in range(sample_num):
            cv2.imwrite("%s/%d_original_hrlq%d.png" %(img_dir, iteration, i), imgs_lr[i])
            cv2.imwrite("%s/%d_original_hrhq%d.png" %(img_dir, iteration, i), imgs_hr[i])
            cv2.imwrite("%s/%d_generated_hrhq%d.png" %(img_dir, iteration, i), fake_hr[i])
            cv2.imwrite("%s/%d_generated_hrlq%d.png" %(img_dir, iteration, i), fake_lr[i])
            cv2.imwrite("%s/%d_reconstr_hrhq%d.png" %(img_dir, iteration, i), reconstr_hr[i])
            cv2.imwrite("%s/%d_reconstr_hrlq%d.png" %(img_dir, iteration, i), reconstr_lr[i])
            
    def train(self, prefix, max_iter, batch_size, 
              snapshot_interval, display_interval, 
              cell_type=None, z_depth=None):
        # Init directories
        start_time = datetime.now()
        weigths_dir = prefix + '_weights'
        img_dir = prefix + '_imgs'
        log_dir = prefix + '_logs/'
        best_dir = "%s/bestmodel" %(weigths_dir)
        if not os.path.exists(weigths_dir):
            os.makedirs(weigths_dir)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(best_dir):
            os.makedirs(best_dir)
        # loss curve
        loss_indicator = float('inf')
        loss_values = {}
        loss_values['content_loss_hq'] = []
        
        # Load the datasets
        hrhq_train, hrhq_test, lrhq_train, lrhq_test,\
            hrlq_train, hrlq_test, lrlq_train, lrlq_test = load_confocal(
                input_shape=self.input_shape,
                cell_type=cell_type, 
                z_depth=z_depth)
                    
        # Training
        for iteration in range(max_iter):
            # randomly select batch_size images for training
            idx = np.random.randint(0, hrhq_train.shape[0], batch_size)
            imgs_hrhq_train, imgs_lrhq_train, imgs_hrlq_train, imgs_lrlq_train = \
                hrhq_train[idx], lrhq_train[idx], hrlq_train[idx], lrlq_train[idx]
                
            # reduce learning rate
            if iteration > int(max_iter/2):
                reduced_learning_rate = self.learning_rate * (max_iter - iteration) / (max_iter - int(max_iter/2))
                K.set_value(self.discriminator_hq_m.optimizer.lr, reduced_learning_rate)
                K.set_value(self.discriminator_lq_m.optimizer.lr, reduced_learning_rate)
                K.set_value(self.combined_hq_m.optimizer.lr, reduced_learning_rate)
                K.set_value(self.combined_lq_m.optimizer.lr, reduced_learning_rate)

            # Using high resolution in-focus and out-of-focus images to generate their corresponding high resolution outputs
            fake_hrhq = self.generator_lq2hq.predict(imgs_hrlq_train, batch_size=1)
            fake_hrlq = self.generator_hq2lq.predict(imgs_hrhq_train, batch_size=1)
            
            # Generate labels (concat two tuples)
            valid_hr = np.ones((batch_size,) + self.disc_patch_hr)
            fake_hr = np.zeros((batch_size,) + self.disc_patch_hr)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real_hr = self.discriminator_hq_m.train_on_batch(imgs_hrhq_train, valid_hr)
            d_loss_fake_hr = self.discriminator_hq_m.train_on_batch(fake_hrhq, fake_hr)
            d_loss_hr = 0.5 * np.add(d_loss_real_hr, d_loss_fake_hr)
            
            d_loss_real_lr = self.discriminator_lq_m.train_on_batch(imgs_hrlq_train, valid_hr)
            d_loss_fake_lr = self.discriminator_lq_m.train_on_batch(fake_hrlq, fake_hr)
            d_loss_lr = 0.5 * np.add(d_loss_real_lr, d_loss_fake_lr)

            # The generators want the discriminators to label the generated images as real
            # Extract ground truth image features using pre-trained VGG19 model
            image_features_hq = self.vgg_hq.predict(imgs_hrhq_train)
            image_features_lq = self.vgg_lq.predict(imgs_hrlq_train)

            # Training
            g_loss_hq = self.combined_hq_m.train_on_batch([imgs_hrlq_train, 
                                                           imgs_hrhq_train], 
                                                          [valid_hr, 
                                                           valid_hr, 
                                                           image_features_hq, 
                                                           image_features_lq,
                                                           imgs_hrlq_train])
            g_loss_lq = self.combined_lq_m.train_on_batch([imgs_hrlq_train, 
                                                           imgs_hrhq_train], 
                                                          [valid_hr, 
                                                           valid_hr, 
                                                           image_features_lq, 
                                                           image_features_hq,
                                                           imgs_hrhq_train])
            # logging loss values
            loss_values['content_loss_hq'].append(g_loss_hq[2])
            
            # Save the best model
            if g_loss_hq[2] < loss_indicator:
                loss_indicator = g_loss_hq[2]
                self.generator_lq2hq.save(best_dir + "/generator_l2h.h5")
                self.generator_hq2lq.save(best_dir + "/generator_h2l.h5")
                self.discriminator_lq.save(best_dir + "/discriminator_l.h5")
                self.discriminator_hq.save(best_dir + "/discriminator_h.h5")
            
            # Output training information
            if (iteration + 1) % display_interval == 0:
                elapsed_time = datetime.now() - start_time
                print('Iteration:%d/%d' %(iteration + 1, max_iter))
                print('Training time:%s' %(str(elapsed_time)))
                print('D_loss_hq:%.6f D_acc_hq:%.6f' %(d_loss_hr[0], d_loss_hr[1]))
                print('D_loss_lq:%.6f D_acc_lq:%.6f' %(d_loss_lr[0], d_loss_lr[1]))
                print('Content_loss_hq:%.6f' %(g_loss_hq[2]))
                print('Content_loss_lq:%.6f' %(g_loss_lq[2]))
                print('-'*60)

            # Save model and the deblured images
            if iteration % snapshot_interval == 0:
                sample_dir = "%s/iteration_%d" %(weigths_dir, iteration)
                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)
                self.generator_lq2hq.save(sample_dir + "/generator_l2h.h5")
                self.generator_hq2lq.save(sample_dir + "/generator_h2l.h5")
                self.discriminator_lq.save(sample_dir + "/discriminator_l.h5")
                self.discriminator_hq.save(sample_dir + "/discriminator_h.h5")
                self.save_imgs(img_dir, iteration, hrhq_test, hrlq_test)
            
        # plot loss curve
        iteration_num = [i for i in range(0, max_iter)]
        plt.plot(iteration_num, loss_values['content_loss_hq'], label='content_loss_hq')
        plt.xlabel('Number of Iteration')
        plt.ylabel('Loss Value')
        plt.title('Loss_Curve')
        plt.legend()
        plt.show()
            
        # testing
        # psnr = []
        # ssim = []
        # PSNR = 0
        # SSIM = 0
        # psnr_lr = []
        # ssim_lr = []
        # PSNR_lr = 0
        # SSIM_lr = 0
        # # predict
        # gen_hrhq = self.generator_lq2hq.predict(hrlq_test, batch_size=1)
        # gen_hrlq = self.generator_hq2lq.predict(gen_hrhq, batch_size=1)
    
        # for i in range(hrhq_test.shape[0]):
        #     p = self.PSNR(0.5 * hrhq_test[i] + 0.5, 0.5 * gen_hrhq[i] + 0.5)
        #     s = self.SSIM(0.5 * hrhq_test[i] + 0.5, 0.5 * gen_hrhq[i] + 0.5)
        #     PSNR += p
        #     SSIM += s
        #     psnr.append(p)
        #     ssim.append(s)
        # print('generated hrhq mean PSNR: ' + str(PSNR / hrhq_test.shape[0]))
        # print('generated hrhq mean SSIM: ' + str(SSIM / hrhq_test.shape[0]))

        # for i in range(hrhq_test.shape[0]):
        #     p_lr = self.PSNR(0.5 * hrlq_test[i] + 0.5, 0.5 * gen_hrlq[i] + 0.5)
        #     s_lr = self.SSIM(0.5 * hrlq_test[i] + 0.5, 0.5 * gen_hrlq[i] + 0.5)
        #     PSNR_lr += p_lr
        #     SSIM_lr += s_lr
        #     psnr_lr.append(p_lr)
        #     ssim_lr.append(s_lr)
        # print('reconstr hrlq mean PSNR: ' + str(PSNR_lr / hrhq_test.shape[0]))
        # print('reconstr hrlq mean SSIM: ' + str(SSIM_lr/ hrhq_test.shape[0]))

if __name__ == '__main__':
    # settings
    input_shape = (80, 80, 3)
    learning_rate = 1e-4
    batch_size = 1
    max_iter = 250
    snapshot_interval = 20
    display_interval = 10
    model_type = 'deblursrgan4'
    cell_type = 'actin'
    z_depth = 'Z005'
    prefix = 'checkpoints/%s/%s_%s_%s' %(datetime.now().strftime('%Y%m%d-%H%M%S'), model_type, cell_type, z_depth)
    # init model
    dcgan = StarGAN(input_shape=input_shape,
                    learning_rate=learning_rate)
    # train
    dcgan.train(prefix=prefix, 
                max_iter=max_iter, 
                batch_size=batch_size,
                snapshot_interval=snapshot_interval, 
                display_interval=display_interval,
                cell_type=cell_type,
                z_depth=z_depth)