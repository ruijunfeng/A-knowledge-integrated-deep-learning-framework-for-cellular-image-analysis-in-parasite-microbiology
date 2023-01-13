import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import logging
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
from lib.reader_image import get_roc_batch

import tensorflow as tf
from models.model import CycleGAN

from evaluate.plot_til.tool import calucate, dense_to_one_hot, roc
from lib import utils

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('image_size', 256, 'image size, default: 256')
tf.flags.DEFINE_integer('classes', 4,
                        'number of classes, default: 4')
tf.flags.DEFINE_string('meta_dir', 'checkpoints/20221126-1321/our_model.ckpt-49.meta',
                       'directory of the saved .meta file of the saved model weights that you wish to testify (e.g. checkpoints/20221119-2211/bestmodel/our_model.ckpt-10.meta), default: None')
tf.flags.DEFINE_string('saved_weights_dir', 'checkpoints/20221126-1321',
                       'direcotry of the saved model weights folder that you wish to testify (e.g. checkpoints/20221119-2211/bestmodel), default: None')

def test():
    # init feed-dict and placeholder
    graph = tf.get_default_graph()
    with graph.as_default():
        cycle_gan = CycleGAN(image_size=FLAGS.image_size)
        G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss, fake_x, fake_y, \
            x_pre, y_pre, fake_x_pre, fake_y_pre, x_correct, y_correct, fake_x_correct, fake_y_correct, \
                f_x, f_y, f_fakeX, f_fakeY, softmax_x, softmax_y, softmax_fakeX, softmax_fakeY = cycle_gan.model()
        
    # init sess
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # load meta graph and restore weights
    saver = tf.compat.v1.train.import_meta_graph(FLAGS.meta_dir)
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.saved_weights_dir))
    sess.run(tf.compat.v1.local_variables_initializer())
        
    # init directory
    result_dir = './results'
    plot_dir = os.path.join(result_dir, 'tsne')
    roc_dir = os.path.join(result_dir, 'auc_roc')
    utils.prepare_dir(plot_dir)
    utils.prepare_dir(roc_dir)
    
    # read images
    test_images_y, test_labels_y = get_roc_batch(FLAGS.image_size, FLAGS.image_size, "./dataset/Y")
    prediction_argmax_results = []
    softmax_results = []
    y_correct_cout = 0
    sample_num = len(test_labels_y)
    print('number of sample %d' %(sample_num))
    features_d = []
    
    # testify
    for i in trange(0, sample_num):
        softmax_y_eval, y_pre_eval, f_y_eval, y_correct_eval = (sess.run(fetches = [softmax_y, y_pre, f_y, y_correct], 
                                                                         feed_dict = {cycle_gan.y: [test_images_y[i]], cycle_gan.y_label: [test_labels_y[i]]}))
        features_d.append(f_y_eval[0])
        if y_correct_eval:
            y_correct_cout += 1
        softmax_results.append(softmax_y_eval[0])
        prediction_argmax_results.append(y_pre_eval[0])
    sess.close()
    print('\ny_accuracy: {}'.format(y_correct_cout / sample_num))
    
    # calucate F1-score precision accuracy recall
    test_labels_y = np.array(test_labels_y, dtype=np.int32)
    calucate(test_labels_y, prediction_argmax_results)
    
    # roc curve and auc area
    # one_hot = dense_to_one_hot(np.array(prediction_argmax_results, dtype=np.int32), num_classes = FLAGS.classes)
    sum_label = dense_to_one_hot(np.array(test_labels_y, dtype=np.int32), num_classes = FLAGS.classes)
    roc(sum_label, softmax_results, FLAGS.classes, roc_dir)
    
    # t-SNE plot
    print('len features_d', len(features_d))
    plt.figure(figsize=(12, 6))
    tsne = TSNE(n_components=2, learning_rate=4).fit_transform(features_d)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=test_labels_y)
    plt.colorbar()
    plt.savefig(os.path.join(plot_dir, 't-SNE.png'))
    plt.show()
            
def main(unused_argv):
    test()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()