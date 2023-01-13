import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import logging
from tqdm import trange
from datetime import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from models.model import CycleGAN
from lib.reader_image import get_training_data, get_train_batch, get_test_batch, get_roc_batch

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 256, 'image size, default: 256')
tf.flags.DEFINE_integer('max_iter', 50, 'max_iter, default: 100000')
tf.flags.DEFINE_bool('use_lsgan', True, 
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance', 
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('lambda1', 10, 
                        'weight for forward cycle loss (X->Y->X), default: 10')
tf.flags.DEFINE_integer('lambda2', 10, 
                        'weight for backward cycle loss (Y->X->Y), default: 10')
tf.flags.DEFINE_float('learning_rate', 2e-4, 
                      'initial learning rate for CycleGAN, default: 0.0002')
tf.flags.DEFINE_float('feature_learning_rate', 2e-6, 
                      'initial learning rate for feature extractor, default: 0.000002')
tf.flags.DEFINE_float('beta1', 0.5, 
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50, 
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64, 
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('load_model', None, 
                       'folder of saved our_model that you wish to continue training (e.g. 20170602-1936), default: None')

def train():
    
    # init checkpoints dir
    if FLAGS.load_model is not None:
        checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
    else:
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        checkpoints_dir = "checkpoints/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass
        
    # init model
    graph = tf.Graph()
    with graph.as_default():
        cycle_gan = CycleGAN(
            batch_size = FLAGS.batch_size, 
            image_size = FLAGS.image_size, 
            use_lsgan = FLAGS.use_lsgan, 
            norm = FLAGS.norm, 
            lambda1 = FLAGS.lambda1, 
            lambda2 = FLAGS.lambda2, 
            learning_rate = FLAGS.learning_rate, 
            beta1 = FLAGS.beta1, 
            ngf = FLAGS.ngf
        )
        G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss, fake_x, fake_y, \
            x_pre, y_pre, fake_x_pre, fake_y_pre, x_correct, y_correct, fake_x_correct, fake_y_correct, \
                f_x, f_y, f_fakeX, f_fakeY, softmax_x, softmax_y, softmax_fakeX, softmax_fakeY = cycle_gan.model()
        optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss)
        summary_op = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.compat.v1.train.Saver()
        best_saver = tf.compat.v1.train.Saver(max_to_keep=1)
    
    # training and testing
    with tf.compat.v1.Session(graph = graph) as sess:
        # load model from checkpoint
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            iteration = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            sess.run(tf.compat.v1.global_variables_initializer())
            iteration = 0
            
        # training
        loss_indicator = float('inf')
        loss_values = {}
        # CycleGAN loss
        loss_values['G_loss_val'] = []
        loss_values['D_Y_loss_val'] = []
        loss_values['F_loss'] = []
        loss_values['D_X_loss'] = []
        # teacher loss (macroscopic classifcation)
        # student loss (microscpoic classifcation)
        # learning loss (combination)
        loss_values['teacher_loss_eval'] = []
        loss_values['student_loss_eval'] = []
        loss_values['learning_loss_eval'] = []
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        print('start trianing')
        try:
            for iteration in range(0, FLAGS.max_iter):
                # read image (because images don't paired one by one, therefore using the random reading can help to create more abundent image pair combination)
                x_image, x_label = get_train_batch("X", FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, "./dataset/")
                y_image, y_label = get_train_batch("Y", FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, "./dataset/")
                # train the model
                _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, teacher_loss_eval, student_loss_eval, learning_loss_eval, summary = (
                    sess.run(
                        fetches = [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss, summary_op], 
                        feed_dict = {cycle_gan.x: x_image, cycle_gan.y: y_image, 
                                     cycle_gan.x_label: x_label, cycle_gan.y_label: y_label}
                    )
                )
                # logging
                train_writer.add_summary(summary, iteration)
                train_writer.flush()
                
                # print outputs
                print('-----------Iteration %d:-------------' % (iteration))
                print('G_loss   : {}'.format(G_loss_val))
                print('D_Y_loss : {}'.format(D_Y_loss_val))
                print('F_loss   : {}'.format(F_loss_val))
                print('D_X_loss : {}'.format(D_X_loss_val))
                print('teacher_loss: {}'.format(teacher_loss_eval))
                print('student_loss: {}'.format(student_loss_eval))
                print('learning_loss: {}'.format(learning_loss_eval))
                # loss curve
                loss_values['G_loss_val'].append(G_loss_val)
                loss_values['D_Y_loss_val'].append(D_Y_loss_val)
                loss_values['F_loss'].append(F_loss)
                loss_values['D_X_loss'].append(D_X_loss)
                loss_values['teacher_loss_eval'].append(teacher_loss_eval)
                loss_values['student_loss_eval'].append(student_loss_eval)
                loss_values['learning_loss_eval'].append(learning_loss_eval)
                
                # save global best model
                if student_loss_eval < loss_indicator:
                    loss_indicator = D_Y_loss_val
                    f = open(checkpoints_dir + "/log.txt", 'w')
                    f.seek(0)
                    f.truncate()
                    f.write('The best model: iteration %d with minimal student_loss_eval %.6f\n' %(iteration, D_Y_loss_val))
                    f.close()
                    save_path = best_saver.save(sess, checkpoints_dir + "/bestmodel/our_model.ckpt", global_step=iteration)
                    print("Model saved in file: %s" % save_path)
            
            # save final model
            save_path = saver.save(sess, checkpoints_dir + "/our_model.ckpt", global_step=iteration)
            print("Last model saved in file: %s" % save_path)
            
            # testing (the goal is to identify parasites, so only the discriminator of Y is tested)
            print('Now is in testing! Please wait for result...')
            # read test images and labels
            test_images_y, test_labels_y = get_roc_batch(FLAGS.image_size, FLAGS.image_size, "./dataset/Y")
            print('Testing %d samples' %(len(test_images_y)))
            # testing
            y_correct_cout = 0
            fake_x_correct_cout = 0
            for i in trange(0, len(test_images_y)):
                y_correct_eval, fake_x_correct_eval = sess.run([y_correct, fake_x_correct], 
                                                               feed_dict = {cycle_gan.y: [test_images_y[i]], cycle_gan.y_label: [test_labels_y[i]]})
                # counting
                if y_correct_eval:
                    y_correct_cout += 1
                if fake_x_correct_eval:
                    fake_x_correct_cout += 1
            # metric
            y_accuracy = y_correct_cout / (len(test_labels_y))
            fake_x_accuracy = fake_x_correct_cout / (len(test_labels_y))
            
            # output result
            print('\nfake_x_correct_cout: %d' %(fake_x_correct_cout))
            print('y_accuracy: %.6f' %(y_accuracy))
            print('fake_x_accuracy: %.6f' %(fake_x_accuracy))
            
            # plot loss curve
            iteration_num = [i for i in range(FLAGS.max_iter)]
            plt.plot(iteration_num, loss_values['student_loss'], label='student_loss')
            plt.xlabel('Number of Iteration')
            plt.ylabel('Loss Value')
            plt.title('Loss_Curve')
            plt.legend()
            plt.show()
                
            coord.request_stop()
        
        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # save the model before the console is off
            # save_path = saver.save(sess, checkpoints_dir + "/our_model.ckpt", global_step = iteration)
            # print("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

def main(unused_argv):
    train()

if __name__  ==  '__main__':
    logging.basicConfig(level = logging.INFO)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_virtual_device_configuration(
    #     gpus[0],
    #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    tf.compat.v1.app.run()
