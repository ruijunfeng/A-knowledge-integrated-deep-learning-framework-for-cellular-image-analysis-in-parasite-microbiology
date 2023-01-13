import tensorflow as tf
import tensorflow.contrib as cb

weight_decay = 0.0005


def convolution(tensor_in, filters, num, reuse=False, name=None):
    with tf.compat.v1.variable_scope("convolution", reuse=reuse):
        activations = tf.layers.conv2d(tensor_in, filters, [3, 3], activation=tf.nn.relu,
                                       kernel_regularizer=cb.layers.l2_regularizer(weight_decay), name=name)
        # num controls the name of activation, 1 for real microscopic image, 2 for fake microscopic image
        if num == '1':
            tf.compat.v1.add_to_collection('Y_VGGconv', activations)
        else:
            tf.compat.v1.add_to_collection('fakeY_VGGconv', activations)
        return activations


def pool(tensor_in, reuse=False, name=None):
    with tf.compat.v1.variable_scope("pool", reuse=reuse):
        return tf.layers.max_pooling2d(tensor_in, [2, 2], [2, 2], name=name)


class Classifier:
    def __init__(self, name, is_training, reuse=False, norm='instance', use_sigmoid=False):
        self.name = name
        self.is_training = is_training
        self.norm = norm
        self.reuse = reuse
        self.use_sigmoid = use_sigmoid

    def __call__(self, input, num):
        """
        Args:
          input: batch_size x image_size x image_size x 3
        Returns:
          output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
                  filled with 0.9 if real, 0.0 if fake
        """
        with tf.compat.v1.variable_scope(self.name):

            print('Building Classifier!')
            conv_1 = convolution(input, 64, num, reuse=self.reuse, name="conv_1")
            conv_2 = convolution(conv_1, 64, num, reuse=self.reuse, name="conv_2")
            pool_1 = pool(conv_2, reuse=self.reuse, name="pool_1")
            # print("pool_1", pool_1)

            conv_3 = convolution(pool_1, 128,num, reuse=self.reuse, name="conv_3")
            conv_4 = convolution(conv_3, 128,num, reuse=self.reuse, name="conv_4")
            pool_2 = pool(conv_4, reuse=self.reuse, name="pool_2")
            # print("pool_2", pool_2)

            conv_5 = convolution(pool_2, 256,num, reuse=self.reuse, name="conv_5")
            conv_6 = convolution(conv_5, 256,num, reuse=self.reuse, name="conv_6")
            conv_7 = convolution(conv_6, 256,num, reuse=self.reuse, name="conv_7")
            pool_3 = pool(conv_7, reuse=self.reuse, name="pool_3")
            # print("pool_3", pool_3)

            conv_8 = convolution(pool_3, 512,num, reuse=self.reuse, name="conv_8")
            conv_9 = convolution(conv_8, 512,num, reuse=self.reuse, name="conv_9")
            conv_10 = convolution(conv_9, 512,num, reuse=self.reuse, name="conv_10")
            pool_4 = pool(conv_10, reuse=self.reuse, name="pool_4")
            # print("pool_4", pool_4)

            conv_11 = convolution(pool_4, 512,num, reuse=self.reuse, name="conv_11")
            conv_12 = convolution(conv_11, 512,num, reuse=self.reuse, name="conv_12")
            conv_13 = convolution(conv_12, 512, num,reuse=self.reuse, name="conv_13")
            pool_5 = pool(conv_13, reuse=self.reuse, name="pool_5")
            # print("pool_5", pool_5)

            pool_shape = pool_5.get_shape().as_list()
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            reshape = tf.reshape(pool_5, [-1, nodes])
            # print("reshape", reshape)

            dense_1 = tf.layers.dense(reshape, 512, activation=tf.nn.relu, reuse=self.reuse,
                                      kernel_regularizer=cb.layers.l2_regularizer(weight_decay), name="dense_1")
            dense_2 = tf.layers.dense(dense_1, 256, activation=tf.nn.relu, reuse=self.reuse,
                                      kernel_regularizer=cb.layers.l2_regularizer(weight_decay), name="dense_2")
            dense_3 = tf.layers.dense(dense_2, 4, activation=tf.nn.relu, reuse=self.reuse,
                                      kernel_regularizer=cb.layers.l2_regularizer(weight_decay), name="dense_3")
            softmax = tf.nn.softmax(dense_3)
            # print("dense_3", dense_3)
            self.reuse = True
            self.variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return dense_3,softmax
