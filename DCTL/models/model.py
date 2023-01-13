import tensorflow as tf
from lib import ops
from models.discriminator import Discriminator
from models.classifier import Classifier
from models.generator import Generator
from models.resnet_classifier import resnetClassifier

REAL_LABEL=0.9

class CycleGAN:
    def __init__(self,
                 batch_size=1,
                 image_size=256,
                 use_lsgan=True,
                 norm='instance',
                 lambda1=10,
                 lambda2=10,
                 learning_rate=2e-4,
                 feature_learning_rate=2e-6,
                 beta1=0.5,
                 ngf=64
                 ):
        """
        Args:
          batch_size: integer, batch size
          image_size: integer, image size
          lambda1: integer, weight for forward cycle loss (X->Y->X)
          lambda2: integer, weight for backward cycle loss (Y->X->Y)
          use_lsgan: boolean
          norm: 'instance' or 'batch'
          learning_rate: float, initial learning rate for Adam
          beta1: float, momentum term of Adam
          ngf: number of gen filters in first conv layer
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_lsgan = use_lsgan
        use_sigmoid = not use_lsgan
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.feature_learning_rate = feature_learning_rate
        self.beta1 = beta1

        self.is_training = tf.compat.v1.placeholder_with_default(True, shape=[], name='is_training')
        # cycle gan
        # micro gan
        self.G = Generator('G', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
        self.D_Y = Discriminator('D_Y', self.is_training, norm=norm, use_sigmoid=use_sigmoid)
        # macro gan
        self.F = Generator('F', self.is_training, norm=norm, image_size=image_size)
        self.D_X = Discriminator('D_X', self.is_training, norm=norm, use_sigmoid=use_sigmoid)
        # feature extracotors
        self.Cx = resnetClassifier("Cx", 1, self.is_training)
        self.Cy = Classifier("Cy", self.is_training, norm=norm, use_sigmoid=use_sigmoid)
        # input and ground truth for domain X and Y
        self.x = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3])
        self.y = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3])
        self.x_label = tf.compat.v1.placeholder(tf.int64, [None])  # denotes class label for DMKM
        self.y_label = tf.compat.v1.placeholder(tf.int64, [None])

    def model(self):
        x = self.x
        y = self.y
        cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y)

        # X -> Y
        fake_y = self.G(x)
        G_gan_loss = self.generator_loss(self.D_Y, fake_y, use_lsgan=self.use_lsgan)
        G_loss = G_gan_loss + cycle_loss
        D_Y_loss = self.discriminator_loss(self.D_Y, y, fake_y, use_lsgan=self.use_lsgan)

        # Y -> X
        fake_x = self.F(y)
        F_gan_loss = self.generator_loss(self.D_X, fake_x, use_lsgan=self.use_lsgan)
        F_loss = F_gan_loss + cycle_loss# - Disperse_loss
        D_X_loss = self.discriminator_loss(self.D_X, x, fake_x, use_lsgan=self.use_lsgan)

        # features (Y is micro and X is macro)
        f_x, softmax_x = self.Cx(x, '1')
        f_y, softmax_y = self.Cy(y, '1')

        f_fakeX, softmax_fakeX=self.Cx(fake_x, '2')
        f_fakeY, softmax_fakeY=self.Cy(fake_y, '2')

        fake_x_pre = tf.argmax(f_fakeX, 1)
        x_pre = tf.argmax(f_x, 1)
        fake_y_pre = tf.argmax(f_fakeY, 1)
        y_pre = tf.argmax(f_y, 1)

        fake_y_correct = tf.equal(fake_y_pre, self.x_label)
        fake_x_correct = tf.equal(fake_x_pre, self.y_label)

        y_correct = tf.equal(y_pre, self.y_label)
        x_correct = tf.equal(x_pre,self.x_label)

        # macro-guiding loss (classifying macroscopic objects)
        teacher_loss_x = self.teacher_loss(f_x, label=self.x_label)
        teacher_loss_fakeX = self.teacher_loss(f_fakeX, label=self.y_label)
        teacher_loss = teacher_loss_x + teacher_loss_fakeX

        # classifcation loss (classifying microscopic objects)
        student_loss_y = self.student_loss(f_y, label=self.y_label)
        student_loss_fakeY = self.student_loss(f_fakeY, label=self.x_label)
        student_loss = student_loss_y + student_loss_fakeY

        teach_loss = teacher_loss + student_loss

        # l2 distance (ensure the consistency in classifcation)
        ts_loss = self.learning_loss(f_x, f_fakeY)
        st_loss = self.learning_loss(f_y, f_fakeX)
        learning_loss = ts_loss + st_loss + teach_loss
        # CycleGAN, feature extractor
        return G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss, fake_x, fake_y, \
            x_pre, y_pre, fake_x_pre, fake_y_pre, x_correct, y_correct, fake_x_correct, fake_y_correct, \
                f_x, f_y, f_fakeX, f_fakeY, softmax_x, softmax_y, softmax_fakeX, softmax_fakeY

    def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss, teacher_loss, student_loss, learning_loss):
        
        def make_optimizer(loss, variables=tf.compat.v1.global_variables, name='Adam', starter_learning_rate=self.learning_rate):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            end_learning_rate = 0.0
            start_decay_step = 100000
            decay_steps = 100000
            beta1 = self.beta1
            learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                                      tf.compat.v1.train.polynomial_decay(starter_learning_rate, 
                                                                          global_step - start_decay_step,
                                                                          decay_steps,
                                                                          end_learning_rate,
                                                                          power=1.0),
                                      starter_learning_rate))

            tf.compat.v1.summary.scalar('learning_rate/{}'.format(name), learning_rate)
            learning_step = (tf.compat.v1.train.AdamOptimizer(learning_rate, beta1=beta1, name=name).minimize(loss, global_step=global_step, var_list=variables))
            return learning_step

        G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G', starter_learning_rate=self.learning_rate)
        D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y', starter_learning_rate=self.learning_rate)
        F_optimizer = make_optimizer(F_loss, self.F.variables, name='Adam_F', starter_learning_rate=self.learning_rate)
        D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X', starter_learning_rate=self.learning_rate)
        # learning_optimizer = make_optimizer(learning_loss, name='Adam_learn', starter_learning_rate=self.learning_rate)
        teacher_optimizer = make_optimizer(teacher_loss, self.Cx.variables, name='Adam_teacher_loss', starter_learning_rate=self.feature_learning_rate)
        student_optimizer = make_optimizer(student_loss, self.Cy.variables,name='Adam_student_loss', starter_learning_rate=self.feature_learning_rate)
        learningCx_optimizer = make_optimizer(learning_loss, self.Cx.variables, name='Adam_Cx', starter_learning_rate=self.feature_learning_rate)
        learningCy_optimizer = make_optimizer(learning_loss, self.Cy.variables, name='Adam_Cy', starter_learning_rate=self.feature_learning_rate)

        with tf.control_dependencies([G_optimizer, D_Y_optimizer,
                                      F_optimizer, D_X_optimizer, 
                                      teacher_optimizer, student_optimizer, 
                                      learningCx_optimizer, learningCy_optimizer]):
            return tf.no_op(name='optimizers')

    def discriminator_loss(self, D, y, fake_y, use_lsgan=True):
        """ Note: default: D(y).shape == (batch_size,5,5,1),
                           fake_buffer_size=50, batch_size=1
        Args:
          G: generator object
          D: discriminator object
          y: 4D tensor (batch_size, image_size, image_size, 3)
        Returns:
          loss: scalar
        """
        if use_lsgan:
            # use mean squared error
            #error_real = tf.reduce_mean(tf.math.squared_difference(D(y), label))
            error_real = tf.reduce_mean(tf.math.squared_difference(D(y), REAL_LABEL))
            error_fake = tf.reduce_mean(tf.square(D(fake_y)))
        else:
            # use cross entropy
            error_real = -tf.reduce_mean(ops.safe_log(D(y)))
            error_fake = -tf.reduce_mean(ops.safe_log(1 - D(fake_y)))
        loss = (error_real + error_fake) / 2
        return loss

    def generator_loss(self, D, fake_y, use_lsgan=True):
        """
        fool discriminator into believing that G(x) is real
        """
        if use_lsgan:
            # use mean squared error
            loss = tf.reduce_mean(tf.math.squared_difference(D(fake_y), REAL_LABEL))
        else:
            # heuristic, non-saturating loss
            loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2
        return loss

    def cycle_consistency_loss(self, G, F, x, y):
        """ cycle consistency loss (L1 norm)
        """
        forward_loss = tf.reduce_mean(tf.abs(F(G(x)) - x))
        backward_loss = tf.reduce_mean(tf.abs(G(F(y)) - y))
        loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return loss

    def teacher_loss(self, result,label):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=label)
        loss = tf.reduce_mean(cross_entropy, name="loss")
        return loss
    
    def student_loss(self,result,label):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=label)
        loss = tf.reduce_mean(cross_entropy)
        return loss

    def learning_loss(self, teacher_out, student_out):
        loss = tf.reduce_mean(tf.math.squared_difference(teacher_out, student_out))
        return loss