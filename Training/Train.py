'''
This is a python file that used for training GAN.
TODO: provide a parser access from terminal.
'''
## Import module
import sys, os
if os.getcwd().endswith("DCGAN"):
    root_dir = os.getcwd()
else:
    root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
import tensorflow as tf
from datetime import datetime
from pytz import timezone
from tensorflow.python import debug as tf_debug

from Training.train_base import Train_base
from Training.Saver import Saver
from Training.Summary import Summary
from utils import *

class Train(Train_base):
    def __init__(self, config, log_dir, save_dir, **kwargs):
        super(Train, self).__init__(config.LEARNING_RATE, config.BETA1)
        self.config = config
        self.save_dir = save_dir
        self.comments = kwargs.get('comments', '')
        if self.config.SUMMARY:
            self.summary = Summary(log_dir, config, \
                                       log_comments=kwargs.get('comments', ''))

    def train(self, Model, DataSet, SAMPLE_X = None, SAMPLE_Y = None):
        # Reset tf graph.
        tf.reset_default_graph()

        # Create input node
        if not self.config.Y_LABLE:
            image_batch, init_op, dataset = self._input_fn(DataSet)
        else:
            image_batch, label_batch, init_op, dataset = self._input_fn_w_label(DataSet)
            # image, label = self._input_fn_NP()  # using numpy array as feed dict

        # Build up the graph and loss
        with tf.device('/gpu:0'):
            # Create placeholder
            if self.config.Y_LABLE:
                y = tf.placeholder(tf.float32, [self.config.BATCH_SIZE, self.config.NUM_CLASSES], name='y') # label batch
                x = tf.placeholder(tf.float32, [self.config.BATCH_SIZE] + self.config.IMAGE_DIM,
                                   name='real_images')  # real image
            else:
                y = None
                x = tf.placeholder(tf.float32, [self.config.BATCH_SIZE] + [self.config.IMAGE_HEIGHT_O, self.config.IMAGE_WIDTH_O,
                                                                           self.config.CHANNEL], name='real_images')  # real image


            z = tf.placeholder(tf.float32, [self.config.BATCH_SIZE, self.config.Z_DIM]) # latent variable

            if self.config.LOSS == "MRGAN":
                # Build up the graph for mrGAN
                G, G_mr, D, D_logits, D_, D_logits_, fm, fm_, D_mr, D_mr_logits, _, model = self._build_train_graph(x, y, z, Model)
                # Create the loss:
                d_loss, g_loss, e_loss = self._loss(D, D_logits, D_, D_logits_, x, G, fm, fm_, model.discriminator, y, \
                                                    G_mr, D_mr_logits)
            else:
                # Build up the graph
                G, D, D_logits, D_, D_logits_, fm, fm_, model = self._build_train_graph(x, y, z, Model)
                # Create the loss:
                d_loss, g_loss = self._loss(D, D_logits, D_, D_logits_, x, G, fm, fm_, model.discriminator, y)

            # Sample the generated image every epoch
            samples = model.sampler(z, y)

        # Create optimizer
        with tf.name_scope('Train'):
            t_vars = tf.trainable_variables()
            theta_G = [var for var in t_vars if 'g_' in var.name]
            theta_D = [var for var in t_vars if 'd_' in var.name]
            if self.config.LOSS == "MRGAN":
                theta_E = [var for var in t_vars if 'e_' in var.name]

            if self.config.LOSS in ["WGAN", "WGAN_GP", "FMGAN"]:
                optimizer = self._RMSProp_optimizer()
                d_optim_ = self._train_op(optimizer, d_loss, theta_D)
            elif self.config.LOSS in ["GAN", "LSGAN", "cGPGAN", "MRGAN"]:
                optimizer = self._Adam_optimizer()


            if self.config.LOSS == "WGAN":
                with tf.control_dependencies([d_optim_]):
                    d_optim = tf.group(*(tf.assign(var, \
                                                   tf.clip_by_value(var, -self.config.WEIGHT_CLIP, \
                                                                    self.config.WEIGHT_CLIP)) for var in theta_D))
            else:
                d_optim = self._train_op(optimizer, d_loss, theta_D)
                if self.config.LOSS == "MRGAN":
                    e_optim = self._train_op(optimizer, e_loss, theta_E)

            g_optim = self._train_op(optimizer, g_loss, theta_G)



        # Add summary
        if self.config.SUMMARY:
            summary_dict = {}
            if self.config.SUMMARY_SCALAR:
                scaler = {'generator_loss': g_loss,
                          'discriminator_loss': d_loss}
                if self.config.LOSS == "MRGAN":
                    scaler['encoder_loss'] = e_loss
                summary_dict['scalar'] = scaler

            merged_summary = self.summary.add_summary(summary_dict)

        # Add saver
        saver = Saver(self.save_dir)

        # Create Session
        sess_config = tf.ConfigProto(allow_soft_placement = True)
        # Use soft_placement to place those variables, which can be placed, on GPU
        with tf.Session(config = sess_config) as sess:
            # if self.config.DEBUG:
            #     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            if self.config.SUMMARY and self.config.SUMMARY_GRAPH:
                self.summary._graph_summary(sess.graph)

            if self.config.RESTORE:
                start_epoch = saver.restore(sess)
            else:
                saver.set_save_path(comments = self.comments)
                start_epoch = 0
                # initialize the variables
                init_var = tf.group(tf.global_variables_initializer(), \
                                    tf.local_variables_initializer())
                sess.run(init_var)
            # sample_z = np.random.uniform(-1, 1, size=(64, 100))
            sample_z = np.random.normal(size = (self.config.BATCH_SIZE, 100))
            if not self.config.Y_LABLE:
                sess.run(init_op)
                sample_x = sess.run(image_batch)
            else:
                # sample_x, sample_y = sess.run([image_batch, label_batch])
                sample_x, sample_y = SAMPLE_X, SAMPLE_Y
                # sample_x, sample_y = image[:64, ...], label[:64, ...] # for numpy input

            # Start Training
            tf.logging.info("Start traininig!")
            for epoch in range(start_epoch + 1, self.config.EPOCHS + start_epoch + 1):
                tf.logging.info("Training for epoch {}.".format(epoch))
                train_pr_bar = tf.contrib.keras.utils.Progbar(target= \
                                                                  int(self.config.TRAIN_SIZE / self.config.BATCH_SIZE))
                sess.run(init_op)
                for i in range(int(self.config.TRAIN_SIZE / self.config.BATCH_SIZE)):
                    batch_z = np.random.uniform(-1, 1, [self.config.BATCH_SIZE, 100]).astype(np.float32)
                    # Fetch a data batch
                    if not self.config.Y_LABLE:
                        image_batch_o = sess.run(image_batch)
                    else:
                        image_batch_o, label_batch_o = sess.run([image_batch, label_batch])


                        ## for numpy input
                        # image_batch_o, label_batch_o = image[i * self.config.BATCH_SIZE : (i + 1) * self.config.BATCH_SIZE], \
                        #                                label[i * self.config.BATCH_SIZE : (i + 1) * self.config.BATCH_SIZE]

                    if not self.config.Y_LABLE:
                        # Update discriminator
                        _, d_loss_o = sess.run([d_optim, d_loss],
                                               feed_dict={x: image_batch_o,
                                                          z: batch_z})
                        # Update generator
                        _ = sess.run([g_optim],
                                     feed_dict={x: image_batch_o,
                                                z: batch_z})
                        _, g_loss_o = sess.run([g_optim, g_loss],
                                               feed_dict={x: image_batch_o,
                                                          z: batch_z})
                        if self.config.LOSS == "MRGAN":
                            # Update encoder
                            _, e_loss_o = sess.run([e_optim, e_loss],
                                                   feed_dict = {x: image_batch_o,
                                                                z: batch_z})
                    else:
                        # Update discriminator
                        _, d_loss_o = sess.run([d_optim, d_loss],
                                    feed_dict = {x: image_batch_o,
                                                 y: label_batch_o,
                                                 z: batch_z})

                        # Update generator
                        _ = sess.run([g_optim],
                                     feed_dict = {x: image_batch_o,
                                                  y: label_batch_o,
                                                  z: batch_z})
                        _, g_loss_o = sess.run([g_optim, g_loss],
                                     feed_dict = {x: image_batch_o,
                                                  y: label_batch_o,
                                                  z: batch_z})
                        if self.config.LOSS == "MRGAN":
                            # Update encoder
                            _, e_loss_o = sess.run([e_optim, e_loss],
                                                   feed_dict = {x: image_batch_o,
                                                                y: label_batch_o,
                                                                z: batch_z})


                    # Update progress bar
                    train_pr_bar.update(i)
                    if i % 100 == 0:
                        if self.config.DEBUG:
                            ## Sample image for every 100 update in debug mode
                            if not self.config.Y_LABLE:
                                samples_o, d_loss_o, g_loss_o, summary_o = sess.run(
                                    [samples, d_loss, g_loss, merged_summary],
                                    feed_dict={x: sample_x,
                                               z: sample_z})
                            else:
                                samples_o, d_loss_o, g_loss_o, summary_o = sess.run(
                                    [samples, d_loss, g_loss, merged_summary],
                                    feed_dict={x: sample_x,
                                               y: sample_y,
                                               z: sample_z})

                            save_images(samples_o, image_manifold_size(samples_o.shape[0]), \
                                        os.path.join(self.config.SAMPLE_DIR, 'train_{:02d}_{:02d}.png'.format(epoch, i)))


                # Save the model per SAVE_PER_EPOCH
                if epoch % self.config.SAVE_PER_EPOCH == 0:
                    save_name = str(epoch)
                    saver.save(sess, 'model_' + save_name.zfill(4) + '.ckpt')

                if self.config.LOSS == "MRGAN":
                    print("Epoch: [%2d/%2d], d_loss: %.8f, g_loss: %.8f, e_loss: %.8f" \
                          % (epoch, self.config.EPOCHS + start_epoch, d_loss_o, g_loss_o, e_loss_o))
                else:
                    print("Epoch: [%2d/%2d], d_loss: %.8f, g_loss: %.8f" \
                          % (epoch, self.config.EPOCHS + start_epoch, d_loss_o, g_loss_o))
                ## Sample image after every epoch
                if not self.config.Y_LABLE:
                    samples_o, d_loss_o, g_loss_o, summary_o = sess.run([samples, d_loss, g_loss, merged_summary],
                                                                        feed_dict={x: sample_x,
                                                                                   z: sample_z})
                else:
                    samples_o, d_loss_o, g_loss_o, summary_o = sess.run([samples, d_loss, g_loss, merged_summary],
                                                       feed_dict = {x: sample_x,
                                                                    y: sample_y,
                                                                    z: sample_z})

                if self.config.SUMMARY:
                    self.summary.summary_writer.add_summary(summary_o, epoch)

                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss_o, g_loss_o))
                save_images(samples_o, image_manifold_size(samples_o.shape[0]), \
                            os.path.join(self.config.SAMPLE_DIR, 'train_{:02d}.png'.format(epoch)))

            if self.config.SUMMARY:
                self.summary.summary_writer.flush()
                self.summary.summary_writer.close()

            # Save the model after all epochs
            save_name = str(epoch)
            saver.save(sess, 'model_' + save_name.zfill(4) + '.ckpt')
            return


    def _loss(self, D, D_logits, D_, D_logits_, real = None, fake = None, real_fm = None, fake_fm = None,
              discriminator = None, label = None, G_mr = None, D_mr_logits = None):
        if self.config.LOSS == "MRGAN":
            d_loss, g_loss, e_loss = self._loss_MRGAN(D, D_logits, D_, D_logits_, fake, G_mr, D_mr_logits)
            return d_loss, g_loss, e_loss
        elif self.config.LOSS == "GAN":
            d_loss, g_loss = self._loss_GAN(D, D_logits, D_, D_logits_)
        elif self.config.LOSS == "WGAN":
            d_loss, g_loss = self._loss_WGAN(D, D_logits, D_, D_logits_)
        elif self.config.LOSS == "WGAN_GP":
            d_loss, g_loss = self._loss_WGAN_GP(D, D_logits, D_, D_logits_, real, fake, discriminator, label)
        elif self.config.LOSS == "LSGAN":
            d_loss, g_loss = self._loss_LSGAN(D, D_logits, D_, D_logits_)
        elif self.config.LOSS == "cGPGAN":
            d_loss, g_loss = self._loss_cGPGAN(D, D_logits, D_, D_logits_, real)
        elif self.config.LOSS == "FMGAN":
            d_loss, g_loss = self._loss_FMGAN(D, D_logits, D_, D_logits_, real, fake,
                                              real_fm, fake_fm, discriminator, label)
        else:
            raise Exception("The GAN type you specified is not found!")
        return d_loss, g_loss

    def _build_train_graph(self, x, y, z, Model):
        """
        Build up the training graph
        :return:
        G: generated image batch
        D: probability (after sigmoid)
        D_logits: logits before sigmoid
        D_: probability for fake data
        D_logits_: logits before sigmoid for fake data
        """
        ## Create the model
        main_graph = Model(self.config)
        if self.config.LOSS == "MRGAN":
            G, G_mr, D, D_logits, D_, D_logits_, fm, fm_, D_mr, D_mr_logits, fm_mr = main_graph.forward_pass(z, x, y)
            return G, G_mr, D, D_logits, D_, D_logits_, fm, fm_, D_mr, D_mr_logits, fm_mr, main_graph
        else:
            G, D, D_logits, D_, D_logits_, fm, fm_ = main_graph.forward_pass(z, x, y)
            return G, D, D_logits, D_, D_logits_, fm, fm_, main_graph



    def _input_fn_w_label(self, DataSet):
        """
        Create the input node
        :return:
        """
        with tf.device('/cpu:0'):
            with tf.name_scope('Input_Data'):
                # Training dataset
                dataset = DataSet(self.config.DATA_DIR, self.config, use_augmentation = True)
                # Inputpipeline
                image_batch, label_batch, init_op = dataset.inputpipline_singleset()
        return image_batch, label_batch, init_op, dataset

    def _input_fn(self, DataSet):
        with tf.device('/cpu:0'):
            with tf.name_scope('Input_Data'):
                # Training dataset
                dataset = DataSet(self.config.DATA_DIR, self.config, use_augmentation = True)
                # Inputpipeline
                try:
                    image_batch, init_op = dataset.inputpipline_singleset()
                except:
                    image_batch, _, init_op = dataset.inputpipline_singleset()
        return image_batch, init_op, dataset

    def _input_fn_NP(self, DataSet):
        """
        Create the input node using numpy function
        :return:
        """
        dataset = DataSet(self.config.DATA_DIR, self.config)
        X, y = dataset.load_mnist()
        return X, y

def _main_train_celebA(FLAGS = None):
    from config import Config
    from Model.DCGAN import DCGAN as Model
    from Inputpipeline.celebADataset import celebADataSet as DataSet
    from time import strftime

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs

    class tempConfig(Config):
        NAME = "CELEBA_WGAN"
        BATCH_SIZE = 64
        RESTORE = False
        TRAIN_SIZE = 202599
        DATA_DIR = os.path.join(root_dir, "Dataset/celebA")
        EPOCHS = 25
        NUM_CLASSES = None

        ## Input image
        IMAGE_HEIGHT = 218
        IMAGE_WIDTH = 178
        CHANNEL = 3

        # Crop and resize
        CROP = True
        IMAGE_HEIGHT_O = 64
        IMAGE_WIDTH_O = 64
        Y_LABEL = False

        # Loss
        LOSS = "WGAN"
        WEIGHT_CLIP = 0.01

        #
        SAMPLE_DIR = os.path.join(root_dir, "Training/samples")

    tmp_config = tempConfig()

    if FLAGS:
        _customize_config(tmp_config, FLAGS)
    tmp_config.display()

    # Folder to save the trained weights
    save_dir = os.path.join(root_dir, "Training/Weights")
    # Folder to save the tensorboard info
    log_dir = os.path.join(root_dir, "Training/Log")
    # Comments log on the current run
    comments = "This training is for celebA using DCGAN."
    comments += tmp_config.config_str() + datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d_%H_%M_%S")
    # Create a training object
    training = Train(tmp_config, log_dir, save_dir, comments=comments)
    training.train(Model, DataSet)

def _main_train_mnist(FLAGS = None):
    from config import Config
    from Model.DCGAN import DCGAN as Model
    from Inputpipeline.mnistDataset import mnistDataSet as DataSet
    from time import strftime

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs

    class tempConfig(Config):
        NAME = "mnist_DCGAN"
        BATCH_SIZE = 64
        RESTORE = False
        TRAIN_SIZE = 70000
        DATA_DIR = os.path.join(root_dir, "Dataset/mnist")
        DATA_NAME = "mnist"
        EPOCHS = 8
        NUM_CLASSES = 10

        Y_LABLE = False

        ## Input image
        IMAGE_HEIGHT = 28
        IMAGE_WIDTH = 28
        CHANNEL = 1

        # Crop and resize
        CROP = False
        IMAGE_HEIGHT_O = 28
        IMAGE_WIDTH_O = 28

        #
        SAMPLE_DIR = os.path.join(root_dir, "Training/samples")

    tmp_config = tempConfig()
    if FLAGS:
        _customize_config(tmp_config, FLAGS)
    tmp_config.display()

    # Folder to save the trained weights
    save_dir = os.path.join(root_dir, "Training/Weights")
    # Folder to save the tensorboard info
    log_dir = os.path.join(root_dir, "Training/Log")
    # Comments log on the current run
    comments = "This training is for prostate using DCGAN."
    comments += tmp_config.config_str() + datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d_%H_%M_%S")

    # Load sample x and sample Y
    SAMPLE_X = np.load(os.path.join(root_dir, "Inputpipeline/mnist_sample_x.npy"))[:64, ...]
    SAMPLE_Y = np.load(os.path.join(root_dir, "Inputpipeline/mnist_sample_y.npy"))[:64, ...]

    # Create a training object
    training = Train(tmp_config, log_dir, save_dir, comments=comments)
    training.train(Model, DataSet, SAMPLE_X, SAMPLE_Y)

def _main_train_prostate(FLAGS = None):
    from config import Config
    from Model.DCGAN import DCGAN as Model
    from Inputpipeline.CedarsDataset import CedarsDataset as DataSet
    from time import strftime

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs

    class tempConfig(Config):
        NAME = "prostate_DCGAN"
        BATCH_SIZE = 64
        RESTORE = True
        TRAIN_SIZE = 62073
        DATA_DIR = os.path.join(root_dir, "Dataset/prostate")
        DATA_NAME = "prostate"
        EPOCHS = 50
        NUM_CLASSES = 2

        Y_LABLE = True

        ## Input image
        IMAGE_HEIGHT = 64
        IMAGE_WIDTH = 64
        CHANNEL = 3

        # Crop and resize
        CROP = False
        IMAGE_HEIGHT_O = 64
        IMAGE_WIDTH_O = 64

        #
        SAMPLE_DIR = os.path.join(root_dir, "Training/samples")

    tmp_config = tempConfig()

    if FLAGS:
        _customize_config(tmp_config, FLAGS)
    tmp_config.display()

    # Folder to save the trained weights
    save_dir = os.path.join(root_dir, "Training/Weights")
    # Folder to save the tensorboard info
    log_dir = os.path.join(root_dir, "Training/Log")
    # Comments log on the current run
    comments = "This training is for prostate using DCGAN."
    comments += tmp_config.config_str() + datetime.now(timezone('US/Eastern')).strftime("%Y-%m-%d_%H_%M_%S")

    # Load sample x and sample Y
    SAMPLE_X = np.load(os.path.join(root_dir, "Inputpipeline/prostate_sample_x.npy"))[:64, ...]
    SAMPLE_Y = np.load(os.path.join(root_dir, "Inputpipeline/prostate_sample_y.npy"))[:64, ...]


    # Create a training object
    training = Train(tmp_config, log_dir, save_dir, comments=comments)
    training.train(Model, DataSet, SAMPLE_X, SAMPLE_Y)

def _customize_config(tmp_config, FLAGS):
    tmp_config.NAME = FLAGS.name
    tmp_config.EPOCHS = FLAGS.epoch
    tmp_config.LEARNING_RATE = FLAGS.learning_rate
    tmp_config.BETA1 = FLAGS.beta1
    tmp_config.BATCH_SIZE = FLAGS.batch_size
    tmp_config.LOSS = FLAGS.GAN_type
    tmp_config.RESTORE = FLAGS.restore
    tmp_config.SAMPLE_DIR = os.path.join(os.path.dirname(tmp_config.SAMPLE_DIR), FLAGS.sample_dir)
    tmp_config.Y_LABLE = FLAGS.C_GAN
    tmp_config.LABEL_SMOOTH = FLAGS.label_smooth
    tmp_config.MINIBATCH_DIS = FLAGS.miniBatchDis
    tmp_config.DEBUG = FLAGS.debug

if __name__ == "__main__":
    # _main_train_prostate()
    # _main_train_mnist()
    _main_train_celebA()