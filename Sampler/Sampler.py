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

from Sampler.sampler_base import Sampler_base
from Training.Saver import Saver
from utils import *

class Sampler(Sampler_base):
    def __init__(self, config, save_dir, **kwargs):
        super(Sampler, self).__init__()
        self.config = config
        self.save_dir = save_dir

    def main_sampler(self, Model, DataSet, SAMPLE_X = None, SAMPLE_Y = None):
        # Reset tf graph.
        tf.reset_default_graph()

        # Create input node
        if not self.config.Y_LABEL:
            image_batch, init_op, dataset = self._input_fn(DataSet)
        else:
            image_batch, label_batch, init_op, dataset = self._input_fn_w_label(DataSet)

        # Build up the graph and loss
        with tf.device('/gpu:0'):
            # Create placeholder
            if self.config.Y_LABEL:
                y = tf.placeholder(tf.float32, [self.config.BATCH_SIZE, self.config.NUM_CLASSES], name='y') # label batch
            else:
                y = None

            z = tf.placeholder(tf.float32, [self.config.BATCH_SIZE, self.config.Z_DIM]) # latent variable

            # Sample the generated image
            model = Model(self.config)
            samples = model.sampler(z, y)

        # Add saver
        saver = Saver(self.save_dir)

        # Create Session
        sess_config = tf.ConfigProto(allow_soft_placement = True)
        # Use soft_placement to place those variables, which can be placed, on GPU
        with tf.Session(config = sess_config) as sess:
            assert self.config.RESTORE, "RESTORE must be true for the sampler mode!"

            _ = saver.restore(sess, self.config.RUN, self.config.RESTORE_EPOCH)

            # Start Sampling
            tf.logging.info("Start sampling!")
            for epoch in range(1, self.config.EPOCHS + 1):
                if self.config.Y_LABEL:
                   sample_y = SAMPLE_Y
                sample_z = np.random.normal(size=(self.config.BATCH_SIZE, self.config.Z_DIM))
                sample_pr_bar = tf.contrib.keras.utils.Progbar(target= self.config.EPOCHS + 1)
                if not self.config.Y_LABEL:
                    samples_o = sess.run(samples,
                                                   feed_dict={z: sample_z})
                else:
                    samples_o = sess.run(samples,
                                                       feed_dict = {y: sample_y,
                                                                    z: sample_z})

                # Update progress bar
                sample_pr_bar.update(epoch)
                save_images(samples_o, image_manifold_size(samples_o.shape[0]), \
                            os.path.join(self.config.SAMPLE_DIR, 'sample_{:02d}.png'.format(epoch)))
            return

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
                if self.config.LOSS == "PacGAN":
                    pass
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
                if self.config.LOSS == "PacGAN":
                    pass
        return image_batch, init_op, dataset

    def _input_fn_NP(self, DataSet):
        """
        Create the input node using numpy function
        :return:
        """
        dataset = DataSet(self.config.DATA_DIR, self.config)
        X, y = dataset.load_mnist()
        return X, y

def _main_sampler_celebA(FLAGS = None):
    from config import Config
    from Model.DCGAN import DCGAN as Model
    from Inputpipeline.celebADataset import celebADataSet as DataSet
    from time import strftime

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs

    class tempConfig(Config):
        NAME = "CELEBA_Sampler"
        BATCH_SIZE = 64
        RESTORE = True
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
        LOSS = "GAN"
        WEIGHT_CLIP = 0.01

        #
        SAMPLE_DIR = os.path.join(root_dir, "Sampler/samples")

    tmp_config = tempConfig()

    if FLAGS:
        _customize_config(tmp_config, FLAGS)
    tmp_config.display()


    # Folder to save the trained weights
    save_dir = os.path.join(root_dir, "Training/Weights")
    # Create a sampler object
    sampler = Sampler(tmp_config, save_dir)
    sampler.main_sampler(Model, DataSet)

def _main_sampler_mnist(FLAGS = None):
    from config import Config
    from Model.DCGAN import DCGAN as Model
    from Inputpipeline.mnistDataset import mnistDataSet as DataSet

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs

    class tempConfig(Config):
        NAME = "mnist_DCGAN"
        BATCH_SIZE = 64
        RESTORE = True
        RUN = "Run_2019-01-07_20_59_48"
        RESTORE_EPOCH = 50
        DATA_DIR = os.path.join(root_dir, "Dataset/mnist")
        DATA_NAME = "mnist"
        NUM_CLASSES = 10

        Y_LABEL = True

        ## Input image
        IMAGE_HEIGHT = 28
        IMAGE_WIDTH = 28
        CHANNEL = 1

        # Crop and resize
        CROP = False
        IMAGE_HEIGHT_O = 28
        IMAGE_WIDTH_O = 28

        EPOCHS = 8

        #
        SAMPLE_DIR = os.path.join(root_dir, "Sampler/samples")

    tmp_config = tempConfig()

    if FLAGS:
        _customize_config(tmp_config, FLAGS)
    tmp_config.display()

    # Folder to save the trained weights
    save_dir = os.path.join(root_dir, "Training/Weights")

    # Load sample x and sample Y
    SAMPLE_X = np.load(os.path.join(root_dir, "Inputpipeline/mnist_sample_x.npy"))[:64, ...]
    SAMPLE_Y = np.load(os.path.join(root_dir, "Inputpipeline/mnist_sample_y.npy"))[:64, ...]

    # Create a training object
    sampler = Sampler(tmp_config, save_dir)
    sampler.main_sampler(Model, DataSet, SAMPLE_X, SAMPLE_Y)

def _main_sampler_prostate(FLAGS = None):
    from config import Config
    from Model.DCGAN import DCGAN as Model
    from Inputpipeline.CedarsDataset import CedarsDataset as DataSet

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs

    class tempConfig(Config):
        NAME = "prostate_sampler"
        BATCH_SIZE = 64
        RESTORE = True
        TRAIN_SIZE = 62073
        DATA_DIR = os.path.join(root_dir, "Dataset/prostate")
        DATA_NAME = "prostate"
        EPOCHS = 50
        NUM_CLASSES = 2

        Y_LABEL = True

        ## Input image
        IMAGE_HEIGHT = 64
        IMAGE_WIDTH = 64
        CHANNEL = 3

        # Crop and resize
        CROP = False
        IMAGE_HEIGHT_O = 64
        IMAGE_WIDTH_O = 64

        #
        SAMPLE_DIR = os.path.join(root_dir, "Sampler/samples")

    tmp_config = tempConfig()

    if FLAGS:
        _customize_config(tmp_config, FLAGS)
    tmp_config.display()

    # Folder to save the trained weights
    save_dir = os.path.join(root_dir, "Training/Weights")

    # Load sample x and sample Y
    SAMPLE_X = np.load(os.path.join(root_dir, "Inputpipeline/prostate_sample_x.npy"))[:64, ...]
    SAMPLE_Y = np.load(os.path.join(root_dir, "Inputpipeline/prostate_sample_y.npy"))[:64, ...]


    # Create a Sampler object
    sampler = Sampler(tmp_config, save_dir)
    sampler.main_sampler(Model, DataSet, SAMPLE_X, SAMPLE_Y)

def _customize_config(tmp_config, FLAGS):
    tmp_config.NAME = FLAGS.name
    tmp_config.EPOCHS = FLAGS.epoch
    tmp_config.RESTORE_EPOCH = FLAGS.restore_epoch
    tmp_config.LEARNING_RATE = FLAGS.learning_rate
    tmp_config.BETA1 = FLAGS.beta1
    tmp_config.BATCH_SIZE = FLAGS.batch_size
    tmp_config.LOSS = FLAGS.GAN_type
    tmp_config.RESTORE = FLAGS.restore
    tmp_config.SAMPLE_DIR = os.path.join(os.path.dirname(tmp_config.SAMPLE_DIR), FLAGS.sample_dir)
    tmp_config.Y_LABEL = FLAGS.C_GAN
    tmp_config.LABEL_SMOOTH = FLAGS.label_smooth
    tmp_config.MINIBATCH_DIS = FLAGS.miniBatchDis
    tmp_config.DEBUG = FLAGS.debug
    tmp_config.RUN = FLAGS.run

if __name__ == "__main__":
    pass
