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

from inference_base import Inference_base
from Training.Saver import Saver
from utils import *

class Inference(Inference_base):
    def __init__(self, config, save_dir, **kwargs):
        super(Inference, self).__init__()
        self.config = config
        self.save_dir = save_dir

    def test(self, Model, DataSet, filename_list):
        # Reset tf graph.
        tf.reset_default_graph()
        # Create input node
        image_batch, label_batch, init_op, = self._input_fn_w_label(DataSet, filename_list)


        # Build up the graph and loss
        with tf.device('/gpu:0'):
            # Sample the generated image
            model = Model(self.config)
            logits = model.forward_pass(image_batch)
            accuracy, update_op, reset_op = self._metric(logits, label_batch)

        # Add saver
        saver = Saver(self.save_dir)
        # Create Session
        sess_config = tf.ConfigProto(allow_soft_placement = True)
        # Use soft_placement to place those variables, which can be placed, on GPU
        with tf.Session(config = sess_config) as sess:
            assert self.config.RESTORE, "RESTORE must be true for the sampler mode!"

            _ = saver.restore(sess, self.config.RUN, self.config.RESTORE_EPOCH)

            # Start Sampling
            tf.logging.info("Start inference!")
            sess.run([init_op, reset_op])
            while True:
                try:
                    accuracy_o, _ = sess.run([accuracy] + update_op)
                except (tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError):
                    break
        print("Validation accuracy: %.8f" % (accuracy_o))
        return

    def _input_fn_w_label(self, DataSet, filename_list):
        """
        Create the input node
        :return:
        """
        with tf.device('/cpu:0'):
            with tf.name_scope('Input_Data'):
                # Training dataset
                dataset = DataSet(self.config.DATA_DIR, self.config, use_augmentation = True)
                # Inputpipeline
                image_batch, label_batch, init_op = dataset.inputpipline_customized(filename_list)
        return image_batch, label_batch, init_op

    def _metric(self, logits, label_batch):
        with tf.name_scope('Metric') as scope:
            lab = tf.argmax(label_batch, axis=-1)
            prediction = tf.argmax(logits, axis=-1)

            # Pridcition accuracy
            accuracy, update_op_a = self._accuracy_metric(lab, prediction)

            # Update op inside each validation run
            update_op = [update_op_a]
            # Reset op for each validation run
            variables = tf.contrib.framework.get_variables(
                scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            reset_op = tf.variables_initializer(variables)
        return accuracy, update_op, reset_op

def _main_inference_mnist(FLAGS = None):
    from config import Config
    from Model.Classifier import Classifier as Model
    from Inputpipeline.mnistDataset import mnistDataSet as DataSet

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs

    class tempConfig(Config):
        NAME = "mnist_Classifier_Inference"
        BATCH_SIZE = 64
        RESTORE = True
        RUN = "Run_2019-01-16_23_46_42"
        RESTORE_EPOCH = 12
        DATA_DIR = os.path.join(root_dir, "Dataset/mnist")
        DATA_NAME = "mnist"
        NUM_CLASSES = 10
        REPEAT = 1
        TRAINING = False

        ## Input image
        IMAGE_HEIGHT = 28
        IMAGE_WIDTH = 28
        CHANNEL = 1

        # Crop and resize
        CROP = False
        IMAGE_HEIGHT_O = 28
        IMAGE_WIDTH_O = 28

    tmp_config = tempConfig()

    if FLAGS:
        _customize_config(tmp_config, FLAGS)
    tmp_config.display()

    # Folder to save the trained weights
    save_dir = os.path.join(root_dir, "Training/Weights_mnist_classifier")

    # Create a inference object
    filename_list = [os.path.join("Tfrecord", "mnist_val.tfrecords")]
    infer = Inference(tmp_config, save_dir)
    infer.test(Model, DataSet, filename_list)

def _main_inference_prostate(FLAGS = None):
    from config import Config
    from Model.Classifier import Classifier as Model
    from Inputpipeline.CedarsDataset import CedarsDataset as DataSet

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs

    class tempConfig(Config):
        NAME = "prostate_Classifier_inference"
        BATCH_SIZE = 64
        RESTORE = True
        RUN = "Run_2019-01-16_23_53_33"
        DATA_DIR = os.path.join(root_dir, "Sampler/samples")
        DATA_NAME = "prostate"
        NUM_CLASSES = 2
        REPEAT = 1
        TRAINING = False

        ## Input image
        IMAGE_HEIGHT = 64
        IMAGE_WIDTH = 64
        CHANNEL = 3

        # Crop and resize
        CROP = False
        IMAGE_HEIGHT_O = 64
        IMAGE_WIDTH_O = 64

    tmp_config = tempConfig()

    if FLAGS:
        _customize_config(tmp_config, FLAGS)
    tmp_config.display()

    # Folder to save the trained weights
    save_dir = os.path.join(root_dir, "Training/Weights_prostate_classifier")

    # Create a Inference object
    filename_list = ["prostate_sampler.tfrecords"]
    infer = Inference(tmp_config, save_dir)
    infer.test(Model, DataSet, filename_list)

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
    _main_inference_prostate()
