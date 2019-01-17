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

class Train_Classifier(Train_base):
    def __init__(self, config, log_dir, save_dir, **kwargs):
        super(Train_Classifier, self).__init__(config.LEARNING_RATE, config.BETA1)
        self.config = config
        self.save_dir = save_dir
        self.comments = kwargs.get('comments', '')
        if self.config.SUMMARY:
            self.summary = Summary(log_dir, config, \
                                       log_comments=kwargs.get('comments', ''))

    def train(self, Model, DataSet):
        # Reset tf graph.
        tf.reset_default_graph()
        image_batch, label_batch, init_op_train, init_op_var = self._input_fn_w_label(DataSet)
        # Build up the graph and loss
        with tf.device('/gpu:0'):
            # Build up the graph
            logits = self._build_train_graph(image_batch, Model)
            # Create the loss:
            loss = self._loss(logits, label_batch)
            # Create metric:
            accuracy, update_op, reset_op = self._metric(logits, label_batch)

        # Create optimizer
        with tf.name_scope('Train'):
            optimizer = self._Adam_optimizer()
            optim = self._train_op(optimizer, loss)

        # Add summary
        if self.config.SUMMARY:
            summary_dict = {}
            if self.config.SUMMARY_SCALAR:
                scaler = {'loss': loss,
                          'accuracy': accuracy}
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

            # Start Training
            tf.logging.info("Start traininig!")
            for epoch in range(start_epoch + 1, self.config.EPOCHS + start_epoch + 1):
                tf.logging.info("Training for epoch {}.".format(epoch))
                train_pr_bar = tf.contrib.keras.utils.Progbar(target= \
                                                                  int(self.config.TRAIN_SIZE / self.config.BATCH_SIZE))
                sess.run(init_op_train)
                for i in range(int(self.config.TRAIN_SIZE / self.config.BATCH_SIZE)):
                    _, loss_o, accuracy_o, summary_o, _ = sess.run([optim, loss, accuracy, merged_summary] + update_op)
                    # Update progress bar
                    train_pr_bar.update(i)
                print("Epoch: [%2d/%2d], training loss: %.8f, training accuracy: %.8f" \
                      % (epoch, self.config.EPOCHS + start_epoch, loss_o, accuracy_o))

                # Do validation
                sess.run(init_op_var + [reset_op])
                for i in range(int(self.config.VAL_SIZE / self.config.BATCH_SIZE)):
                    try:
                        accuracy_o, loss_o, _ = sess.run([accuracy, loss] + update_op)
                    except (tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError):
                        break

                print("Epoch: [%2d/%2d], validation loss: %.8f, validation accuracy: %.8f" \
                      % (epoch, self.config.EPOCHS + start_epoch, loss_o, accuracy_o))

                # Save the model per SAVE_PER_EPOCH
                if epoch % self.config.SAVE_PER_EPOCH == 0:
                    save_name = str(epoch)
                    saver.save(sess, 'model_' + save_name.zfill(4) + '.ckpt')

                if self.config.SUMMARY:
                    self.summary.summary_writer.add_summary(summary_o, epoch)

            if self.config.SUMMARY:
                self.summary.summary_writer.flush()
                self.summary.summary_writer.close()

            # Save the model after all epochs
            save_name = str(epoch)
            saver.save(sess, 'model_' + save_name.zfill(4) + '.ckpt')
            return


    def _loss(self, logits, label_batch):
        loss = self._cross_entropy_loss_w_logits(label_batch, logits)
        return loss

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



    def _build_train_graph(self, x, Model):
        """
        Build up the training graph
        """
        ## Create the model
        main_graph = Model(self.config)
        logits = main_graph.forward_pass(x)
        return logits



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
                image_batch, label_batch, init_op_train, init_op_val = dataset.inputpipline_train_val()
        return image_batch, label_batch, init_op_train, init_op_val


def _main_train_mnist(FLAGS = None):
    from config import Config
    from Model.Classifier import Classifier as Model
    from Inputpipeline.mnistDataset import mnistDataSet as DataSet
    from time import strftime

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs

    class tempConfig(Config):
        NAME = "mnist_DCGAN"
        BATCH_SIZE = 64
        RESTORE = False
        TRAIN_SIZE = 60000
        VAL_SIZE = 10000
        DATA_DIR = os.path.join(root_dir, "Dataset/mnist")
        DATA_NAME = "mnist"
        EPOCHS = 50
        NUM_CLASSES = 10
        TRAINING = True


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
    # Folder to save the tensorboard info
    log_dir = os.path.join(root_dir, "Training/Log_mnist_classifer")
    # Comments log on the current run
    comments = "This training is for mnist classifier."
    comments += tmp_config.config_str() + datetime.now(timezone('US/Pacific')).strftime("%Y-%m-%d_%H_%M_%S")

    # Create a training object
    training = Train_Classifier(tmp_config, log_dir, save_dir, comments=comments)
    training.train(Model, DataSet)

def _main_train_prostate(FLAGS = None):
    from config import Config
    from Model.Classifier import Classifier as Model
    from Inputpipeline.CedarsDataset import CedarsDataset as DataSet
    from time import strftime

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs

    class tempConfig(Config):
        NAME = "prostate_Classifier"
        BATCH_SIZE = 64
        RESTORE = False
        TRAIN_SIZE = 49610
        VAL_SIZE = 12463
        DATA_DIR = os.path.join(root_dir, "Dataset/prostate")
        DATA_NAME = "prostate"
        EPOCHS = 50
        NUM_CLASSES = 2

        TRAINING = True
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
    # Folder to save the tensorboard info
    log_dir = os.path.join(root_dir, "Training/Log_prostate_classifier")
    # Comments log on the current run
    comments = "This training is for prostate classifier."
    comments += tmp_config.config_str() + datetime.now(timezone('US/Pacific')).strftime("%Y-%m-%d_%H_%M_%S")

    # Create a training object
    training = Train_Classifier(tmp_config, log_dir, save_dir, comments=comments)
    training.train(Model, DataSet)

def _customize_config(tmp_config, FLAGS):
    tmp_config.NAME = FLAGS.name
    tmp_config.EPOCHS = FLAGS.epoch
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

if __name__ == "__main__":
    # _main_train_mnist()
    _main_train_prostate()