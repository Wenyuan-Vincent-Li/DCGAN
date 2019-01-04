import os
import numpy as np
import tensorflow as tf

from utils import pp
import Training.Train as train

flags = tf.app.flags
flags.DEFINE_string("name", "DCGAN", "Descriptive name of current run")
flags.DEFINE_integer("epoch", 50, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("GAN_type", "GAN", "The type of GAN [GAN, WGAN, WGAN_GP, LSGAN, cGPGAN]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [celebA, mnist, prostate]")
flags.DEFINE_boolean("restore", False, "Weather to restore the pre-trained weights")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("C_GAN", False, "True for using conditional GAN")
flags.DEFINE_string("GPU", "0", "Which GPU used to process the data")
FLAGS = flags.FLAGS


def main(_):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU

    if FLAGS.train:
        if FLAGS.dataset == "mnist":
            train._main_train_mnist(FLAGS)
        elif FLAGS.dataset == "celebA":
            train._main_train_celebA(FLAGS)
        elif FLAGS.dataset == "prostate":
            train._main_train_prostate(FLAGS)
        else:
            raise Exception("The dataset you specified is not found!")

if __name__ == '__main__':
    tf.app.run()
