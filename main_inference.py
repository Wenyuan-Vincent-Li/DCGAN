import os
import numpy as np
import tensorflow as tf
import Inference.Inference as infer

flags = tf.app.flags
flags.DEFINE_string("dataset", "prostate", "The name of dataset [mnist, prostate]")
flags.DEFINE_string("data_dir", "samples", "Directory name that saved the sampled tf record")
flags.DEFINE_string("GPU", "0", "Which GPU used to process the data")
FLAGS = flags.FLAGS


def main(_):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    if FLAGS.dataset == "mnist":
        infer._main_inference_mnist(FLAGS)
    elif FLAGS.dataset == "prostate":
        infer._main_inference_prostate(FLAGS)
    else:
        raise Exception("The dataset you specified is not found!")



if __name__ == '__main__':
    tf.app.run()