'''
This is a python file that used for training GAN.
TODO: provide a parser access from terminal.
'''
## Import module
import sys, os
root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
import numpy as np
import tensorflow as tf
from time import strftime
from datetime import datetime
from pytz import timezone

from Input_Pipeline.mnistDataset import mnistDataSet as DataSet
from Training.train_base import Train_base
from Training.Saver import Saver
from Training.Summary import Summary
import Training.utils as utils
from Training.utils import initialize_uninitialized_vars

class Train(Train_base):
    def __init__(self, config, log_dir, save_dir, **kwargs):
        super(Train, self).__init__(config.LEARNING_RATE, config.BETA1)
        self.config = config
        self.save_dir = save_dir
        self.comments = kwargs.get('comments', '')
        if self.config.SUMMARY:
            if self.config.SUMMARY_TRAIN_VAL:
                self.summary_train = Summary(log_dir, config, log_type='train', \
                                             log_comments=kwargs.get('comments', ''))
                self.summary_val = Summary(log_dir, config, log_type='val', \
                                           log_comments=kwargs.get('comments', ''))
            else:
                self.summary = Summary(log_dir, config, \
                                       log_comments=kwargs.get('comments', ''))

    def train(self, model):
        # Reset tf graph.
        tf.reset_default_graph()

        # Create input node
        init_op_train, init_op_val, real_lab_input, \
        real_lab, real_unl_input, dataset_train = self._input_fn_train_val()

        # Build up the graph
        with tf.device('/gpu:0'):
            d_loss, g_loss, accuracy, roc_auc, pr_auc, \
            update_op, reset_op, preds, probs, main_graph, scalar_train_sum_dict \
                = training._build_train_graph(real_lab_input, \
                                              real_unl_input, real_lab, model)
