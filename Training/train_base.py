#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 16:34:59 2018
@author: wenyuan
"""
import tensorflow as tf

class Train_base(object):
    def __init__(self, learning_rate, beta1):
        self.learning_rate = learning_rate
        self.beta1 = beta1

    def _input_fn(self):
        raise NotImplementedError(
            'metirc() is implemented in Model sub classes')

    def _build_train_graph(self):
        raise NotImplementedError(
            'loss() is implemented in Model sub classes')

    def _loss(self, target, network_output):
        raise NotImplementedError(
            'loss() is implemented in Model sub classes')

    def _metric(self, labels, network_output):
        raise NotImplementedError(
            'metirc() is implemented in Model sub classes')

    def _train_op(self, optimizer, loss, var_list = None):
        train_op = optimizer.minimize(loss,
                                      global_step=tf.train.get_global_step(),
                                      var_list = var_list)
        return train_op

    def _train_op_w_grads(self, optimizer, loss, var_list = None):
        grads = optimizer.compute_gradients(loss, var_list = var_list)
        train_op = optimizer.apply_gradients(grads)
        return train_op, grads

    def _Adam_optimizer(self, name = 'Adam_optimizer'):
        optimizer = tf.train.AdamOptimizer(
            learning_rate = self.learning_rate,
            beta1 = self.beta1,
            name = name
        )
        return optimizer

    def _cross_entropy_loss_w_logits(self, labels, logits):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, \
                                                          logits=logits)
        loss = tf.reduce_mean(loss)
        return loss

    def _sigmoid_cross_entopy_w_logits(self, labels, logits):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits)
        loss = tf.reduce_mean(loss)
        return loss

