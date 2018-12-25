#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 16:04:53 2018
@author: wenyuan
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class GAN_Base(object):
    def __init__(self, data_format, batch_norm_decay=0.9,
                 batch_norm_epsilon=1e-5):
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        assert data_format in ('channels_first', 'channels_last'), \
            "Not valide image data format!"
        self._data_format = data_format

    def forward_pass(self, x, y):
        raise NotImplementedError(
            'forward_pass() is implemented in Model sub classes')

    def sampler(self):
        raise NotImplementedError(
            'sampler() is implemented in Model sub classes')

    def _linear_fc(self, input_, output_size, scope = None, stddev = 0.02, bias_start = 0.0):
        """
        Usually used for convert the latent vector z to the conv feature pack
        :param input_:
        :param output_size:
        :param scope:
        :param stddev:
        :param bias_start:
        :param with_w:
        :return:
        """
        with tf.variable_scope(scope):
            x = tf.layers.dense(
                inputs = input_,
                units = output_size,
                use_bias = True,
                kernel_initializer = tf.random_normal_initializer(stddev = stddev),
                bias_initializer = tf.constant_initializer(bias_start),
                name = scope
            )
        return x

    def _batch_norm(self, x, name, train = False):
        if self._data_format == 'channels_first':
            axis = 1
        else:
            axis = -1
        x = tf.layers.batch_normalization(
            x,
            axis = axis,
            momentum = self._batch_norm_decay,
            center = True,
            scale = True,
            epsilon = self._batch_norm_epsilon,
            training = train,
            name = name)
        return x

    def _batch_norm_contrib(self, x, name, train = False):
        x = tf.contrib.layers.batch_norm(x,
                                        decay = self._batch_norm_decay,
                                        updates_collections = None,
                                        epsilon = self._batch_norm_epsilon,
                                        scale = True,
                                        is_training = train,
                                        scope = name)
        return x

    def _conv_cond_concat(self, x, y):
        """Concatenate conditioning vector on feature map axis."""
        x_shapes = x.get_shape()
        y_shapes = y.get_shape()
        return tf.concat([
            x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

    def _deconv2d(self, input_, output_shape,
                  k_h = 5, k_w = 5, d_h = 2, d_w = 2,
                  stddev = 0.02, name = "deconv2d"):
        with tf.variable_scope(name):
            x = tf.layers.conv2d_transpose(inputs = input_,
                                           filters = output_shape,
                                           kernel_size = (k_h, k_w),
                                           strides = (d_h, d_w),
                                           padding = 'same',
                                           kernel_initializer = tf.random_normal_initializer(stddev = stddev),
                                           name = name)

        return x

    def _conv2d(self, input_, output_dim,
                k_h = 5, k_w = 5, d_h = 2, d_w = 2,
                stddev = 0.02, name = "conv2d"):
        with tf.variable_scope(name):
            x = tf.layers.conv2d(inputs = input_,
                                 filters = output_dim,
                                 kernel_size = (k_h, k_w),
                                 strides = (d_h, d_w),
                                 padding = 'same',
                                 kernel_initializer = tf.truncated_normal_initializer(stddev = stddev),
                                 name = name)
        return x
