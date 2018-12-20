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

    def _linear(self, input_, output_size, scope = None, stddev = 0.02, bias_start = 0.0, with_w = False):
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
        shape = input_.get_shape().as_list()
        with tf.variable_scope(scope or "Linear"):
            try:
                matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                         tf.random_normal_initializer(stddev=stddev))
            except ValueError as err:
                msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
                err.args = err.args + (msg,)
                raise
            bias = tf.get_variable("bias", [output_size],
                                   initializer=tf.constant_initializer(bias_start))
            if with_w:
                return tf.matmul(input_, matrix) + bias, matrix, bias
            else:
                return tf.matmul(input_, matrix) + bias

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

    def _conv_cond_concat(self, x, y):
        """Concatenate conditioning vector on feature map axis."""
        x_shapes = x.get_shape()
        y_shapes = y.get_shape()
        return tf.concat([
            x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

    def _deconv2d(self, input_, output_shape,
                 k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                 name = "deconv2d", with_w = False):
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev = stddev))

            deconv = tf.nn.conv2d_transpose(input_, w, output_shape = output_shape,
                                                strides=[1, d_h, d_w, 1])

            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

            if with_w:
                return deconv, w, biases
            else:
                return deconv

    def _conv2d(self, input_, output_dim, k_h =5, k_w = 5, d_h = 2, d_w = 2, stddev = 0.02, name = "conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
