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

    def _RMSProp_optimizer(self, name = 'RMSProp_optimizer'):
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate = self.learning_rate,
            decay = 0.9,
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


    def _loss_GAN(self, D, D_logits, D_, D_logits_):
        """
        :param D: (0, 1) after sigmoid function for real data
        :param D_logits: logits for real data
        :param D_: (0, 1) after sigmoid function for fake data
        :param D_logits_: logits for fake data
        :return:
        """
        with tf.name_scope('Loss'):
            # Discriminator loss
            if self.config.LABEL_SMOOTH:
                d_loss_real = self._sigmoid_cross_entopy_w_logits(0.9 * tf.ones_like(D), D_logits)
            else:
                d_loss_real = self._sigmoid_cross_entopy_w_logits(tf.ones_like(D), D_logits)
            d_loss_fake = self._sigmoid_cross_entopy_w_logits(tf.zeros_like(D_), D_logits_)
            d_loss = d_loss_fake + d_loss_real
            # Generator loss
            g_loss = self._sigmoid_cross_entopy_w_logits(tf.ones_like(D_), D_logits_)
        return d_loss, g_loss

    def _loss_FMGAN(self, D, D_logits, D_, D_logits_, real, fake, real_fm, fake_fm, discriminator, label = None):
        with tf.name_scope('Loss'):
            # Discriminator loss
            wd = tf.reduce_mean(D_logits) - tf.reduce_mean(D_logits_)
            gp = self._gradient_penalty(real, fake, discriminator, label)
            d_loss = -wd + gp * 10.0
            # Generator loss
            fm_loss = tf.reduce_mean(self._huber_loss(real_fm, fake_fm))
            g_loss = 0 * -tf.reduce_mean(D_logits_) + fm_loss

        return d_loss, g_loss

    def _loss_cGPGAN(self, D, D_logits, D_, D_logits_, real):
        """
        which training methods for gans do actually converge?
        https://arxiv.org/pdf/1801.04406.pdf
        :param D:
        :param D_logits:
        :param D_:
        :param D_logits_:
        :param real:
        :return:
        """
        with tf.name_scope('Loss'):
            # Discriminator loss
            if self.config.LABEL_SMOOTH:
                d_loss_real = self._sigmoid_cross_entopy_w_logits(0.9 * tf.ones_like(D), D_logits)
            else:
                d_loss_real = self._sigmoid_cross_entopy_w_logits(tf.ones_like(D), D_logits)
            d_loss_fake = self._sigmoid_cross_entopy_w_logits(tf.zeros_like(D_), D_logits_)
            d_loss_gp = self._zero_centered_gradient_penalty(D_logits, real)
            d_loss = d_loss_fake + d_loss_real + d_loss_gp * 10
            # Generator loss
            g_loss = self._sigmoid_cross_entopy_w_logits(tf.ones_like(D_), D_logits_)
        return d_loss, g_loss

    def _loss_WGAN(self, D, D_logits, D_, D_logits_):
        with tf.name_scope('Loss'):
            # Discriminator loss
            wd = tf.reduce_mean(D_logits) - tf.reduce_mean(D_logits_)
            d_loss = -wd
            # Generator loss
            g_loss = -tf.reduce_mean(D_logits_)
        return d_loss, g_loss

    def _loss_WGAN_GP(self, D, D_logits, D_, D_logits_, real, fake, discriminator, label = None):
        ## TODO: input function
        with tf.name_scope('Loss'):
            # Discriminator loss
            wd = tf.reduce_mean(D_logits) - tf.reduce_mean(D_logits_)
            gp = self._gradient_penalty(real, fake, discriminator, label)
            d_loss = -wd + gp * 10.0
            # Generator loss
            g_loss = -tf.reduce_mean(D_logits_)
        return d_loss, g_loss

    def _loss_LSGAN(self, D, D_logits, D_, D_logits_):
        with tf.name_scope('Loss'):
            # Discriminator loss
            if self.config.LABEL_SMOOTH:
                d_r_loss = self._sigmoid_cross_entopy_w_logits(0.9 * tf.ones_like(D), D_logits)
            else:
                d_r_loss = self._sigmoid_cross_entopy_w_logits(tf.ones_like(D), D_logits)
            d_f_loss = tf.losses.mean_squared_error(tf.zeros_like(D_logits_), D_logits_)
            d_loss = (d_r_loss + d_f_loss) / 2.0
            # Generator loss
            g_loss = tf.losses.mean_squared_error(tf.ones_like(D_logits), D_logits_)
        return d_loss, g_loss


    def _gradient_penalty(self, real, fake, f, label):
        def interpolate(a, b):
            shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

        x = interpolate(real, fake)
        _, pred, _ = f(x, y = label, reuse = True)
        gradients = tf.gradients(pred, x)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis = 1))
        gp = tf.reduce_mean((slopes - 1.) ** 2)
        return gp

    def _zero_centered_gradient_penalty(self, D_logits, real):
        gradients = tf.gradients(D_logits, real)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis = 1))
        gp = tf.reduce_mean(tf.square(slopes))
        return gp

    def _huber_loss(self, labels, predictions, delta = 1.0):
        residual = tf.abs(predictions - labels)
        condition = tf.less(residual, delta)
        small_res = 0.5 * tf.square(residual)
        large_res = delta * residual - 0.5 * tf.square(delta)
        return tf.where(condition, small_res, large_res)