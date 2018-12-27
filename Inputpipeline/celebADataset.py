#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 16:54:08 2018
This script demonstrate an example of inputpipline on using prostate dataset
@author: wenyuan
"""
import os, sys
root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
import tensorflow as tf
import numpy as np


class celebADataSet(object):
    """
    Mnist Dataset
    """
    def __init__(self, data_dir, config, subset = None, use_augmentation=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_augmentation = use_augmentation
        self.config = config

    def get_filenames(self):
        return [os.path.join(self.data_dir, 'Tfrecord/' \
                                + 'celebA.tfrecords')]

    def input_from_tfrecord_filename(self):
        filename = self.get_filenames()
        dataset = tf.data.TFRecordDataset(filename)
        return dataset

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([self.config.IMAGE_HEIGHT * self.config.IMAGE_WIDTH * self.config.CHANNEL])

        image = tf.cast(tf.reshape(image, self.config.IMAGE_DIM), tf.float32)
        if self.use_augmentation:
            image = self.preprocessing(image)
        return image

    def preprocessing(self, image):
        if self.config.CROP:
            assert((self.config.IMAGE_WIDTH_O is not None) and (self.config.IMAGE_HEIGHT_O is not None)), \
                "Please specify the image shape after crop."
            image = tf.image.resize_image_with_crop_or_pad(image,
                                                           target_height = 108,
                                                           target_width = 108)
            image = tf.image.resize_images(image, size = [self.config.IMAGE_HEIGHT_O, self.config.IMAGE_WIDTH_O])
        image = tf.div(image - 127.5, 255) - 1
        return image

    def shuffle_and_repeat(self, dataset, repeat = -1):
        dataset = dataset.shuffle(buffer_size= \
                                      self.config.MIN_QUEUE_EXAMPLES + \
                                      3 * self.config.BATCH_SIZE, \
                                  )
        dataset = dataset.repeat(repeat)
        return dataset

    def batch(self, dataset):
        dataset = dataset.batch(batch_size=self.config.BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=self.config.BATCH_SIZE)
        return dataset

    def inputpipline_singleset(self):
        # 1 Read in tfrecords
        dataset = self.input_from_tfrecord_filename()
        # 2 Parser tfrecords and preprocessing the data
        dataset = dataset.map(self.parser, \
                              num_parallel_calls=self.config.BATCH_SIZE)

        # 3 Shuffle and repeat
        dataset = self.shuffle_and_repeat(dataset, repeat = self.config.REPEAT)
        # 4 Batch it up
        dataset = self.batch(dataset)
        # 5 Make iterator
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        image_batch = iterator.get_next()
        image_batch.set_shape([self.config.BATCH_SIZE] + \
                              [self.config.IMAGE_HEIGHT_O, self.config.IMAGE_WIDTH_O, self.config.CHANNEL])
        return image_batch, init_op

###### Testing code
def _main_inputpipline_singleset():
    from config import Config

    class tempConfig(Config):
        BATCH_SIZE = 64
        REPEAT = 1

        ## Input image
        IMAGE_HEIGHT = 218
        IMAGE_WIDTH = 178
        CHANNEL = 3

        ## Total number of images: 202,599

        # Crop and resize
        CROP = True
        IMAGE_HEIGHT_O = 64
        IMAGE_WIDTH_O = 64

    tmp_config = tempConfig()
    data_dir = os.path.join(root_dir, "Dataset/celebA")

    num_dataset = 0
    tf.reset_default_graph()
    with tf.device('/cpu:0'):
        dataset = celebADataSet(data_dir, tmp_config, use_augmentation = True)
        image_batch, init_op = dataset.inputpipline_singleset()
        with tf.Session() as sess:
            sess.run(init_op)
            while True:
                try:
                    image_batch_output = \
                        sess.run(image_batch)
                    num_dataset += 1
                except tf.errors.OutOfRangeError:
                    break
    print(image_batch_output.shape, num_dataset)

if __name__ == "__main__":
    _main_inputpipline_singleset()