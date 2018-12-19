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

HEIGHT = 28
WIDTH = 28
DEPTH = 1


class mnistDataSet(object):
    """
    Mnist Dataset
    """

    def __init__(self, data_dir, config, subset='train', use_augmentation=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_augmentation = use_augmentation
        self.config = config

    def get_filenames(self):
        if self.subset in ['train', 'val']:
            return os.path.join(self.data_dir, 'Tfrecord/' \
                                + 'mnist_' + self.subset + '.tfrecords')
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def input_from_tfrecord_placeholder(self):
        filename = tf.placeholder(tf.string, shape=[None], \
                                  name="input_filenames")
        # make filenames as placeholder for training and validating purpose
        dataset = tf.data.TFRecordDataset(filename)
        return dataset, filename

    def input_from_tfrecord_filename(self):
        filename = self.get_filenames()
        dataset = tf.data.TFRecordDataset(filename)
        return dataset

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([HEIGHT * WIDTH * DEPTH])

        label = tf.cast(features['label'], tf.int32)
        image = tf.cast(tf.reshape(image, [HEIGHT, WIDTH, DEPTH]), tf.float32)
        if self.use_augmentation:
            image, label = self.preprocessing(image, label)

        return image, label

    def preprocessing(self, image, label):
        image = tf.div(image, 255)
        label = tf.one_hot(label, depth=self.config.NUM_CLASSES)
        return image, label

    def shuffle_and_repeat(self, dataset):
        dataset = dataset.shuffle(buffer_size= \
                                      self.config.MIN_QUEUE_EXAMPLES + \
                                      3 * self.config.BATCH_SIZE, \
                                  )
        dataset = dataset.repeat(1)
        return dataset

    def batch(self, dataset):
        dataset = dataset.batch(batch_size=self.config.BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=self.config.BATCH_SIZE)
        return dataset

    def inputpipline_singleset(self):
        # 1 Read in tfrecords
        dataset, filename = self.input_from_tfrecord_placeholder()
        # 2 Parser tfrecords and preprocessing the data
        dataset = dataset.map(self.parser, \
                              num_parallel_calls=self.config.BATCH_SIZE)

        # 3 Shuffle and repeat
        dataset = self.shuffle_and_repeat(dataset)
        # 4 Batch it up
        dataset = self.batch(dataset)
        # 5 Make iterator
        iterator = dataset.make_initializable_iterator()
        image_batch, label_batch = iterator.get_next()

        return image_batch, label_batch, filename, iterator
        ## return the input tensor, iterator, placeholder

    def inputpipline_train_val(self, other):
        # 1 Read in tfrecords
        dataset_train = self.input_from_tfrecord_filename()
        dataset_val = other.input_from_tfrecord_filename()
        # 2 Parser tfrecords and preprocessing the data
        dataset_train = dataset_train.map(self.parser, \
                                          num_parallel_calls=self.config.BATCH_SIZE)
        dataset_val = dataset_val.map(self.parser, \
                                      num_parallel_calls=self.config.BATCH_SIZE)
        # 3 Shuffle and repeat
        dataset_train = self.shuffle_and_repeat(dataset_train)
        # 4 Batch it up
        dataset_train = self.batch(dataset_train)
        dataset_val = self.batch(dataset_val)
        # 5 Make iterator
        iterator = tf.data.Iterator.from_structure(dataset_train.output_types,
                                                   dataset_train.output_shapes)
        image_batch, label_batch = iterator.get_next()
        init_op_train = iterator.make_initializer(dataset_train)
        init_op_val = iterator.make_initializer(dataset_val)

        return image_batch, label_batch, init_op_train, init_op_val

###### Testing code
def _main_inputpipline_singleset():
    from config import Config
    class tempConfig(Config):
        BATCH_SIZE = 64

    tmp_config = tempConfig()
    data_dir = os.path.join(root_dir, "Dataset/mnist")

    num_dataset = 0
    tf.reset_default_graph()
    with tf.device('/cpu:0'):
        dataset = mnistDataSet(data_dir, tmp_config, \
                               'train', use_augmentation=False)
        image_batch, label_batch, filename, iterator = dataset.inputpipline_singleset()
        with tf.Session() as sess:
            sess.run(iterator.initializer, \
                     feed_dict={filename: [dataset.get_filenames()]})
            while True:
                try:
                    image_batch_output, label_batch_output = \
                        sess.run([image_batch, label_batch])
                    num_dataset += 1
                except tf.errors.OutOfRangeError:
                    break
    print(image_batch_output.shape, label_batch_output.shape, num_dataset)

if __name__ == "__main__":
    from config import Config
    class tempConfig(Config):
        BATCH_SIZE = 64
    tmp_config = tempConfig()

    data_dir = os.path.join(root_dir, "Dataset/mnist")
    num_dataset = 0
    tf.reset_default_graph()
    with tf.device('/cpu:0'):
        dataset_train = mnistDataSet(data_dir, tmp_config, \
                                        'train', use_augmentation=True)
        dataset_val = mnistDataSet(data_dir, tmp_config, \
                                      'val', use_augmentation=True)

        image_batch, label_batch, init_op_train, init_op_val \
            = dataset_train.inputpipline_train_val(dataset_val)
        with tf.Session() as sess:
            sess.run(init_op_train)
            while True:
                try:
                    image_batch_output, label_batch_output = \
                        sess.run([image_batch, label_batch])
                except tf.errors.OutOfRangeError:
                    break
            sess.run(init_op_val)
            while True:
                try:
                    image_batch_output, label_batch_output = \
                        sess.run([image_batch, label_batch])
                    num_dataset += 1
                except tf.errors.OutOfRangeError:
                    break
    print(image_batch_output.shape, label_batch_output.shape, num_dataset)