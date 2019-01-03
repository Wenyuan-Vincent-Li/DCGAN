#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 17:54:43 2018
Template
Base Configurations class.
@author: wenyuan
"""

"""
Written by Wenyuan Li
"""

import math
import numpy as np


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    BATCH_SIZE = 64

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Image data format
    DATA_FORMAT = "channels_last"

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Number of classification classes (including background)
    NUM_CLASSES = None # Override in sub-classes

    # Whether to use the Y_LABEL as conditional GAN
    Y_LABLE = None

    # Latent variable z dimension
    Z_DIM = 100

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Network Property
    BATCH_NORM_DECAY = 0.9
    BATCH_NORM_EPSILON = 1e-5

    # Training Property
    EPOCHS = 50
    LEARNING_RATE = 0.0002
    BETA1 = 0.5
    SAVE_PER_EPOCH = 2
    RESTORE = True

    # Loss Function
    LOSS = "GAN" ## selection of ["GAN", "WGAN", "WGAN_GP", "LSGAN"]
    WEIGHT_CLIP = 0.01


    # Input Pipeline
    DATA_DIR = ""  # rewrite this as dataset directory.
    # Input image
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    CHANNEL = 1
    REPEAT = -1
    MIN_QUEUE_EXAMPLES = 15
    # Crop and resize
    CROP = False
    IMAGE_HEIGHT_O = None
    IMAGE_WIDTH_O = None

    # Summary
    SUMMARY = True
    SUMMARY_GRAPH = True
    SUMMARY_SCALAR = True
    SUMMARY_IMAGE = False
    SUMMARY_HISTOGRAM = False

    # SAVE RESULTS:
    SAMPLE_DIR = None

    def __init__(self):
        """Set values of computed attributes."""
        self.MIN_QUEUE_EXAMPLES = int(15 * 0.4)
        self.IMAGE_DIM = [self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNEL]

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def config_str(self):
        """Return a configurations string"""
        s = "\nConfigurations:\n"
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                s += "{:30} {}".format(a, getattr(self, a))
                s += "\n"
        return s
