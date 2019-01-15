'''
This is a python file that used for training GAN.
TODO: provide a parser access from terminal.
'''
## Import module
import sys, os
if os.getcwd().endswith("DCGAN"):
    root_dir = os.getcwd()
else:
    root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
import tensorflow as tf
from datetime import datetime
from pytz import timezone
from tensorflow.python import debug as tf_debug

from Training.train_base import Train_base
from Training.Saver import Saver
from Training.Summary import Summary
from utils import *

class Train_UnrolledGAN(Train_base):
    ## TODO: fix the bug of unrolled GAN
    def __init__(self, config, log_dir, save_dir, **kwargs):
        super(Train_UnrolledGAN, self).__init__(config.LEARNING_RATE, config.BETA1)
        self.config = config
        self.save_dir = save_dir
        self.comments = kwargs.get('comments', '')
        if self.config.SUMMARY:
            self.summary = Summary(log_dir, config, \
                                       log_comments=kwargs.get('comments', ''))

    def train(self, Model, DataSet, SAMPLE_X = None, SAMPLE_Y = None):
        # Reset tf graph.
        tf.reset_default_graph()

        # Create input node
        if not self.config.Y_LABLE:
            image_batch, init_op, dataset = self._input_fn(DataSet)
        else:
            image_batch, label_batch, init_op, dataset = self._input_fn_w_label(DataSet)
            # image, label = self._input_fn_NP()  # using numpy array as feed dict

        # Build up the graph and loss
        with tf.device('/gpu:0'):
            # Create placeholder
            if self.config.Y_LABLE:
                y = tf.placeholder(tf.float32, [self.config.BATCH_SIZE, self.config.NUM_CLASSES], name='y') # label batch
                x = tf.placeholder(tf.float32, [self.config.BATCH_SIZE] + self.config.IMAGE_DIM,
                                   name ='real_images')  # real image
            else:
                y = None
                x = tf.placeholder(tf.float32, [self.config.BATCH_SIZE] + [self.config.IMAGE_HEIGHT_O, self.config.IMAGE_WIDTH_O,
                                                                           self.config.CHANNEL], name='real_images')  # real image


            z = tf.placeholder(tf.float32, [self.config.BATCH_SIZE, self.config.Z_DIM]) # latent variable

            # Build up the graph

            G, D, D_logits, D_, D_logits_, fm, fm_, model = self._build_train_graph(x, y, z, Model)
            # # Create the loss:
            # d_loss, g_loss = self._loss(D, D_logits, D_, D_logits_, x, G, fm, fm_, model.discriminator, y)

            samples = model.sampler(z, y)

        # Create optimizer
        with tf.name_scope('Train'):
            t_vars = tf.trainable_variables()
            # theta_G = [var for var in t_vars if 'g_' in var.name]
            # theta_D = [var for var in t_vars if 'd_' in var.name]
            theta_G = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "generator")
            theta_D = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "discriminator")
            optimizer = self._Adam_optimizer()

            # Discriminator loss
            if self.config.LABEL_SMOOTH:
                d_loss_real = self._sigmoid_cross_entopy_w_logits(0.9 * tf.ones_like(D), D_logits)
            else:
                d_loss_real = self._sigmoid_cross_entopy_w_logits(tf.ones_like(D), D_logits)
            d_loss_fake = self._sigmoid_cross_entopy_w_logits(tf.zeros_like(D_), D_logits_)
            d_loss = d_loss_fake + d_loss_real

            d_updates = tf.keras.optimizers.Adam(lr=1e-4, beta_1=self.config.BETA1).get_updates(d_loss, theta_D)
            d_optim = tf.group(*d_updates, name = "d_train_op")

            if self.config.UNROLLED_STEP > 0:
                update_dict = self._extract_update_dict(d_updates)
                cur_update_dict = update_dict
                for i in range(self.config.UNROLLED_STEP - 1):
                    cur_update_dict = self._graph_replace(update_dict, cur_update_dict)
                g_loss = - self._graph_replace(d_loss, cur_update_dict)
            else:
                g_loss = -d_loss

            g_optim = self._train_op(optimizer, g_loss, theta_G)

        # Add summary
        if self.config.SUMMARY:
            summary_dict = {}
            if self.config.SUMMARY_SCALAR:
                scaler = {'generator_loss': g_loss,
                          'discriminator_loss': d_loss}

            merged_summary = self.summary.add_summary(summary_dict)

        # Add saver
        saver = Saver(self.save_dir)

        # Create Session
        sess_config = tf.ConfigProto(allow_soft_placement = True)
        # Use soft_placement to place those variables, which can be placed, on GPU
        with tf.Session(config = sess_config) as sess:
            # if self.config.DEBUG:
            #     sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            if self.config.SUMMARY and self.config.SUMMARY_GRAPH:
                self.summary._graph_summary(sess.graph)

            if self.config.RESTORE:
                start_epoch = saver.restore(sess)
            else:
                saver.set_save_path(comments = self.comments)
                start_epoch = 0
                # initialize the variables
                init_var = tf.group(tf.global_variables_initializer(), \
                                    tf.local_variables_initializer())
                sess.run(init_var)

            sample_z = np.random.normal(size = (self.config.BATCH_SIZE, 100))
            if not self.config.Y_LABLE:
                ## TODO: support PacGAN for conditional case
                sess.run(init_op)
                sample_x = sess.run(image_batch)
            else:
                # sample_x, sample_y = sess.run([image_batch, label_batch])
                sample_x, sample_y = SAMPLE_X, SAMPLE_Y
                # sample_x, sample_y = image[:64, ...], label[:64, ...] # for numpy input

            # Start Training
            tf.logging.info("Start unrolledGAN traininig!")
            for epoch in range(start_epoch + 1, self.config.EPOCHS + start_epoch + 1):
                tf.logging.info("Training for epoch {}.".format(epoch))
                train_pr_bar = tf.contrib.keras.utils.Progbar(target= \
                                                                  int(self.config.TRAIN_SIZE / self.config.BATCH_SIZE))
                sess.run(init_op)
                for i in range(int(self.config.TRAIN_SIZE / self.config.BATCH_SIZE)):
                    batch_z = np.random.normal(size = (self.config.BATCH_SIZE, self.config.Z_DIM)).astype(np.float32)
                    # Fetch a data batch
                    if not self.config.Y_LABLE:
                        image_batch_o = sess.run(image_batch)
                    else:
                        image_batch_o, label_batch_o = sess.run([image_batch, label_batch])


                    ## for numpy input
                    # image_batch_o, label_batch_o = image[i * self.config.BATCH_SIZE : (i + 1) * self.config.BATCH_SIZE], \
                    #                                label[i * self.config.BATCH_SIZE : (i + 1) * self.config.BATCH_SIZE]

                    if not self.config.Y_LABLE:
                        # Update discriminator
                        _, d_loss_o = sess.run([d_optim, d_loss],
                                               feed_dict={x: image_batch_o,
                                                          z: batch_z})
                        # Update generator
                        _ = sess.run([g_optim],
                                     feed_dict={x: image_batch_o,
                                                z: batch_z})
                        _, g_loss_o = sess.run([g_optim, g_loss],
                                               feed_dict={x: image_batch_o,
                                                          z: batch_z})
                    else:
                        # Update discriminator
                        _, d_loss_o = sess.run([d_optim, d_loss],
                                    feed_dict = {x: image_batch_o,
                                                 y: label_batch_o,
                                                 z: batch_z})

                        # Update generator
                        _ = sess.run([g_optim],
                                     feed_dict = {x: image_batch_o,
                                                  y: label_batch_o,
                                                  z: batch_z})
                        _, g_loss_o = sess.run([g_optim, g_loss],
                                     feed_dict = {x: image_batch_o,
                                                  y: label_batch_o,
                                                  z: batch_z})


                    # Update progress bar
                    train_pr_bar.update(i)
                    if i % 100 == 0:
                        if self.config.DEBUG:
                            ## Sample image for every 100 update in debug mode
                            if not self.config.Y_LABLE:
                                samples_o, d_loss_o, g_loss_o, summary_o = sess.run(
                                    [samples, d_loss, g_loss, merged_summary],
                                    feed_dict={x: sample_x,
                                               z: sample_z})
                            else:
                                samples_o, d_loss_o, g_loss_o, summary_o = sess.run(
                                    [samples, d_loss, g_loss, merged_summary],
                                    feed_dict={x: sample_x,
                                               y: sample_y,
                                               z: sample_z})

                            save_images(samples_o, image_manifold_size(samples_o.shape[0]), \
                                        os.path.join(self.config.SAMPLE_DIR, 'train_{:02d}_{:02d}.png'.format(epoch, i)))


                # Save the model per SAVE_PER_EPOCH
                if epoch % self.config.SAVE_PER_EPOCH == 0:
                    save_name = str(epoch)
                    saver.save(sess, 'model_' + save_name.zfill(4) + '.ckpt')

                print("Epoch: [%2d/%2d], d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, self.config.EPOCHS + start_epoch, d_loss_o, g_loss_o))
                ## Sample image after every epoch
                if not self.config.Y_LABLE:
                    samples_o, d_loss_o, g_loss_o, summary_o = sess.run([samples, d_loss, g_loss, merged_summary],
                                                                            feed_dict={x: sample_x,
                                                                                       z: sample_z})
                else:
                    samples_o, d_loss_o, g_loss_o, summary_o = sess.run([samples, d_loss, g_loss, merged_summary],
                                                       feed_dict = {x: sample_x,
                                                                    y: sample_y,
                                                                    z: sample_z})

                if self.config.SUMMARY:
                    self.summary.summary_writer.add_summary(summary_o, epoch)

                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss_o, g_loss_o))
                save_images(samples_o, image_manifold_size(samples_o.shape[0]), \
                            os.path.join(self.config.SAMPLE_DIR, 'train_{:02d}.png'.format(epoch)))

            if self.config.SUMMARY:
                self.summary.summary_writer.flush()
                self.summary.summary_writer.close()

            # Save the model after all epochs
            save_name = str(epoch)
            saver.save(sess, 'model_' + save_name.zfill(4) + '.ckpt')
            return

    def _build_train_graph(self, x, y, z, Model):
        """
        Build up the training graph
        :return:
        G: generated image batch
        D: probability (after sigmoid)
        D_logits: logits before sigmoid
        D_: probability for fake data
        D_logits_: logits before sigmoid for fake data
        """
        ## Create the model
        main_graph = Model(self.config)
        G, D, D_logits, D_, D_logits_, fm, fm_ = main_graph.forward_pass(z, x, y)
        return G, D, D_logits, D_, D_logits_, fm, fm_, main_graph



    def _input_fn_w_label(self, DataSet):
        """
        Create the input node
        :return:
        """
        with tf.device('/cpu:0'):
            with tf.name_scope('Input_Data'):
                # Training dataset
                dataset = DataSet(self.config.DATA_DIR, self.config, use_augmentation = True)
                # Inputpipeline
                image_batch, label_batch, init_op = dataset.inputpipline_singleset()
        return image_batch, label_batch, init_op, dataset

    def _input_fn(self, DataSet):
        with tf.device('/cpu:0'):
            with tf.name_scope('Input_Data'):
                # Training dataset
                dataset = DataSet(self.config.DATA_DIR, self.config, use_augmentation = True)
                # Inputpipeline
                try:
                    image_batch, init_op = dataset.inputpipline_singleset()
                except:
                    image_batch, _, init_op = dataset.inputpipline_singleset()
        return image_batch, init_op, dataset

    def _input_fn_NP(self, DataSet):
        """
        Create the input node using numpy function
        :return:
        """
        dataset = DataSet(self.config.DATA_DIR, self.config)
        X, y = dataset.load_mnist()
        return X, y