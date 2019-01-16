import sys, os
if os.getcwd().endswith("DCGAN"):
    root_dir = os.getcwd()
else:
    root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
import tensorflow as tf
from Model import model_base

class DCGAN(model_base.GAN_Base):
    def __init__(self, config):
        super(DCGAN, self).__init__(config.DATA_FORMAT, \
                                    config.BATCH_NORM_DECAY, config.BATCH_NORM_EPSILON)
        self.config = config

    def mrGAN_encoder(self, image, reuse = False):
        with tf.variable_scope("encoder") as scope:
            if reuse:
                scope.reuse_variables()

            if self.config.CHANNEL == 3:
                h0 = self._conv2d(image, 64, name = 'e_h0_conv')
                h0 = tf.nn.leaky_relu(h0)

                h1 = self._conv2d(h0, 64 * 2, name = 'e_h1_conv')
                h1 = self._batch_norm_contrib(h1, name = 'd_h1_bn', train = True)
                h1 = tf.nn.leaky_relu(h1)

                h2 = self._conv2d(h1, 64 * 4, name = 'e_h2_conv')
                h2 = self._batch_norm_contrib(h2, name = 'd_h2_bn', train = True)
                h2 = tf.nn.leaky_relu(h2)

                h3 = self._conv2d(h2, 64 * 8, name = 'e_h3_conv')
                h3 = self._batch_norm_contrib(h3, name = 'e_h3_bn', train = True)
                h3 = tf.nn.leaky_relu(h3)

                h4 = tf.reshape(h3, [self.config.BATCH_SIZE, -1])

                h4 = self._linear_fc(h4, 100, 'e_h4_lin')
                return tf.tanh(h4)

            else:
                # first conv
                h0 = self._conv2d(image, 1 + self.config.NUM_CLASSES, name = 'e_h0_conv')
                h0 = tf.nn.leaky_relu(h0, alpha = 0.2, name = 'e_leaky0')

                # second conv
                h1 = self._conv2d(h0, 64 + self.config.NUM_CLASSES, name = 'e_h1_conv')
                h1 = self._batch_norm_contrib(h1, name = 'e_h1_bn', train = True)
                h1 = tf.nn.leaky_relu(h1, alpha = 0.2, name = 'e_leaky1')

                # reshape and concat the label
                h1 = tf.reshape(h1, [self.config.BATCH_SIZE, -1])

                ## fc layer
                h2 = self._linear_fc(h1, 1024, 'e_h2_lin')
                h2 = self._batch_norm_contrib(h2, name = 'e_h2_bn', train = True)
                h2= tf.nn.leaky_relu(h2, alpha = 0.2, name = 'e_leaky2')

                h3 = self._linear_fc(h2, 100, 'e_h3_lin')

                return h3


    def generator(self, z, y = None, reuse = False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            if not self.config.Y_LABEL: ## there is no y, don't use conditional GAN
                if self.config.CHANNEL == 1:
                    ## first linear layer
                    h0 = self._linear_fc(z, 1024, 'g_h0_lin')
                    h0 = self._batch_norm_contrib(h0, 'g_bn0', train=True)
                    h0 = tf.nn.relu(h0, 'g_rl0')

                    ## second linear layer
                    h1 = self._linear_fc(h0, self.config.BATCH_SIZE * 2 * 7 * 7, 'g_h1_lin')
                    h1 = self._batch_norm_contrib(h1, 'g_bn1', train=True)
                    h1 = tf.nn.relu(h1, 'g_rl1')

                    ## reshape to conv feature pack and concat with label condition
                    h1 = tf.reshape(h1, [self.config.BATCH_SIZE, 7, 7, 64 * 2])

                    ## first layer deconv
                    h2 = self._deconv2d(h1, 128, name='g_dconv0')
                    h2 = self._batch_norm_contrib(h2, 'g_bn2', train=True)
                    h2 = tf.nn.relu(h2, 'g_rl2')

                    ## output layer: sigmoid to map the data range to [0, 1]
                    h3 = self._deconv2d(h2, 1, name='g_dconv1')
                    h3 = tf.nn.sigmoid(h3, name='sigmoid')
                    return h3

                else:
                    # project 'z' and reshape
                    z = self._linear_fc(z, 64 * 8 * 4 * 4, 'g_h0_lin')
                    h0 = tf.reshape(z, [-1, 4, 4, 64 * 8])
                    h0 = self._batch_norm_contrib(h0, 'g_bn0', train = True)
                    h0 = tf.nn.relu(h0, 'g_rl0') ## [4, 4]

                    h1 = self._deconv2d(h0, 64 * 4, name = 'g_dconv0')
                    h1 = self._batch_norm_contrib(h1, 'g_bn1', train = True)
                    h1 = tf.nn.relu(h1, 'g_rl1') ## [8, 8]

                    h2 = self._deconv2d(h1, 64 * 2, name = 'g_dconv1')
                    h2 = self._batch_norm_contrib(h2, 'g_bn2', train = True)
                    h2 = tf.nn.relu(h2, 'g_rl2') ## [16, 16]

                    h3 = self._deconv2d(h2, 64 * 1, name = 'g_dconv2')
                    h3 = self._batch_norm_contrib(h3, 'g_bn3', train = True)
                    h3 = tf.nn.relu(h3, 'g_rl3') ## [32, 32]

                    h4 = self._deconv2d(h3, self.config.CHANNEL, name = 'g_dconv3')
                    h4 = tf.nn.tanh(h4)
                    ## [64, 64]
                    return h4

            else: ## use conditional GAN
                if self.config.DATA_NAME == "mnist":
                    yb = tf.reshape(y, [self.config.BATCH_SIZE, 1, 1, self.config.NUM_CLASSES]) ## [None, 1, 1, 10]
                    z = tf.concat([z, y], 1) # concat the z and y in the latent space

                    ## first linear layer
                    h0 = self._linear_fc(z, 1024, 'g_h0_lin')
                    h0 = self._batch_norm_contrib(h0, 'g_bn0', train = True)
                    h0 = tf.nn.relu(h0, 'g_rl0')
                    h0 = tf.concat([h0, y], 1)

                    ## second linear layer
                    h1 = self._linear_fc(h0, self.config.BATCH_SIZE * 2 * 7 * 7, 'g_h1_lin')
                    h1 = self._batch_norm_contrib(h1, 'g_bn1', train = True)
                    h1 = tf.nn.relu(h1, 'g_rl1')

                    ## reshape to conv feature pack and concat with label condition
                    h1 = tf.reshape(h1, [self.config.BATCH_SIZE, 7, 7, 64 * 2])
                    h1 = self._conv_cond_concat(h1, yb)

                    ## first layer deconv
                    h2 = self._deconv2d(h1, 128, name = 'g_dconv0')
                    h2 = self._batch_norm_contrib(h2, 'g_bn2', train = True)
                    h2 = tf.nn.relu(h2, 'g_rl2')
                    h2 = self._conv_cond_concat(h2, yb)

                    ## output layer: sigmoid to map the data range to [0, 1]
                    h3 = self._deconv2d(h2, 1, name = 'g_dconv1')
                    h3 = tf.nn.sigmoid(h3, name = 'sigmoid')
                    return h3
                elif self.config.DATA_NAME == "prostate":
                    # project 'z' and reshape
                    yb = tf.reshape(y, [self.config.BATCH_SIZE, 1, 1, self.config.NUM_CLASSES])
                    z = tf.concat([z, y], 1)

                    z = self._linear_fc(z, 64 * 8 * 4 * 4, 'g_h0_lin')
                    h0 = tf.reshape(z, [-1, 4, 4, 64 * 8])
                    h0 = self._batch_norm_contrib(h0, 'g_bn0', train=True)
                    h0 = tf.nn.relu(h0, 'g_rl0')  ## [4, 4]
                    h0 = self._conv_cond_concat(h0, yb)

                    h1 = self._deconv2d(h0, 64 * 4, name='g_dconv0')
                    h1 = self._batch_norm_contrib(h1, 'g_bn1', train=True)
                    h1 = tf.nn.relu(h1, 'g_rl1')  ## [8, 8]
                    h1 = self._conv_cond_concat(h1, yb)

                    h2 = self._deconv2d(h1, 64 * 2, name='g_dconv1')
                    h2 = self._batch_norm_contrib(h2, 'g_bn2', train=True)
                    h2 = tf.nn.relu(h2, 'g_rl2')  ## [16, 16]
                    h2 = self._conv_cond_concat(h2, yb)

                    h3 = self._deconv2d(h2, 64 * 1, name='g_dconv2')
                    h3 = self._batch_norm_contrib(h3, 'g_bn3', train=True)
                    h3 = tf.nn.relu(h3, 'g_rl3')  ## [32, 32]
                    h3 = self._conv_cond_concat(h3, yb)

                    h4 = self._deconv2d(h3, 3, name='g_dconv3')
                    h4 = tf.nn.tanh(h4)
                    ## [64, 64]
                    return h4

    def discriminator(self, image, y = None, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.config.Y_LABEL:
                if self.config.CHANNEL == 3:
                    image = self._add_noise(image)
                    h0 = self._conv2d(image, 64, name = 'd_h0_conv')
                    h0 = tf.nn.leaky_relu(h0)

                    h1 = self._conv2d(h0, 64 * 2, name = 'd_h1_conv')
                    h1 = self._batch_norm_contrib(h1, name = 'd_h1_bn', train = True)
                    h1 = tf.nn.leaky_relu(h1)

                    h2 = self._conv2d(h1, 64 * 4, name = 'd_h2_conv')
                    h2 = self._batch_norm_contrib(h2, name = 'd_h2_bn', train = True)
                    h2 = tf.nn.leaky_relu(h2)

                    h3 = self._conv2d(h2, 64 * 8, name = 'd_h3_conv')
                    h3 = self._batch_norm_contrib(h3, name = 'd_h3_bn', train = True)
                    h3 = tf.nn.leaky_relu(h3)
                    fm = h3

                    h4 = tf.reshape(h3, [self.config.BATCH_SIZE, -1])

                    if self.config.MINIBATCH_DIS:
                        f = self._minibatch_discrimination(h4, 100)
                        h4 = tf.concat([h4, f], 1)

                    h4 = self._linear_fc(h4, 1, 'd_h4_lin')
                    return tf.nn.sigmoid(h4), h4, fm
                else:
                    image = self._add_noise(image)
                    # first conv
                    h0 = self._conv2d(image, 1 + self.config.NUM_CLASSES, name='d_h0_conv')
                    h0 = tf.nn.leaky_relu(h0, alpha=0.2, name='d_leaky0')

                    # second conv
                    h1 = self._conv2d(h0, 64 + self.config.NUM_CLASSES, name='d_h1_conv')
                    h1 = self._batch_norm_contrib(h1, name='d_h1_bn', train=True)
                    h1 = tf.nn.leaky_relu(h1, alpha=0.2, name='d_leaky1')

                    fm = h1
                    # reshape and concat the label
                    h1 = tf.reshape(h1, [self.config.BATCH_SIZE, -1])

                    ## fc layer
                    h2 = self._linear_fc(h1, 1024, 'd_h2_lin')
                    h2 = self._batch_norm_contrib(h2, name='d_h2_bn', train=True)
                    h2 = tf.nn.leaky_relu(h2, alpha=0.2, name='d_leaky2')

                    if self.config.MINIBATCH_DIS:
                        f = self._minibatch_discrimination(h2, 100)
                        h2 = tf.concat([h2, f], 1)

                    h3 = self._linear_fc(h2, 1, 'd_h3_lin')
                    return tf.nn.sigmoid(h3), h3, fm

            else:
                if self.config.DATA_NAME == "mnist":
                    image = self._add_noise(image)
                    yb = tf.reshape(y, [self.config.BATCH_SIZE, 1, 1, self.config.NUM_CLASSES])
                    image = self._conv_cond_concat(image, yb)

                    # first conv
                    h0 = self._conv2d(image, 1 + self.config.NUM_CLASSES, name = 'd_h0_conv')
                    h0 = tf.nn.leaky_relu(h0, alpha = 0.2, name = 'd_leaky0')
                    h0 = self._conv_cond_concat(h0, yb)

                    # second conv
                    h1 = self._conv2d(h0, 64 + self.config.NUM_CLASSES, name = 'd_h1_conv')
                    h1 = self._batch_norm_contrib(h1, name = 'd_h1_bn', train = True)
                    h1 = tf.nn.leaky_relu(h1, alpha = 0.2, name = 'd_leaky1')

                    fm = h1
                    # reshape and concat the label
                    h1 = tf.reshape(h1, [self.config.BATCH_SIZE, -1])
                    h1 = tf.concat([h1, y], 1)

                    ## fc layer
                    h2 = self._linear_fc(h1, 1024, 'd_h2_lin')
                    h2 = self._batch_norm_contrib(h2, name = 'd_h2_bn', train = True)
                    h2= tf.nn.leaky_relu(h2, alpha = 0.2, name = 'd_leaky2')
                    h2 = tf.concat([h2, y], 1)
                    if self.config.MINIBATCH_DIS:
                        f = self._minibatch_discrimination(h2, 100)
                        h2 = tf.concat([h2, f], 1)
                    h3 = self._linear_fc(h2, 1, 'd_h3_lin')
                    return tf.nn.sigmoid(h3), h3, fm

                elif self.config.DATA_NAME == "prostate":
                    image = self._add_noise(image)

                    yb = tf.reshape(y, [self.config.BATCH_SIZE, 1, 1, self.config.NUM_CLASSES])
                    image = self._conv_cond_concat(image, yb)

                    h0 = self._conv2d(image, 64, name='d_h0_conv')
                    h0 = tf.nn.leaky_relu(h0)
                    h0 = self._conv_cond_concat(h0, yb)

                    h1 = self._conv2d(h0, 64 * 2, name='d_h1_conv')
                    h1 = self._batch_norm_contrib(h1, name='d_h1_bn', train=True)
                    h1 = tf.nn.leaky_relu(h1)
                    h1 = self._conv_cond_concat(h1, yb)

                    h2 = self._conv2d(h1, 64 * 4, name='d_h2_conv')
                    h2 = self._batch_norm_contrib(h2, name='d_h2_bn', train=True)
                    h2 = tf.nn.leaky_relu(h2)
                    h2 = self._conv_cond_concat(h2, yb)

                    h3 = self._conv2d(h2, 64 * 8, name='d_h3_conv')
                    h3 = self._batch_norm_contrib(h3, name='d_h3_bn', train=True)
                    h3 = tf.nn.leaky_relu(h3)
                    fm = h3
                    h3 = self._conv_cond_concat(h3, yb)

                    h4 = tf.reshape(h3, [self.config.BATCH_SIZE, -1])

                    if self.config.MINIBATCH_DIS:
                        f = self._minibatch_discrimination(h4, 100)
                        h4 = tf.concat([h4, f], 1)

                    h4 = tf.concat([h4, y], 1)
                    h4 = self._linear_fc(h4, 1, 'd_h4_lin')
                    return tf.nn.sigmoid(h4), h4, fm

    def forward_pass(self, z, image, label = None):
        """

        :param z: latent variable
        :param image: input image
        :param label: input label (e.g. mnist)
        :return:
        """
        if self.config.LOSS == "MRGAN":
            G = self.generator(z, label)
            G_mr = self.generator(self.mrGAN_encoder(G), label, reuse = True)
            D, D_logits, fm = self.discriminator(image, label, reuse = False)
            D_, D_logits_, fm_ = self.discriminator(G, label, reuse = True)
            D_mr, D_mr_logits, fm_mr = self.discriminator(G_mr, label, reuse = True)
            return G, G_mr, D, D_logits, D_, D_logits_, fm, fm_, D_mr, D_mr_logits, fm_mr
        # elif self.config.LOSS == "VEEGAN":
        #     ## Currently VEEGAN doesn't suport condition GAN
        #     G = self.generator(z)
        #     z_vee = self.mrGAN_encoder(G)
        #     ## Connect image and latent vector:
        #     G_z = self._conv_cond_concat(G, z)
        #     G_z_vee = self._conv_cond_concat(image, z_vee)
        #     D, D_logits, fm = self.discriminator(G_z, reuse = False)
        #     D_, D_logits_, fm_ = self.discriminator(G_z_vee, reuse = True)

        else:
            if self.config.LOSS == "PacGAN":
                G_sep = []
                for i in range(self.config.PAC_NUM):
                    reuse = True if i > 0 else False
                    G_sep.append(self.generator(z[i], label, reuse))
                G = tf.concat(G_sep, 3)
            else:
                G = self.generator(z, label)
            D, D_logits, fm = self.discriminator(image, label, reuse = False)
            D_, D_logits_, fm_ = self.discriminator(G, label, reuse = True)
            return G, D, D_logits, D_, D_logits_, fm, fm_

    def sampler(self, z, y = None):
        with tf.variable_scope("generator", reuse = tf.AUTO_REUSE) as scope:
            if not self.config.Y_LABEL:
                tf.logging.info("Apply unconditional GAN!")
                if self.config.CHANNEL == 3:
                    # project 'z' and reshape
                    z = self._linear_fc(z, 64 * 8 * 4 * 4, 'g_h0_lin')
                    h0 = tf.reshape(z, [-1, 4, 4, 64 * 8])
                    h0 = self._batch_norm_contrib(h0, 'g_bn0', train = False)
                    h0 = tf.nn.relu(h0, 'g_rl0')  ## [4, 4]

                    h1 = self._deconv2d(h0, 64 * 4, name='g_dconv0')
                    h1 = self._batch_norm_contrib(h1, 'g_bn1', train = False)
                    h1 = tf.nn.relu(h1, 'g_rl1')  ## [8, 8]

                    h2 = self._deconv2d(h1, 64 * 2, name='g_dconv1')
                    h2 = self._batch_norm_contrib(h2, 'g_bn2', train = False)
                    h2 = tf.nn.relu(h2, 'g_rl2')  ## [16, 16]

                    h3 = self._deconv2d(h2, 64 * 1, name='g_dconv2')
                    h3 = self._batch_norm_contrib(h3, 'g_bn3', train = False)
                    h3 = tf.nn.relu(h3, 'g_rl3')  ## [32, 32]

                    h4 = self._deconv2d(h3, self.config.CHANNEL, name='g_dconv3')
                    h4 = tf.nn.tanh(h4)
                    ## [64, 64]
                    return h4
                else:
                    ## first linear layer
                    h0 = self._linear_fc(z, 1024, 'g_h0_lin')
                    h0 = self._batch_norm_contrib(h0, 'g_bn0', train=True)
                    h0 = tf.nn.relu(h0, 'g_rl0')

                    ## second linear layer
                    h1 = self._linear_fc(h0, self.config.BATCH_SIZE * 2 * 7 * 7, 'g_h1_lin')
                    h1 = self._batch_norm_contrib(h1, 'g_bn1', train=True)
                    h1 = tf.nn.relu(h1, 'g_rl1')

                    ## reshape to conv feature pack and concat with label condition
                    h1 = tf.reshape(h1, [self.config.BATCH_SIZE, 7, 7, 64 * 2])

                    ## first layer deconv
                    h2 = self._deconv2d(h1, 128, name='g_dconv0')
                    h2 = self._batch_norm_contrib(h2, 'g_bn2', train=True)
                    h2 = tf.nn.relu(h2, 'g_rl2')

                    ## output layer: sigmoid to map the data range to [0, 1]
                    h3 = self._deconv2d(h2, 1, name='g_dconv1')
                    h3 = tf.nn.sigmoid(h3, name='sigmoid')
                    return h3

            else:
                tf.logging.info("Apply conditional GAN!")
                if self.config.DATA_NAME == "mnist":
                    yb = tf.reshape(y, [self.config.BATCH_SIZE, 1, 1, self.config.NUM_CLASSES])  ## [None, 1, 1, 10]
                    z = tf.concat([z, y], 1)  # concat the z and y in the latent space

                    ## first linear layer
                    h0 = self._linear_fc(z, 1024, 'g_h0_lin')
                    h0 = self._batch_norm_contrib(h0, 'g_bn0', train = False)
                    h0 = tf.nn.relu(h0, 'g_rl0')
                    h0 = tf.concat([h0, y], 1)

                    ## second linear layer
                    h1 = self._linear_fc(h0, self.config.BATCH_SIZE * 2 * 7 * 7, 'g_h1_lin')
                    h1 = self._batch_norm_contrib(h1, 'g_bn1', train = False)
                    h1 = tf.nn.relu(h1, 'g_rl1')

                    ## reshape to conv feature pack and concat with label condition
                    h1 = tf.reshape(h1, [self.config.BATCH_SIZE, 7, 7, 64 * 2])
                    h1 = self._conv_cond_concat(h1, yb)

                    ## first layer deconv
                    h2 = self._deconv2d(h1, 128, name='g_dconv0')
                    h2 = self._batch_norm_contrib(h2, 'g_bn2', train = False)
                    h2 = tf.nn.relu(h2, 'g_rl2')
                    h2 = self._conv_cond_concat(h2, yb)

                    ## output layer: sigmoid to map the data range to [0, 1]
                    h3 = self._deconv2d(h2, 1, name='g_dconv1')
                    h3 = tf.nn.sigmoid(h3, name='sigmoid')
                    return h3
                elif self.config.DATA_NAME == "prostate":
                    # project 'z' and reshape
                    yb = tf.reshape(y, [self.config.BATCH_SIZE, 1, 1, self.config.NUM_CLASSES])
                    z = tf.concat([z, y], 1)

                    z = self._linear_fc(z, 64 * 8 * 4 * 4, 'g_h0_lin')
                    h0 = tf.reshape(z, [-1, 4, 4, 64 * 8])
                    h0 = self._batch_norm_contrib(h0, 'g_bn0', train=True)
                    h0 = tf.nn.relu(h0, 'g_rl0')  ## [4, 4]
                    h0 = self._conv_cond_concat(h0, yb)

                    h1 = self._deconv2d(h0, 64 * 4, name='g_dconv0')
                    h1 = self._batch_norm_contrib(h1, 'g_bn1', train=True)
                    h1 = tf.nn.relu(h1, 'g_rl1')  ## [8, 8]
                    h1 = self._conv_cond_concat(h1, yb)

                    h2 = self._deconv2d(h1, 64 * 2, name='g_dconv1')
                    h2 = self._batch_norm_contrib(h2, 'g_bn2', train=True)
                    h2 = tf.nn.relu(h2, 'g_rl2')  ## [16, 16]
                    h2 = self._conv_cond_concat(h2, yb)

                    h3 = self._deconv2d(h2, 64 * 1, name='g_dconv2')
                    h3 = self._batch_norm_contrib(h3, 'g_bn3', train=True)
                    h3 = tf.nn.relu(h3, 'g_rl3')  ## [32, 32]
                    h3 = self._conv_cond_concat(h3, yb)

                    h4 = self._deconv2d(h3, 3, name='g_dconv3')
                    h4 = tf.nn.tanh(h4)
                    ## [64, 64]
                    return h4


if __name__ == "__main__":
    pass