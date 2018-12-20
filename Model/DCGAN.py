import sys, os
root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
import tensorflow as tf
from Model import model_base

class DCGAN(model_base.GAN_Base):
    def __init__(self, config):
        super(DCGAN, self).__init__(config.DATA_FORMAT, \
                                    config.BATCH_NORM_DECAY, config.BATCH_NORM_EPSILON)
        self.config = config

    def generator(self, z, y = None):
        """
        Generator in GAN
        :param z: [batch_size, latent_dims]
        :param y: [batch_size, NUM_CLASSES]
        :return:
        """
        with tf.variable_scope("generator") as scope:
            if not self.config.NUM_CLASSES:
                pass
            else: ## use conditional GAN
                yb = tf.reshape(y, [self.config.BATCH_SIZE, 1, 1, self.config.NUM_CLASSES]) ## [None, 1, 1, 10]
                z = tf.concat([z, y], 1) # concat the z and y in the latent space

                ## first linear layer
                h0 = self._linear(z, 1024, 'g_h0_lin')
                h0 = self._batch_norm(h0, 'g_bn0', train = True)
                h0 = tf.nn.relu(h0, 'g_rl0')
                h0 = tf.concat([h0, y], 1)

                ## second linear layer
                h1 = self._linear(h0, self.config.BATCH_SIZE * 2 * 7 * 7, 'g_h1_lin')
                h1 = self._batch_norm(h1, 'g_bn1', train = True)
                h1 = tf.nn.relu(h1, 'g_rl1')

                ## reshape to conv feature pack and concat with label condition
                h1 = tf.reshape(h1, [self.config.BATCH_SIZE, 7, 7, 64 * 2])
                h1 = self._conv_cond_concat(h1, yb)

                ## first layer deconv
                h2 = self._deconv2d(h1, [self.config.BATCH_SIZE, 14, 14, 128], name = 'g_dconv0')
                h2 = self._batch_norm(h2, 'g_bn2', train = True)
                h2 = tf.nn.relu(h2, 'g_rl2')
                h2 = self._conv_cond_concat(h2, yb)

                ## output layer: sigmoid to map the data range to [0, 1]
                h3 = self._deconv2d(h2, [self.config.BATCH_SIZE, 28, 28, 1], name = 'g_dconv1')
                h3 = tf.nn.sigmoid(h3, name = 'sigmoid')

                return h3

    def discriminator(self, image, y = None, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.config.NUM_CLASSES:
                pass
            else:
                yb = tf.reshape(y, [self.config.BATCH_SIZE, 1, 1, self.config.NUM_CLASSES])
                x = self._conv_cond_concat(image, yb)

                # first conv
                h0 = self._conv2d(x, 1 + self.config.NUM_CLASSES, name = 'd_h0_conv')
                h0 = tf.nn.leaky_relu(h0, alpha = 0.2, name = 'd_leaky0')
                h0 = self._conv_cond_concat(h0, yb)

                # second conv
                h1 = self._conv2d(h0, 64 + self.config.NUM_CLASSES, name = 'd_h1_conv')
                h1 = self._batch_norm(h1, name = 'd_h1_bn', train = True)
                h1 = tf.nn.leaky_relu(h1, alpha = 0.2, name = 'd_leaky1')

                # reshape and concat the label
                h1 = tf.reshape(h1, [self.config.BATCH_SIZE, -1])
                h1 = tf.concat([h1, y], 1)

                ## fc layer
                h2 = self._linear(h1, 1024, 'd_h2_lin')
                h2 = self._batch_norm(h2, name = 'd_h2_bn', train = True)
                h2= tf.nn.leaky_relu(h2, alpha = 0.2, name = 'd_leaky2')
                h2 = tf.concat([h2, y], 1)

                h3 = self._linear(h2, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h3), h3

    def forward_pass(self, z, image, label = None):
        """

        :param z: latent variable
        :param image: input image
        :param label: input label (e.g. mnist)
        :return:
        """
        G = self.generator(z, label)
        D, D_logits = self.discriminator(image, label, reuse = False)
        ## TODO: self.sampler to visualize the generated data through training
        D_, D_logits_ = self.discriminator(G, label, reuse = True)

        return G, D, D_logits, D_, D_logits_

    def sampler(self, z, y = None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            yb = tf.reshape(y, [self.config.BATCH_SIZE, 1, 1, self.config.NUM_CLASSES])  ## [None, 1, 1, 10]
            z = tf.concat([z, y], 1)  # concat the z and y in the latent space

            ## first linear layer
            h0 = self._linear(z, 1024, 'g_h0_lin')
            h0 = self._batch_norm(h0, 'g_bn0', train = False)
            h0 = tf.nn.relu(h0, 'g_rl0')
            h0 = tf.concat([h0, y], 1)

            ## second linear layer
            h1 = self._linear(h0, self.config.BATCH_SIZE * 2 * 7 * 7, 'g_h1_lin')
            h1 = self._batch_norm(h1, 'g_bn1', train = False)
            h1 = tf.nn.relu(h1, 'g_rl1')

            ## reshape to conv feature pack and concat with label condition
            h1 = tf.reshape(h1, [self.config.BATCH_SIZE, 7, 7, 64 * 2])
            h1 = self._conv_cond_concat(h1, yb)

            ## first layer deconv
            h2 = self._deconv2d(h1, [self.config.BATCH_SIZE, 14, 14, 128], name='g_dconv0')
            h2 = self._batch_norm(h2, 'g_bn2', train = False)
            h2 = tf.nn.relu(h2, 'g_rl2')
            h2 = self._conv_cond_concat(h2, yb)

            ## output layer: sigmoid to map the data range to [0, 1]
            h3 = self._deconv2d(h2, [self.config.BATCH_SIZE, 28, 28, 1], name='g_dconv1')
            h3 = tf.nn.sigmoid(h3, name='sigmoid')

        return h3


if __name__ == "__main__":
    from config import Config
    class tempConfig(Config):
        BATCH_SIZE = 64
        NUM_CLASSES = 10

    tmp_config = tempConfig()
    model = DCGAN(tmp_config)

    tf.reset_default_graph()
    with tf.device('/gpu:0'):
        z = tf.ones([tmp_config.BATCH_SIZE, 100])
        y = tf.ones([tmp_config.BATCH_SIZE, tmp_config.NUM_CLASSES])
        image = tf.ones([tmp_config.BATCH_SIZE, 28, 28, 1])

        output = model.forward_pass(z, image, y)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        o = sess.run(output)

    for i in o:
        print(i.shape)
