import sys, os
if os.getcwd().endswith("DCGAN"):
    root_dir = os.getcwd()
else:
    root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
import tensorflow as tf
from Model import model_base

class Classifier(model_base.GAN_Base):
    def __init__(self, config):
        super(Classifier, self).__init__(config.DATA_FORMAT, \
                                    config.BATCH_NORM_DECAY, config.BATCH_NORM_EPSILON)
        self.config = config

    def classifier(self, image):
        with tf.variable_scope("classifier"):
            if self.config.CHANNEL == 3:
                image = self._add_noise(image)
                h0 = self._conv2d(image, 64, name = 'd_h0_conv')
                h0 = tf.nn.leaky_relu(h0)

                h1 = self._conv2d(h0, 64 * 2, name = 'd_h1_conv')
                h1 = self._batch_norm_contrib(h1, name = 'd_h1_bn', train = self.config.TRAINING)
                h1 = tf.nn.leaky_relu(h1)

                h2 = self._conv2d(h1, 64 * 4, name = 'd_h2_conv')
                h2 = self._batch_norm_contrib(h2, name = 'd_h2_bn', train = self.config.TRAINING)
                h2 = tf.nn.leaky_relu(h2)

                h3 = self._conv2d(h2, 64 * 8, name = 'd_h3_conv')
                h3 = self._batch_norm_contrib(h3, name = 'd_h3_bn', train = self.config.TRAINING)
                h3 = tf.nn.leaky_relu(h3)

                h4 = tf.reshape(h3, [self.config.BATCH_SIZE, -1])

                h4 = self._linear_fc(h4, self.config.NUM_CLASSES, 'd_h4_lin')
                return h4

            else:
                image = self._add_noise(image)
                # first conv
                h0 = self._conv2d(image, 32, name='d_h0_conv')
                h0 = tf.nn.leaky_relu(h0, alpha=0.2, name='d_leaky0')

                # second conv
                h1 = self._conv2d(h0, 64, name='d_h1_conv')
                h1 = self._batch_norm_contrib(h1, name='d_h1_bn', train = self.config.TRAINING)
                h1 = tf.nn.leaky_relu(h1, alpha=0.2, name='d_leaky1')

                # reshape and concat the label
                h1 = tf.reshape(h1, [self.config.BATCH_SIZE, -1])

                ## fc layer
                h2 = self._linear_fc(h1, 1024, 'd_h2_lin')
                h2 = self._batch_norm_contrib(h2, name='d_h2_bn', train = self.config.TRAINING)
                h2 = tf.nn.leaky_relu(h2, alpha=0.2, name='d_leaky2')

                if self.config.MINIBATCH_DIS:
                    f = self._minibatch_discrimination(h2, 100)
                    h2 = tf.concat([h2, f], 1)

                h3 = self._linear_fc(h2, self.config.NUM_CLASSES, 'd_h3_lin')
                return h3


    def forward_pass(self, image):
        """

        :param z: latent variable
        :param image: input image
        :param label: input label (e.g. mnist)
        :return:
        """
        logits = self.classifier(image)
        return logits


if __name__ == "__main__":
    from config import Config
    class tempConfig(Config):
        NUM_CLASSES = 2
        CHANNEL = 1

    tmp_config = tempConfig()
    tf.reset_default_graph()
    image = tf.ones((64, 64, 64, 3))
    model = Classifier(tmp_config)
    logits = model.forward_pass(image)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logits_o = sess.run(logits)

    print(logits_o.shape)
