import tensorflow as tf
from collections import OrderedDict

class Inference_base(object):
    def __init__(self):
        pass

    def _input_fn(self):
        raise NotImplementedError(
            'input_fn() is implemented in Model sub classes')

    def _build_train_graph(self):
        raise NotImplementedError(
            'loss() is implemented in Model sub classes')

    def _accuracy_metric(self, labels, predictions):
        return tf.metrics.accuracy(labels, predictions)