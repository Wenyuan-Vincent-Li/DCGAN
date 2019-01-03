## import module
import os, sys
import tensorflow as tf
import scipy.io
if os.getcwd().endswith("DCGAN"):
    root_dir = os.getcwd()
else:
    root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)

class CedarsDataset(object):
    """
    cedars dataset with label{0: no cancer 1: cancer in the image}
    """
    def __init__(self, data_dir, config, file_mid_name = "_Set_%d_Por_%d"%(0, int(0.4 * 100)), subset= None, use_augmentation=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_augmentation = use_augmentation
        self.config = config
        self.file_mid_name = file_mid_name
        self.y_dim = 2 # two classes: cancer, no-cancer

    def get_filenames(self):
        return [os.path.join(self.data_dir, 'Tfrecord/Train'\
                             + self.file_mid_name + '_Lab' + '.tfrecords'),
                os.path.join(self.data_dir, 'Tfrecord/Train'\
                             + self.file_mid_name + '_Unl' + '.tfrecords'),
                os.path.join(self.data_dir, 'Tfrecord/Test'\
                             + self.file_mid_name + '.tfrecords')]


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
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64)
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        label = tf.cast(features['label'], tf.int32)
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        image = tf.cast(tf.reshape(image, [height, width, 3]), tf.float32)
        image.set_shape([self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3])

        ## pre-processing data do augmentation
        if self.use_augmentation:
            image, label = self.pre_processing(image, label)
        return image, label

    def pre_processing(self, image, label):
        image = tf.div(image, 127.5) - 1 ## map pixel value to {-1, 1}
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, k = tf.random_uniform(shape = [],\
                                           maxval = 3, dtype=tf.int32))
        # one-hot encoding for the label
        label = tf.one_hot(label, depth = self.y_dim)
        ## TODO random flip label
        return image, label

    def shuffle_and_repeat(self, dataset, repeat = 1):
        dataset = dataset.shuffle(buffer_size= \
                                      self.config.MIN_QUEUE_EXAMPLES + \
                                      30 * self.config.BATCH_SIZE, \
                                  )
        dataset = dataset.repeat(count = repeat)
        return dataset

    def batch(self, dataset):
        dataset = dataset.batch(batch_size=self.config.BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=self.config.BATCH_SIZE)
        return dataset

    def inputpipline_singleset(self):
        ## Inputpipline that used for training
        ## Return: init_op_train (list); init_op_val (list); lab_input (tensor);
        ## lab_output (tensor); train_unl_input (tensor).

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
        image_batch, label_batch = iterator.get_next()
        image_batch.set_shape([self.config.BATCH_SIZE] + self.config.IMAGE_DIM)
        label_batch.set_shape([self.config.BATCH_SIZE, self.y_dim])

        return image_batch, label_batch, init_op

if __name__ == "__main__":
    from config import Config

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all debugging logs

    class Temp_Config(Config):
        BATCH_SIZE = 64
        REPEAT = 1
        IMAGE_HEIGHT = 64
        IMAGE_WIDTH = 64
        CHANNEL = 3

        # Total number of image: 62073

    tmp_config = Temp_Config()
    data_dir = os.path.join(root_dir, "Dataset/prostate")
    test_folder = 0
    label_portion = 0.4
    file_mid_name = "_Set_%d_Por_%d"%(test_folder, int(label_portion * 100))
    dataset = CedarsDataset(data_dir, tmp_config, file_mid_name, 'Train', True)
    image_batch, label_batch, init_op = dataset.inputpipline_singelset()

    num_batch = 0
    with tf.Session() as sess:
        sess.run(init_op)
        while True:
            try:
                image_batch_o, label_batch_o = \
                    sess.run([image_batch, label_batch])
                num_batch += 1
            except tf.errors.OutOfRangeError:
                train_batch_shape = image_batch_o.shape
                break;

    ## Print the Statistics
    print("DATA STATISTICS: \n", \
          "NUM_BATCH_PER_EPOCH: %d \n" % num_batch, \
          "BATCH_SIZE_TRAIN: ", train_batch_shape, '\n')
