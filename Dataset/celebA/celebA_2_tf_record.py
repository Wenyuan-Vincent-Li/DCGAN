import os, sys
import numpy as np
from glob import glob
import tensorflow as tf
import imageio
root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(root_dir)

data_dir = './celebA'
file_path = os.path.join(data_dir, '*.jpg') # celebA image size: (218, 178, 3)

file_names = glob(file_path)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

with tf.python_io.TFRecordWriter("./Tfrecord/celebA.tfrecords") as record_writer:
    for idx, file_name in enumerate(file_names):
        image = imageio.imread(file_name).astype(np.uint8)
        image = np.array(image)
        example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': _bytes_feature(image.tobytes())
                    }))
        record_writer.write(example.SerializeToString())
        if idx % 5000 == 0:
            print("Finished Processing %d/%d image!"%(idx, len(file_names)))