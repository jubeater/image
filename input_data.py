
import tensorflow as tf
import numpy as np
import os
import pathlib
def read(file):
    f = open(file, 'r', encoding="utf-8")
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels


def read_images_from_disk(input_queue,img_height, img_width):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    return example, label


def get_batch(file, img_height, img_width, batch_size):
    image_list, label_list = read(file)
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
    input_queue = tf.train.slice_input_producer([images, labels]
                                                ,shuffle=True)
    image, label = read_images_from_disk(input_queue,  img_height, img_width)
    image = tf.image.resize_images(image, [img_height, img_width])
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size)
    n_classes = 10
    label_batch = tf.one_hot(label_batch, depth=n_classes)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])

    return image_batch, label_batch


if __name__ == "__main__":
    a, b = read('.//data//data//filelist')
    for x in range(len(a)):
        print(a[x] + ' ' + str(b[x]))