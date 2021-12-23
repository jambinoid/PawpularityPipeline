import tensorflow as tf


def resize(image, height, width):
    image = tf.cast(image, tf.float32)
    return tf.image.resize(image, (height, width))