import tensorflow as tf


def augment_v1(image, label):
    image_shape = image.shape
    
    # Flip horizontaly
    image = tf.image.flip_left_right(image)
    
    image = tf.image.resize_with_crop_or_pad(
        image, image_shape[0] + 6, image_shape[1] + 6)
    # Random crop back to the original size.
    image = tf.image.random_crop(
        image, size=image_shape)
    # We don't rescale images, EffNet does! So clip to 255
    image = tf.clip_by_value(image, 0, 255)

    return image, label