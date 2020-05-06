# import tensorflow as tf
import tensorflow.compat.v1 as tf

def get_var_wrap(name,
                 shape = None,
                 initializer = None,
                 regularizer = None,
                 trainable = None,
                 cpu_variable = None):
    if cpu_variable:
        with tf.device('/cpu:0'):
            return tf.get_variable(name,
                                   shape=shape,
                                   initializer=initializer,
                                   regularizer=regularizer,
                                   trainable=trainable)
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           regularizer=regularizer,
                           trainable=trainable)
