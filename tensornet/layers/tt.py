# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
from .aux import get_var_wrap
from .quant import *
def tt(inp,
       inp_modes,
       out_modes,
       mat_ranks,
       cores_initializer=tf.initializers.glorot_normal(),
       cores_regularizer=None,
       biases_initializer=tf.zeros_initializer,
       biases_regularizer=None,
       trainable=True,
       cpu_variables=False,
       scope=None):
    """ tt-layer (tt-matrix by full tensor product)
    Args:
        inp: input tensor, float - [batch_size, prod(inp_modes)]
        inp_modes: input tensor modes
        out_modes: output tensor modes
        mat_ranks: tt-matrix ranks
        cores_initializer: cores init function, could be a list of functions for specifying different function for each core
        cores_regularizer: cores regularizer function, could be a list of functions for specifying different function for each core
        biases_initializer: biases init function (if None then no biases will be used)
        biases_regularizer: biases regularizer function        
        trainable: trainable variables flag, bool
        cpu_variables: cpu variables flag, bool
        scope: layer variable scope name, string
    Returns:
        out: output tensor, float - [batch_size, prod(out_modes)]
    """
    with tf.variable_scope(scope):
        dim = inp_modes.size
        
        mat_cores = []
        
        for i in range(dim):
            if type(cores_initializer) == list:
                cinit = cores_initializer[i]
            else:
                cinit = cores_initializer
            
            if type(cores_regularizer) == list:
                creg = cores_regularizer[i]
            else:
                creg = cores_regularizer
                
            mat_cores.append(get_var_wrap('mat_core_%d' % (i + 1),
                                          shape=[out_modes[i] * mat_ranks[i + 1], mat_ranks[i] * inp_modes[i]],
                                          initializer=cinit,
                                          regularizer=creg,
                                          trainable=trainable,
                                          cpu_variable=cpu_variables))
            

        print("INPUT SHAPE ===================> ", tf.shape(inp))
        out = tf.reshape(inp, [-1, np.prod(inp_modes)])
        out = tf.transpose(out, [1, 0])
        
        for i in range(dim):
            out = tf.reshape(out, [mat_ranks[i] * inp_modes[i], -1])
                        
            out = tf.matmul(mat_cores[i], out)
            out = tf.reshape(out, [out_modes[i], -1])
            out = tf.transpose(out, [1, 0])        
        
        if biases_initializer is not None:
            
            biases = get_var_wrap('biases',
                                  shape=[np.prod(out_modes)],
                                  initializer=biases_initializer,
                                  regularizer=biases_regularizer,
                                  trainable=trainable,
                                  cpu_variable=cpu_variables)
                                                                    
            out = tf.add(tf.reshape(out, [-1, np.prod(out_modes)]), biases, name="out")
        else:
            out = tf.reshape(out, [-1, np.prod(out_modes)], name="out")

    return out


def binarized_tt(inp,
       inp_modes,
       out_modes,
       mat_ranks,
       cores_initializer=tf.initializers.glorot_normal(),
       cores_regularizer=None,
       biases_initializer=tf.zeros_initializer,
       biases_regularizer=None,
       trainable=True,
       cpu_variables=False,
       binarize_input = False,
       scope=None):
    """ binarized tt-layer (tt-matrix by binarizing full tensor product)
    Args:
        inp: input tensor, float - [batch_size, prod(inp_modes)]
        inp_modes: input tensor modes
        out_modes: output tensor modes
        mat_ranks: tt-matrix ranks
        cores_initializer: cores init function, could be a list of functions for specifying different function for each core
        cores_regularizer: cores regularizer function, could be a list of functions for specifying different function for each core 
        biases_initializer: biases init function (if None then no biases will be used)
        biases_regularizer: biases regularizer function 
        trainable: trainable variables flag, bool
        cpu_variables: cpu variables flag, bool
        binarize_input: binarize_input flag, bool
        scope: layer variable scope name, string
    Returns:
        out: output tensor, float - [batch_size, prod(out_modes)]
    """
    with tf.variable_scope(scope):
        dim = inp_modes.size
        
        mat_cores = []
        
        for i in range(dim):
            if type(cores_initializer) == list:
                cinit = cores_initializer[i]
            else:
                cinit = cores_initializer
            
            if type(cores_regularizer) == list:
                creg = cores_regularizer[i]
            else:
                creg = cores_regularizer
                
            mat_cores.append(get_var_wrap('mat_core_%d' % (i + 1),
                                          shape=[out_modes[i] * mat_ranks[i + 1], mat_ranks[i] * inp_modes[i]],
                                          initializer=cinit,
                                          regularizer=creg,
                                          trainable=trainable,
                                          cpu_variable=cpu_variables))
            

        
        #out = tf.reshape(inp, [-1, np.prod(inp_modes)])
        # Reconstruct Weights matrices in floating point domain before binarizing
        print("INPUT SHAPE ===================> ", inp.get_shape())
        Wr =  tf.reshape(inp, [-1, np.prod(inp_modes)])#tf.ones_like(tf.reshape(inp, [-1, np.prod(inp_modes)]))
        Wr = tf.transpose(Wr, [1, 0])
        
        for i in range(dim):
            Wr = tf.reshape(Wr, [mat_ranks[i] * inp_modes[i], -1])
                        
            Wr = tf.matmul(mat_cores[i], Wr)
            Wr = tf.reshape(Wr, [out_modes[i], -1])
            Wr = tf.transpose(Wr, [1, 0])   
        tf.clip_by_value(Wr, -1, 1)     
        Wb = binarize(Wr)
        out = tf.reshape(inp, [-1, np.prod(inp_modes)])
        if binarize_input:
            out = binarize(out)
        out = tf.matmul(tf.reshape(Wb, [-1, np.prod(inp_modes)]), out)
        if biases_initializer is not None:
            
            biases = get_var_wrap('biases',
                                  shape=[np.prod(out_modes)],
                                  initializer=biases_initializer,
                                  regularizer=biases_regularizer,
                                  trainable=trainable,
                                  cpu_variable=cpu_variables)
                                                                    
            out = tf.add(tf.reshape(out, [-1, np.prod(out_modes)]), biases, name="out")
        else:
            out = tf.reshape(out, [-1, np.prod(out_modes)], name="out")

    return out