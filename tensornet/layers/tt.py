# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
from .aux_win import get_var_wrap
from .quant import *
import sys
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
            


        # inp = tf.Print(inp, [inp], "inp = ")
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
        # out = tf.Print(out, [out], "out = ")

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
            

        
        # out = tf.reshape(inp, [np.prod(inp_modes), -1])
        # Reconstruct Weights matrices in floating point domain before binarizing
        # inp = tf.Print(inp, [tf.shape(inp)], message="inp=")
        out  = tf.reshape(inp, [np.prod(inp_modes), -1])
        # out  = tf.Print(out, [tf.shape(out)], message="out=")
        # out  = tf.transpose(out, [1, 0])
        
        if binarize_input:
            out = binarize(out)
        for i in range(dim):
            out = tf.reshape(out, [mat_ranks[i] * inp_modes[i], -1])
            # out = tf.Print(out, [tf.shape(out)], message="out=")
            # core = tf.Print(mat_cores[i], [tf.shape(mat_cores[i])], message="Mat_core_{}=".format(i))
            coreb = mat_cores[i]
            #tf.clip_by_value(core, -1, 1)     
            #coreb = binarize(core)
            out = tf.matmul(coreb, out)
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
        # out = tf.Print(out, [tf.shape(out)], message="out=")
    return out

def bnn_tt(inp,
       inp_modes,
       out_modes,
       mat_ranks,
       cores_initializer=tf.initializers.glorot_normal(),
       cores_regularizer=None,
       biases_initializer=tf.zeros_initializer,
       biases_regularizer=None,
       trainable=True,
       binarize_input = False,
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
            mat_cores[-1] = binarize(mat_cores[-1])
            # mat_cores[-1] = tf.Print(mat_cores[-1], [tf.unique(tf.reshape(mat_cores[-1], [-1]))[0]], "UNIQUE = ")
        # inp = tf.Print(inp, [inp], "inp = ")
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
        # out = tf.Print(out, [out], "out = ")

    return out

def regular_bnn_layer(inp,
       weights_initializer=tf.initializers.glorot_normal(),
       weights_regularizer=None,
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
        inp_shape = inp.get_shape()
        Weights = get_var_wrap('weights',
                                shape=[inp_shape[-1], 4096],
                                initializer=weights_initializer,
                                regularizer=weights_regularizer,
                                trainable=trainable,
                                cpu_variable=cpu_variables)
        
      
        tf.clip_by_value(Weights, -1, 1)     
        Wb = binarize(Weights)
        input_ = inp
        if binarize_input:
            input_= binarize(input_)
        out = tf.matmul(input_, Wb)

        if biases_initializer is not None:
            biases = get_var_wrap('biases',
                                  shape=[4096],
                                  initializer=biases_initializer,
                                  regularizer=biases_regularizer,
                                  trainable=trainable,
                                  cpu_variable=cpu_variables)
                                                                    
            out = tf.add(out, biases, name="out")
    return out

def tt_full_bnn(inp,
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
            


        # inp = tf.Print(inp, [inp], "inp = ")
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
        # out = tf.Print(out, [out], "out = ")

    return out