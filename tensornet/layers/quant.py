import tensorflow.compat.v1 as tf
from .aux_win import get_var_wrap

#https://github.com/ivanbergonzani/binarized-neural-network/blob/master/layers.py
def ap2(x):
	return tf.sign(x) * tf.pow(2.0, tf.round(tf.log(tf.abs(x)) / tf.log(2.0)))
    
def shift_batch_norm(x, 
        training=True,
        momentum=0.99, 
        epsilon=1e-8, 
        scope=None):
	
	xshape = x.get_shape()[1:]
	
	with tf.variable_scope(scope):
		gamma = get_var_wrap('gamma', xshape, initializer=tf.ones_initializer, trainable=True)
		beta  = get_var_wrap('beta', xshape, initializer=tf.zeros_initializer, trainable=True)
		
		mov_avg = get_var_wrap('mov_avg', xshape, initializer=tf.zeros_initializer, trainable=False)
		mov_var = get_var_wrap('mov_std', xshape, initializer=tf.ones_initializer, trainable=False)
		
		def training_xdot():
			avg = tf.reduce_mean(x, axis=0)							# feature means
			cx = x - avg											# centered input
			var = tf.reduce_mean(tf.multiply(cx, ap2(cx)), axis=0)	# apx variance
			
			# updating ops. for moving average and moving variance used at inference time
			avg_update = tf.assign(mov_avg, momentum * mov_avg + (1.0 - momentum) * avg)
			var_update = tf.assign(mov_var, momentum * mov_var + (1.0 - momentum) * var)
			
			with tf.control_dependencies([avg_update, var_update]):
				return cx / ap2(tf.sqrt(var + epsilon))				# normalized input
			
		def inference_xdot():
			return (x - mov_avg) / ap2(tf.sqrt(mov_var + epsilon))
		
		xdot = tf.cond(training, training_xdot, inference_xdot)
		out = tf.multiply(ap2(gamma), xdot) + beta					# scale and shift input distribution
	
	return out
#https://github.com/uranusx86/BinaryNet-on-tensorflow/blob/master/binary_layer.py
def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.)/2., 0, 1)

def binarize(x):
	# we also have to reassign the sign gradient otherwise it will be almost everywhere equal to zero
	# using the straight through estimator
	with tf.get_default_graph().gradient_override_map({'Sign': 'Identity'}):
		#return tf.sign(x)				#	<-- wrong sign doesn't return +1 for zero
		return tf.sign(tf.sign(x)+1e-8) #	<-- this should be ok, ugly but okay
def binary_tanh_unit(x):
	with tf.get_default_graph().gradient_override_map({'Round': 'Identity'}):
		return 2. * tf.round(hard_sigmoid(x)) - 1.

def binary_sigmoid_unit(x):
	with tf.get_default_graph().gradient_override_map({'Round': 'Identity'}):
		return tf.round(hard_sigmoid(x))

def quant_2bits_binary_tanh_unit(inp,
           scope=None):
    """ binary_tanh_unit
    Args:  
        inp : input tensor, float - [batch_size, prod(inp_modes)]
        scope: layer variable scope name, string
    Returns:
        out: output tensor, float - [batch_size, out_size]
    """

    with tf.variable_scope(scope):
        inp_bin = binary_tanh_unit(inp)
        return inp_bin

def quant_2bits_binary_sigmoid_unit(inp,
           scope=None):
    """ binary_sigmoid_unit
    Args:  
        inp : input tensor, float - [batch_size, prod(inp_modes)]
        scope: layer variable scope name, string
    Returns:
        out: output tensor, float - [batch_size, out_size]
    """
    with tf.variable_scope(scope):
        inp_bin = binary_sigmoid_unit(inp)
        return inp_bin

# def quant_8bits(inp,
#            out_size,
#            weights_initializer=tf.initializers.glorot_normal(),
#            weights_regularizer=None,
#            biases_initializer=tf.zeros_initializer,
#            biases_regularizer=None,
#            trainable=False,
#            cpu_variables=False,
#            scope=None):
#     """ quantization layer
#     Args:
#         inp: input tensor, float - [batch_size, inp_size]        
#         out_size: layer units count, int
#         weights_initializer: weights init function
#         weights_regularizer: weights regularizer function
#         biases_initializer: biases init function (if None then no biases will be used)
#         biases_regularizer: biases regularizer function        
#         trainable: trainable variables flag, bool
#         cpu_variables: cpu variables flag, bool
#         scope: layer variable scope name, string
#     Returns:
#         out: output tensor, float - [batch_size, out_size]
#     """
#     with tf.variable_scope(scope):
#         tf.quantization.quantize()