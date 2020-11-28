
import tensorflow as tf
import numpy as np
from layers import *
from collections import OrderedDict

class mynet(object):
	def __init__(self, input_shape, keep_prob, control_bit = 2):
		"""
		The input_shape = [widht, height, channels]
		"""
		self.control_bit = control_bit
		self.nx = input_shape[0]
		self.ny = input_shape[1]
		self.channels=input_shape[2]
		self.keep_prob = keep_prob

		self.layers = 3
		
		self.weights = []	
		self.biases = []	
		self.layer_conv2d = []
		
		self.layer_dconv2d = OrderedDict()
		self.conv_out = OrderedDict()
		self.layer_pools = OrderedDict()	
		self.up_stream = OrderedDict()	
		self.variable_list = []
		self.net = tf.function(func = self.create_net,input_signature = [tf.TensorSpec(shape=[self.nx, self.ny, self.channels], dtype=tf.float32)])	
		self.step_count=0

	def predict(self, inputs):
		return self.net.get_concrete_function(inputs)


	def create_net(self, x):
		features_root = 16
		filter_size = 3
		pool_size = 2
		in_size = 1000
		size = in_size
		with tf.name_scope("preprocessing"):
			x_image = tf.reshape(x,shape=[1, self.nx, self.ny, self.channels])
			this_node = x_image
			#x_image = tf.reshape(x,tf.stack([-1, self.nx, self.ny, self.channels]))
			#this_node = x_image
			batch_size = tf.shape(x_image)[0]
	
		for layer in range(0,self.layers):
		    with tf.name_scope("down_stream_conv_{}".format(str(layer))):
		        features = 2**layer*features_root
		        stddev = np.sqrt(2/(filter_size**2*features))
		        if layer ==0:
		            w1 = weight_variable([filter_size, filter_size, self.channels, features], stddev, name = "w1")
		        else:
		            w1 = weight_variable([filter_size, filter_size, features//2, features], stddev, name="w1")
		        w2 = weight_variable([filter_size,filter_size,features,features], stddev, name="w2")
		        b1 = bias_variable_randomNorm([features], name = "b1")
		        b2 = bias_variable_randomNorm([features], name = "b2")
		
		        conv1 = conv2d(this_node, w1, b1, self.keep_prob)
		        out_conv1 = tf.nn.relu(conv1)
		        conv2 = conv2d(out_conv1, w2, b2, self.keep_prob)
		        self.conv_out[layer] = tf.nn.relu(conv2)
		
		        self.weights.append((w1,w2))
		        self.biases.append((b1,b2))
		        self.layer_conv2d.append((conv1,conv2))
		
		        size -= 2*2*(filter_size//2)
		        if layer < self.layers-1:
		            self.layer_pools[layer] = max_pool(self.conv_out[layer],pool_size)
		            this_node = self.layer_pools[layer]
		            size /=pool_size
		
		this_node = self.conv_out[self.layers -1]
		
		# upstream
		for layer in range(self.layers -2 , -1, -1):
		    with tf.name_scope("up_stream_conv_{}".format(str(layer))):
		        features = 2**(layer+1) * features_root
		        stddev = np.sqrt(2/(filter_size**2*features))
		
		        wd = weight_variable([pool_size, pool_size, features//2, features], stddev, name = "wd")
		        bd = bias_variable_randomNorm([features//2],name="bd")
		        out_dconv = tf.nn.relu(deconv2d(this_node, wd, pool_size)+bd)
		
		        out_concat = crop_and_concat(self.conv_out[layer], out_dconv)
		        self.layer_dconv2d[layer] = out_concat
		    
		        w1 = weight_variable([filter_size, filter_size, features, features//2], stddev, name = "w1")
		        w2 = weight_variable([filter_size, filter_size, features//2, features//2], stddev, name = "w2")
		        b1 = bias_variable_randomNorm([features // 2], name="b1")
		        b2 = bias_variable_randomNorm([features // 2], name="b2")
		
		        conv1 = conv2d(out_concat, w1, b1, self.keep_prob)
		        out_conv = tf.nn.relu(conv1)
		        conv2 = conv2d(out_conv, w2, b2, self.keep_prob)
		        this_node = tf.nn.relu(conv2)
		        self.up_stream[layer] = this_node
		        self.weights.append((w1,w2))
		        self.biases.append((b1,b2))
		        self.layer_conv2d.append((conv1,conv2))
		
		        size *= pool_size
		        size -= 2*2*(filter_size//2)
		
		with tf.name_scope("output_map"):
		    weights = weight_variable([1,1,features_root, self.control_bit],stddev)
		    bias = bias_variable_randomNorm([self.control_bit],name="bias")
		    conv = conv2d(this_node, weights, bias, tf.constant(1.0))
		    output_map = tf.nn.relu(conv)
		    self.up_stream["out"]=output_map
		
		for w1, w2 in self.weights:
		    self.variable_list.append(w1)
		    self.variable_list.append(w2)
		for b1, b2 in self.biases:
		    self.variable_list.append(b1)
		    self.variable_list.append(b2)
		
		return output_map

	@tf.function
	def cost(self, output, logits):
		with tf.name_scope("cost"):
			flat_logits = tf.reshape(logits, [-1, self.control_bit])
			flat_labels = tf.reshape(output, [-1, self.control_bit])
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = flat_logits, labels=flat_labels))
		return loss


class Unet(object):
    def __init__(self, channels, control_bit, cost="cross_entropy", **kwargs):
        self.control_bit = control_bit
        self.channels = channels

        self.x = tf.placeholder("float", shape=[None,None,None,channels],name="x")
        self.y = tf.placeholder("float", shape=[None,None,None,channels],name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_probability")

        logits, self.variables, self.offset = create_net(self.x, self.keep_prob, channels, control_bit, **kwargs)

        self.cost = self.get_cost(logits)
        self.gradients_node = tf.gradients(self.cost, self.variables)

        with tf.name_scope("cross_entropy"):
            self.cross_entropy = cross_entropy(tf.reshape(self.y, [-1, n_class]),
                    tf.reshape(pixel_wise_softmax(logits), [-1, n_class]))

        with tf.name_scope("results"):
            self.predicter = pixel_wise_softmax(logits)
            self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


class Trainer(object):
	def __init__(self, net, batch_size=1, verification_batch_size = 4, norm_grads = False,  opt_kwargs={}):
		self.net = net
		self.batch_size = batch_size
		self.verification_batch_size = verification_batch_size
		self.norm_grads = norm_grads
		self.opt_kwargs = opt_kwargs
		
		self.optimizer = self.get_optimizer()

	def get_optimizer(self):
		learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
		self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")
		
		optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_node)
		
		return optimizer
	

