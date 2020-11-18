
import tensorflow as tf
import numpy as np
from layers import *
from collections import OrderedDict

def create_net(x, channels = 10, keep_prob = 0.7, control_bit = 2, layers=3, features_root = 16, filter_size = 6, pool_size = 2):
    weights = []
    biases   = []
    conv2d = []

    dconv2d= OrderedDict()
    conv_out = OrderedDict()
    pools = OrderedDict()
    up_stream= OrderedDict()

    # downstream
    with tf.name_scope("preprocessing"):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x,tf.stack([-1, nx, ny, channels]))
        this_node = x_image
        batch_size = tf.shape(x_image)[0]

    for layer in range(0,layers):
        with tf.name_scope("down_stream_conv_{}".format(str(layer))):
            features = 2**layer*features_root
            stddev = np.sqrt(2/(filter_size**2*features))
            if layer ==0:
                w1 = weight_variable([filter_size, filter_size, channels, features], stddev, name = "w1")
            else:
                w1 = weight_variable([filter_size, filter_size, features//2, features], stddev, name="w1")
            w2 = weight_variable([filter_size,filter_size,features,features], stddev, name="w2")
            b1 = bias_variable_randomNorm([features], name = "b1")
            b2 = bias_variable_randomNorm([features], name = "b2")

            conv1 = conv2d(this_node, w1, b1, keep_prob)
            out_conv1 = tf.nn.relu(conv1)
            conv2 = conv2d(out_conv1, w2, b2, keep_prob)
            conv_out[layer] = tf.nn.relu(conv2)

            weights.append((w1,w2))
            biases.append((b1,b2))
            conv2d.append((conv1,conv2))

            size -= 2*2*(filter_size//2)
            if layer < layers-1:
                pols[layer] = max_pool(conv_out[layer],pool_size)
                this_node = pools[layer]
                size /=pool_size

    this_node = conv_out[layers -1]

    # upstream
    for layer in range(layers -2 , -1, -1):
        with tf.name_scope("up_stream_conv_{}".format(str(layer))):
            features = 2**(layer+1) * features_root
            stddev = np.sqrt(2/(filter_size**2*features))

            wd = weight_variable([pool_size, pool_size, features//2, features], stddev, name = "wd")
            bd = bias_variable_randomNorm([features//2],name="bd")
            out_dconv = tf.nn.relu(deconv2d(this_node, wd, pool_size)+bd)

            out_concat = crop_and_concat(conv_out[layer], out_dconv)
            dconv2d[layer] = out_concat
        
            w1 = weight_variable([filter_size, filter_size, features, features//2], stddev, name = "w1")
            w2 = weight_variable([filter_size, filter_size, features//2, features//2], stddev, name = "w2")
            b1 = bias_variable_randomNorm([features // 2], name="b1")
            b2 = bias_variable_randomNorm([features // 2], name="b2")

            conv1 = conv2d(out_concat, w1, b1, keep_prob)
            out_conv = tf.nn.relu(conv1)
            conv2 = conv2d(out_conv, w2, b2, keep_prob)
            this_node = tf.nn.relu(conv2)
            up_stream[layer] = this_node
            weights.append((w1,w2))
            biases.append((b1,b2))
            conv2d.append((conv1,conv2))

            size *= pool_size
            size -= 2*2*(filter_size//2)

    with tf.name_scope("output_map"):
        weights = weight_variable([1,1,features_root, control_bit],stddev)
        bias = bias_variable_randomNorm([control_bit],name="bias")
        conv = conv2d(this_node, weight, bias, tf.constant(1.0))
        output_map = tf.nn.relu(conv)
        up_stream["out"]=output_map

    variables = []
    for w1, w2 in weights:
        variables.append(w1)
        variables.append(w2)
    for b1, b2 in biases:
        variables.append(b1)
        variables.append(b2)

    return output_map, variables, int(in_size - size)


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

    def get_cost(self, logits):
        with tf.name_scope("cost"):
            flat_logits = tf.reshape(logits, [-1, self.control_bit])
            flat_labels = tf.reshape(self.y, [-1, self.control_bit])
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = flat_logits, labels=flat_labels))
        return loss

    def predict(self, model_path, x_test):
        init=tf.global_variables_initializer()
	
	"""
        with tf.Session() as sess:
            sess.run(init)

            self.restore(sess,model_path)

            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})
	"""
        return prediction

#    def save(self, sess, model_path):
#	saver = tf.train.Checkpoint(self.net)
#	save_path = saver.save(sess, model_path)
#	return save_path
#
#    def restore(self, sess, model_path):
#	saver= tf.train.Saver()
#	saver.restore(sess, model_path)


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
	

    def train(self, dataWrapper,  training_iters = 10, epochs = 100, dropout = 0.5, display_step=1):

	for epoch in range(epochs):
		total_loss =0
		for step in range((epoch*training_iters), ((epoch +1)*training_iters)):
			batch_x, batch_y = dataWrapper.generate(self.batch_size)

			# Run optimization op (backprop)
                	_, loss, lr, gradients = sess.run(
                	    (self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node),
                	    feed_dict={self.net.x: batch_x,
                	               self.net.y: util.crop_to_shape(batch_y, pred_shape),
                	               self.net.keep_prob: dropout})

                	total_loss += loss
		
	return total_loss


def _update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step + 1)))) + (gradients[i] / (step + 1))

    return avg_gradients

	
