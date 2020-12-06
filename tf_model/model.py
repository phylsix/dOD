
import tensorflow as tf
import numpy as np
from layers import *
from collections import OrderedDict

class mynet(object):
	def __init__(self, input_shape, keep_prob, control_bit = 2):
		"""
		The input_shape = [widht, height, channels]
		"""
		
		self.depth = 3
		self.control_bit = control_bit
		self.nx = input_shape[0]
		self.ny = input_shape[1]
		self.channels=input_shape[2]
		self.keep_prob = keep_prob
		self.ishape = input_shape
		self.layers_downstream = {}
		self.layers_upstream = {}


		self.net = self.buildNet()
		self.net.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam())
		
	def buildNet(self):
		inputs = tf.keras.Input(shape = self.ishape, name="inputs")
		x = inputs
		for idepth in range(0, self.depth):
			x = sequentialConv2DLayer(kernal_shape=(3,3,3), nlayer = 2, padding="same", stride=(1,1), drop_rate=self.keep_prob, input_shape=self.ishape)(x)
			self.layers_downstream[idepth] = x
			x = tf.keras.layers.MaxPooling2D((2,2))(x)

		for idepth in range(0, self.depth):
			x = cropConcat()(self.layers_downstream[self.depth-idepth-1],x)
			x = sequentialConv2DTransposeLayer(kernal_shape=(3,3,3), nlayer = 2, padding="same", stride=(1,1), drop_rate=self.keep_prob, input_shape=self.ishape)(x)
			self.layers_upstream[self.depth-idepth-1] = x
			x = tf.keras.layers.MaxPooling2D((2,2))(x)
			
		model = tf.keras.Model(inputs, x, name = "test")
		return model
