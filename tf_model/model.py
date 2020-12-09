
import tensorflow as tf
import numpy as np
from layers import *
from collections import OrderedDict

class mynet(object):
	def __init__(self, input_shape, drop_rate,padding, control_bit = 2):
		"""
		The input_shape = [widht, height, channels]
		"""
		
		self.depth = 5
		self.control_bit = control_bit
		self.nx = input_shape[0]
		self.ny = input_shape[1]
		self.channels=input_shape[2]
		self.drop_rate = drop_rate
		self.ishape = input_shape
		self.layers_downstream = {}
		self.layers_upstream = {}
		self.root_feature = 64
		self.net = None
		self.nclass = 1
		self.padding = padding

		
	def buildNet(self, verbose = False):
		inputs = tf.keras.Input(shape = self.ishape, name="inputs", dtype=tf.float32)
		x = inputs
		feature =0
		for idepth in range(0, self.depth):
			feature = 2**idepth*self.root_feature
			if idepth == 0 :
				x = sequentialConv2DLayer( kernal_shape=(3,3,self.root_feature), nlayer = 2, padding=self.padding, stride=(1,1), drop_rate=self.drop_rate, input_shape=self.ishape)(x)
			else:
				x = sequentialConv2DLayer(kernal_shape=(3,3,feature), nlayer = 2, padding=self.padding, stride=(1,1), drop_rate=self.drop_rate)(x)
			self.layers_downstream[idepth] = x
			if verbose: print("--: downstream layer",idepth," shape: ", x.shape)
			if idepth < self.depth-1: x = tf.keras.layers.MaxPooling2D((2,2))(x)
			if verbose: print("--: maxpool layer",idepth," shape: ", x.shape)

		for idepth in range(0, self.depth-1):
			feature = feature//2
			x = sequentialConv2DTransposeLayer(kernal_shape=(2,2,feature), nlayer = 1, padding=self.padding, stride=(2,2), drop_rate=0)(x)
			if verbose: print("--: upsampling",self.depth-2-idepth," shape ", x.shape)
			if verbose: print("--: crop and concat",self.layers_downstream[self.depth-idepth-2].shape," with ", self.depth-2-idepth," shape ", x.shape)
			x = cropConcat()(self.layers_downstream[self.depth-idepth-2],x)
			x = sequentialConv2DLayer(kernal_shape=(3,3, feature), nlayer = 2, padding=self.padding, stride=(1,1), drop_rate=self.drop_rate)(x)
			if verbose: print("--: upstream layer",self.depth-2-idepth," shape ", x.shape)
			self.layers_upstream[self.depth-idepth-1] = x

		output = tf.keras.layers.Conv2D(self.nclass, (1,1), padding = self.padding, activation = 'relu')(x)
		if verbose: print("--: output shape ", output.shape)
			
		model = tf.keras.Model(inputs, output, name = "test")
		self.net = model
		self.net.compile(loss="binary_crossentropy", optimizer = "Adam", metrics=['binary_crossentropy','binary_accuracy'])
		return 
