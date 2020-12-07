
import tensorflow as tf

# the shape of the tensor is [batch, width, height, channels]

class sequentialConv2DLayer(tf.keras.layers.Layer):
	def __init__(self,  kernal_shape, nlayer, padding, stride, drop_rate, **kwargs):
		super(sequentialConv2DLayer, self).__init__(**kwargs)
		# shape = [width, height, channels]
		self.ishape = None
		self.nlayer = nlayer
		self.drop_rate = drop_rate
		self.layers_conv2d = []
		self.layers_dropout = []
		self.kernal_shape = kernal_shape

		if "input_shape" in kwargs:
			self.ishape = kwargs["input_shape"]

		if self.ishape is None:	
			self.layers_conv2d.append(tf.keras.layers.Conv2D(kernal_shape[2], kernal_shape[0:2], strides=stride, padding = padding, activation = 'relu'))
		else :
			self.layers_conv2d.append(tf.keras.layers.Conv2D(kernal_shape[2], kernal_shape[0:2], strides=stride, padding = padding, input_shape = self.ishape, activation = 'relu'))
		if self.drop_rate > 0 : 
			self.layers_dropout.append(tf.keras.layers.Dropout(self.drop_rate))
	
		for ilyer in range(1,self.nlayer):
			self.layers_conv2d.append(tf.keras.layers.Conv2D(self.ishape[2], kernal_shape[0:2], strides=stride, padding = padding, activation = 'relu'))
			if self.drop_rate > 0 : 
				self.layers_dropout.append(tf.keras.layers.Dropout(self.drop_rate))
	
	def call(self, inputs, training=None, **kwargs):
		x = inputs
		for ilyer in range(0,self.nlayer):
			x = self.layers_conv2d[ilyer](x)
			if self.drop_rate > 0 : 
				x = self.layers_dropout[ilyer](x)
		return x

class sequentialConv2DTransposeLayer(tf.keras.layers.Layer):
	def __init__(self,  kernal_shape, nlayer, padding, stride, drop_rate, **kwargs):
		super(sequentialConv2DTransposeLayer, self).__init__(**kwargs)
		# shape = [width, height, channels]
		self.ishape = None
		self.nlayer = nlayer
		self.drop_rate = drop_rate
		self.layers_conv2dT = []
		self.layers_dropout = []
		self.kernal_shape = kernal_shape

		if "input_shape" in kwargs:
			self.ishape = kwargs["input_shape"]

		if self.ishape is None:	
			self.layers_conv2dT.append(tf.keras.layers.Conv2DTranspose(kernal_shape[2], kernal_shape[0:2], strides=stride, padding = padding, activation = 'relu'))
		else :
			self.layers_conv2dT.append(tf.keras.layers.Conv2DTranspose(kernal_shape[2], kernal_shape[0:2], strides=stride, padding = padding, input_shape = self.ishape, activation = 'relu'))
		self.layers_dropout.append(tf.keras.layers.Dropout(self.drop_rate))
	
		for ilyer in range(1,self.nlayer):
			self.layers_conv2dT.append(tf.keras.layers.Conv2DTranspose(self.ishape[2], kernal_shape[0:2], strides=stride, padding = padding, activation = 'relu'))
			self.layers_dropout.append(tf.keras.layers.Dropout(self.drop_rate))
	
	def call(self, inputs, training=None, **kwargs):
		x = inputs
		for ilyer in range(0,self.nlayer):
			x = self.layers_conv2dT[ilyer](x)
			x = self.layers_dropout[ilyer](x)
		return x

class cropConcat(tf.keras.layers.Layer):
	def call(self, x1, x2):
		x1_shape = tf.shape(x1)
		x2_shape = tf.shape(x2)
		height_diff = (x1_shape[1] - x2_shape[1]) // 2
		width_diff = (x1_shape[2] - x2_shape[2]) // 2
		
		down_layer_cropped = x1[:,
			height_diff: (x1_shape[1] - height_diff),
			width_diff: (x1_shape[2] - width_diff),
			:]
		
		x = tf.concat([down_layer_cropped, x2], axis=-1)
		return x
