
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class data_wrapper(object):
	
	"""
	a base class for wrapper the samples into a generator feeding to nn for trainning
	FUNC:
		pop: (vritual) generate one sample
		generate: generate a batch of samples
		show: current draw the sample by the matplotlib
	"""
	def __init__(self,name, pixel_width, pixel_height, FPS=10):
		"""
		FPS (frames per sample): number of frames for each sample = pop()
		"""
		self.name = name 
		self.p_w = pixel_width
		self.p_h = pixel_height
		self.fps = FPS
	
	def generate(self, size):
		""" 
		The virutal function pop here suppose to be implemented in child class with the output:
		data, sig, logist = self.pop()
		where 'data' is the sig+bkg	
		"""
		data = []
		sig = []
		for i in range(size):
			d, s = self.pop()
			data.append(d)
			sig.append(s)
		return np.array(data), np.array(sig)
	
	def update(self, i, data, fg, ax, ntrail):
		label = 'Frame step: {0}/{1}'.format(i+1, ntrail*self.fps)
		fg.set_data(data[i])
		ax.set_xlabel(label)
		return fg, ax

	def show(self):
		ntrails = int(40/self.fps);
		if ntrails*self.fps < 40 : ntrails+=1
		data = []
		mask = []
		for j in range(ntrails):
			d, sig = self.pop()
			for i in range(self.fps):
				mask.append(sig[i])
				data.append(d[i])
		fig,(ax1,ax2) = plt.subplots(1,2, sharey=True)
		print(data[0].shape)
		fg1 = ax1.imshow(data[0])
		fg2 = ax2.imshow(mask[0])
		ax1.set_title("data")
		ax2.set_title("mask")
		for i in range(1, ntrails*self.fps):
			self.update(i, data, fg1, ax1, ntrails)
			self.update(i, mask, fg2, ax2, ntrails)
			plt.pause(2/self.fps/ntrails)
		plt.show()
			
