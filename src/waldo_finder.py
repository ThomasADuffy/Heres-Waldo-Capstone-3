import cv2
import imutils
import os
import sys
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf

SRCpath = os.path.split(os.path.abspath(__file__))[0]
ROOTpath = os.path.split(SRCpath)[0]
IMGSpath = os.path.join(ROOTpath,'imgs')
GIFpath = os.path.join(IMGSpath,'gif')
IMGTESTpath = os.path.join(IMGSpath,'test.jpg')
MODELpath = os.path.join(ROOTpath,'model')
FOUNDWALDOpath=os.path.join(IMGSpath,'waldo_found')


class WaldoFinder():

	def __init__(self,imgpath):

		self.img = cv2.imread(f'{imgpath}')
		self.img_resized = None
		self.model = None

	def rescale_image(self, scale):
		w = int(self.img.shape[1] / scale)
		self.img_resized = imutils.resize(self.img, width=w)

	def sliding_window(self, image, stepSize, windowSize, resized):
		if resized:
		# slide a window across the image
			for y in range(0, self.img_resized.shape[0], stepSize):
				for x in range(0, self.img_resized.shape[1], stepSize):
					# yield the current window
					yield (x, y, self.img_resized[y:y + windowSize[1], x:x + windowSize[0]])
		else:
		# slide a window across the image
			for y in range(0, self.img.shape[0], stepSize):
				for x in range(0, self.img.shape[1], stepSize):
					# yield the current window
					yield (x, y, self.img[y:y + windowSize[1], x:x + windowSize[0]])

	

	def test_sliding_window(self,windowsize,stepsize, savedir=None, resized=None):
		p=0
		(winW, winH) = windowsize
		if resized:
			img = self.img_resized
		else:
			img = self.img
		for (x, y, window) in self.sliding_window(img, stepSize=stepsize, windowSize=(winW, winH),resized=resized):
				# if the window does not meet our desired window size, ignore it
				if window.shape[0] != winH or window.shape[1] != winW:
					continue
				clone = img.copy()
				cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 0), 2)
				cv2.imshow("Window", clone)
				cv2.waitKey(1)
				if savedir:
					cv2.imwrite(savedir+f'/{p}.jpg',clone)
				p+=1

	def load_model(self,modelpath):
		self.model=load_model(modelpath)

	def find_waldo(self,stepsize, savedir=None, resized=None):
		if self.model:
			pass
		else:
			return 'No model loaded! please run load_model.'
		(winW, winH) = (64,64)
		if resized:
			img = self.img_resized
		else:
			img = self.img
		p=0
		for (x, y, window) in self.sliding_window(img, stepSize=stepsize, windowSize=(winW, winH),resized=resized):
				# if the window does not meet our desired window size, ignore it
				if window.shape[0] != winH or window.shape[1] != winW:
					continue
				prediction=self.model.predict(np.array([window],dtype='float64'))
				clone = img.copy()
				if savedir:
					if prediction[0]>.5:
						cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
						cv2.putText(clone, text=f'Waldo!{round(prediction[0],3)}', org=(x,y),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,255,0))
						cv2.imshow("Window", clone)
						cv2.imwrite(savedir+f'/{p}.jpg',clone)
						print(f'Found Waldo at {x},{y}')
						p+=1
					else:
						cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 0), 2)
						cv2.imshow("Window", clone)
				cv2.waitKey(1)


if __name__ == "__main__":
	waldofind = WaldoFinder(IMGTESTpath)
	# waldofind.rescale_image(2.5)
	# waldofind.test_sliding_window((64,64),128,resized=True,savedir=GIFpath)
	waldofind.load_model(os.path.join(MODELpath,'model_v2.h5'))
	waldofind.find_waldo(20,FOUNDWALDOpath)
