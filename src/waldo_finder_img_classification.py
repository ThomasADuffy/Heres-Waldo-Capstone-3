import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from skimage.transform import rescale,resize
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Activation, Dropout, Flatten, Dense
from skimage import io, color, filters
import imutils
import tensorflow as tf

SRCpath = os.path.split(os.path.abspath(__file__))[0]
ROOTpath = os.path.split(SRCpath)[0]
IMGSpath = os.path.join(ROOTpath,'imgs')
GIFpath = os.path.join(IMGSpath,'gif')
IMGTESTpath = os.path.join(IMGSpath,'test3.jpg')
MODELpath = os.path.join(ROOTpath,'model')
FOUNDWALDOpath=os.path.join(IMGSpath,'waldo_found')


class WaldoFinder():

	''' This class will scan through and image and apply a model(in .h5 format)
	to classify each window set by a bounding box. It automatically rescales the image if they are over
	1700 pixels wide to a smaller size for speed reasons. The default box is 64,64 but any box can be used.
	it will resize the window to 64 by 64 to pass into the model because that is what these models are trained on.
	'''
	def __init__(self,imgpath):

		self.imgpath=imgpath
		self.img = cv2.imread(f'{imgpath}')
		self.resized = False
		self.model = None
		self.keras_img = img_to_array(load_img(f'{imgpath}'))
		self.scale=None
		self.resize_window=False
		self.rescale_check()
		self.img_name = os.path.split(self.imgpath)[1].strip('.jpg')

	def rescale_check(self):
		''' This will check how big the image is and see if it needs to rescale
		the image to a more reasonable size'''

		if self.img.shape[1]>=3000:
			self.rescale_image(.45)
		if 3000>self.img.shape[1]>=2750:
			self.rescale_image(.50)
		if 2750>self.img.shape[1]>=2500:
			self.rescale_image(.55)
		if 2500>self.img.shape[1]>=2250:
			self.rescale_image(.6)
		if 2250>self.img.shape[1]>=2000:
			self.rescale_image(.7)
		if 2000>self.img.shape[1]>=1700:
			self.rescale_image(.8)

	def rescale_image(self, scale):
		''' this will automatically rescale the image to the fraction set by
		the scale metric'''

		w = int(self.img.shape[1] * scale)
		self.img_resized = imutils.resize(self.img, width=w)
		self.keras_img_resized = rescale(self.keras_img ,scale,anti_aliasing=False)
		self.resized = True
		self.scale = scale

	def __sliding_window(self, image, stepSize, windowSize):
		'''This is the sliding window which goes across the image it just increments
		the y and x at the size of the windowSize, This is a helper function do not use!'''
		if self.resized:
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

	

	def _test_sliding_window(self,windowsize,stepsize, savedir=None):
		p=0
		(winW, winH) = windowsize
		if self.resized:
			img = self.img_resized
		else:
			img = self.img
		for (x, y, window) in self.sliding_window(img, stepSize=stepsize, windowSize=(winW, winH)):
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
		self.model_name=os.path.split(modelpath)[1].strip('.h5')

	def find_waldo(self,stepsize,windowsize=(64,64), savedir=None):
		if self.model:
			pass
		else:
			return 'No model loaded! please run load_model.'

		if self.resized:
			img = self.img_resized
		else:
			img = self.img
		p=0
		(winW, winH) = windowsize
		if winW!=winH:
			return 'Fix window size to be equal!'
		if winW!=64:
			self.resize_window=True
		cordlist=[]
		problist=[]
		for (x, y, window) in self.__sliding_window(img, stepSize=stepsize, windowSize=(winW, winH)):
				# if the window does not meet our desired window size, ignore it
				if window.shape[0] != winH or window.shape[1] != winW:
					continue
				if self.resized:
					keras_window=self.keras_img_resized[y:y + winH, x:x + winW]
					if self.resize_window:
						keras_window=resize(keras_window,(64,64))
					window_gen=ImageDataGenerator(rescale=1./255).flow(np.array([keras_window],dtype='float32'))
					prediction=self.model.predict(window_gen)[0][0]
					predictionr=round(self.model.predict(window_gen)[0][0],3)
				else:
					keras_window=self.keras_img[y:y + winH, x:x + winW]
					if self.resize_window:
						keras_window=resize(keras_window,(64,64))
					window_gen=ImageDataGenerator(rescale=1./255).flow(np.array([keras_window]))
					prediction=self.model.predict(window_gen)[0][0]
					predictionr=str(round(self.model.predict(window_gen)[0][0],3))
				print(predictionr)
				clone = img.copy()
				if savedir:
					if prediction>.50:
						cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 3)
						cv2.putText(clone, text=f'Waldo!{predictionr}', org=(x,y),
									fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=self.scale, color=(0,255,0),thickness=2)
						cv2.imshow("Window", clone)
						io.imsave(savedir+f'/window{p}_{self.img_name}_{self.model_name}.jpg',keras_window.astype('uint8'))
						print(f'Found Waldo at {x},{y}')
						cordlist.append((x,y))
						problist.append(predictionr)
						p+=1
					else:
						cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 0), 2)
						cv2.imshow("Window", clone)
				cv2.waitKey(1)
		print(f"Found waldo {p} times!")
		final=img.copy()
		for cord,prob in zip(cordlist,problist):
			cv2.rectangle(final, cord, (cord[0] + winW, cord[1] + winH), (0, 255, 0), 2)
			cv2.putText(final, text=f'Waldo!{prob}', org=cord,
						fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=self.scale, color=(0,255,0),thickness=2)

		cv2.imshow('final',final)
		cv2.imwrite(savedir+f'/{p}_waldos_{self.img_name}_{self.model_name}.jpg',final)

if __name__ == "__main__":
	# waldofind._test_sliding_window((64,64),128,resized=True,savedir=GIFpath)
	testinglst1 = ['test4.jpg','test5.jpg','test6.jpg']
	testinglst2 = ['test1.jpg','test2.jpg','test3.jpg']
	holdoutlst1 = ['holdout1.jpg','holdout2.jpg','holdout3.jpg']
	holdoutlst2 = ['holdout4.jpg','holdout5.jpg','holdout6.jpg']
	holdoutlst3 = ['holdout7.jpg','holdout8.jpg','holdout9.jpg']

	for imgname in holdoutlst1:
		imgpath=os.path.join(IMGSpath,imgname)
		waldofind = WaldoFinder(imgpath)
		waldofind.load_model(os.path.join(MODELpath,'model_v2.h5'))
		waldofind.find_waldo(20,(50,50),FOUNDWALDOpath)

