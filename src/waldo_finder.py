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

	def __init__(self,imgpath):

		self.imgpath=imgpath
		self.img = cv2.imread(f'{imgpath}')
		self.resized = False
		self.model = None
		self.keras_img = img_to_array(load_img(f'{imgpath}'))
		self.scale=None
		self.resize_window=False
		self.rescale_check()

	def rescale_check(self):
		if self.keras_img.shape[0]>=3000:
			self.rescale_image(.4)
		if 3000>self.keras_img.shape[0]>=2750:
			self.rescale_image(.45)
		if 2750>self.keras_img.shape[0]>=2500:
			self.rescale_image(.5)
		if 2500>self.keras_img.shape[0]>=2250:
			self.rescale_image(.55)
		if 2250>self.keras_img.shape[0]>=2000:
			self.rescale_image(.6)
		if 2000>self.keras_img.shape[0]>=1700:
			self.rescale_image(.7)

	def rescale_image(self, scale):
		w = int(self.img.shape[1] * scale)
		self.img_resized = imutils.resize(self.img, width=w)
		self.keras_img_resized = rescale(self.keras_img ,scale,anti_aliasing=False)
		self.resized = True
		self.scale = scale

	def sliding_window(self, image, stepSize, windowSize):
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

	

	def test_sliding_window(self,windowsize,stepsize, savedir=None):
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
		for (x, y, window) in self.sliding_window(img, stepSize=stepsize, windowSize=(winW, winH)):
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
					predictionr=round(self.model.predict(window_gen)[0][0],3)
				print(predictionr)
				clone = img.copy()
				if savedir:
					if prediction>.80:
						cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 5)
						cv2.putText(clone, text=f'Waldo!{predictionr}', org=(x,y),
									fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0),thickness=3)
						cv2.imshow("Window", clone)
						io.imsave(savedir+f'/window{p}.jpg',keras_window.astype('uint8'))
						print(f'Found Waldo at {x},{y}')
						cordlist.append((x,y))
						problist.append((predictionr))
						p+=1
					else:
						cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 0), 2)
						cv2.imshow("Window", clone)
				cv2.waitKey(1)
		print(f"Found waldo {p} times!")
		final=img.copy()
		for cord,prob in zip(cordlist,problist):
			cv2.rectangle(final, cord, (cord[0] + winW, cord[1] + winH), (0, 255, 0), 3)
			cv2.putText(clone, text=f'Waldo!{prob}', org=cord,
						fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0),thickness=2)
		fn = os.path.split(self.imgpath)[1]
		cv2.imshow('img',final)
		cv2.imwrite(savedir+f'/{p}_waldos_{fn}.jpg',final)

if __name__ == "__main__":
	waldofind = WaldoFinder(IMGTESTpath)
	# waldofind.test_sliding_window((64,64),128,resized=True,savedir=GIFpath)
	waldofind.load_model(os.path.join(MODELpath,'model_v1.h5'))
	waldofind.find_waldo(20,(50,50),FOUNDWALDOpath)
