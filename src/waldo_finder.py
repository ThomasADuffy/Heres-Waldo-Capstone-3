import cv2
import imutils
import os
import sys

SRCpath=os.path.splt(os.path.abspath(__file__))[0]
ROOTpath = os.path.split(SRCpath)[0]
IMGTESTpath = os.path.join(ROOTpath,'imgtest')
GIFpath = os.path.join(IMGTESTpath,'gif')
IMGpath = os.path.join(IMGTESTpath,'test.jpg')


class WaldoFinder():

	def __init__(self,imgpath):

		self.img = cv2.imread(f'{imgpath}')
		self.img_resized=None

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
		stepSize=stepsize
		if resized:
			img = self.img_resized
		else:
			img = self.img
		for (x, y, window) in self.sliding_window(img, stepSize=64, windowSize=(winW, winH),resized=resized):
				# if the window does not meet our desired window size, ignore it
				if window.shape[0] != winH or window.shape[1] != winW:
					continue

				# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
				# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
				# WINDOW
				# since we do not have a classifier, we'll just draw the window
				clone = img.copy()
				cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 0), 2)
				cv2.imshow("Window", clone)
				cv2.waitKey(1)
				if savedir:
					cv2.imwrite(f'{savedir}/{p}.jpg',clone)
				p+=1

	def find_waldo(self):
		pass


if __name__ == "__main__":
	p