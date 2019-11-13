import os
import sys
import cv2
import imutils
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from functools import partial
from skimage import io, color, filters
from skimage.transform import rescale, resize
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

SRCpath = os.path.split(os.path.abspath(__file__))[0]
ROOTpath = os.path.split(SRCpath)[0]
IMGSpath = os.path.join(ROOTpath, 'imgs')
GIFpath = os.path.join(IMGSpath, 'gif')
IMGTESTpath = os.path.join(IMGSpath, 'test3.jpg')
MODELpath = os.path.join(ROOTpath, 'model')
FOUNDWALDOpath = os.path.join(IMGSpath, 'waldo_found')

sys.path.append(SRCpath)

from multiprocessing_helpers import *


class WaldoFinder():

    ''' This class will scan through and image and apply a model(in .h5 format)
    to classify each window set by a bounding box. It automatically rescales
    the image if they are over 1700 pixels wide to a smaller size for speed
    reasons. The default box is 64,64 but any box can be used. it will resize
    the window to 64 by 64 to pass into the model because that is what
    these models are trained on.

    Arguments:

    imgpath(str) = which is the path to the image
    flask(bool) = this designates if you this is on the flask app or not.
    vizulization(bool) = this will either vizualize or not the process,
    if flask is set to true this will not be able to be active currently.
    threshold(float) = this is probabilitiy threshold of deciding to classify
    if a window is waldo or not

    '''

    def __init__(self, imgpath, vizualization=False,
                 flask=False, threshold=.655, parallel=False):

        self.imgpath = imgpath
        self.img = cv2.imread(f'{imgpath}')
        self.model = None
        self.keras_img = img_to_array(load_img(f'{imgpath}'))
        self.scale = None
        self.resize_window = False
        self.rescale_check()
        self.img_name = os.path.split(self.imgpath)[1].strip('.jpg')
        self.flask = flask
        self.threshold = threshold
        if self.flask:
            self.vizualization = False
            print('''**Can not have vizulaization
            within flask app currently setting vizualization to False**''')
        else:
            self.vizualization = vizualization
        self.num_cores = mp.cpu_count()
        self.__argcheck()
        self.parallel = parallel
        if self.vizualization:
            self.parallel = False
            print('''**Can not have vizulaization
            with parallel model currently setting parallel to False**''')

    def __argcheck(self):
        if not isinstance(self.vizualization, bool):
            raise TypeError("Vizualization must be True or False")
        if not isinstance(self.flask, bool):
            raise TypeError("Flask must be True or False")
        if not isinstance(self.threshold, float):
            raise TypeError("Threshold must be a float between 0 and 1")
        if not isinstance(self.imgpath, str):
            raise TypeError("Imgpath must be a string")

    def rescale_check(self):
        ''' This will check how big the image is and see if it needs to rescale
        the image to a more reasonable size'''

        if self.img.shape[0] >= 3000:
            self.rescale_image(.6)
        elif 3000 > self.img.shape[0] >= 2750:
            self.rescale_image(.65)
        elif 2750 > self.img.shape[0] >= 2500:
            self.rescale_image(.7)
        elif 2500 > self.img.shape[0] >= 2250:
            self.rescale_image(.75)
        elif 2250 > self.img.shape[0] >= 2000:
            self.rescale_image(.8)
        elif 2000 > self.img.shape[0] >= 1700:
            self.rescale_image(.85)

    def rescale_image(self, scale):
        ''' this will automatically rescale the image to the fraction set by
        the scale metric

        Arguments:

        scale = this is the fraction that the picture will be scaled by'''

        w = int(self.img.shape[1] * scale)
        self.img = imutils.resize(self.img.copy(), width=w)
        self.keras_img = rescale(self.keras_img, scale,
                                         anti_aliasing=False,
                                         multichannel=True)
        self.scale = scale

    def __sliding_window(self, stepSize, windowSize):
        '''This is the sliding window which goes across the image it just
        increments the y and x at the size of the windowSize, This is a helper
        function do not use!

        Arguments:

        stepSize = the size of the steps fort the window in pixels
        windowSize = this is a tuple that is the size of the window,
        has to be equal'''
        for y in range(0, self.img.shape[0], stepSize):
            for x in range(0, self.img.shape[1], stepSize):
                # yield the current window
                yield (x, y,
                       self.img[y:y + windowSize[1], x:x + windowSize[0]])

    def _test_sliding_window(self, windowsize, stepsize, savedir=None):
        """This is to test the sliding window and visualize it (mainly used to
        create a gif) for an example

        Arguments:

        stepSize = the size of the steps fort the window in pixels
        windowSize = this is a tuple that is the size of the window,
        has to be equal
        savedir = the directory where images it will be saved(Used for gif)"""

        p = 0
        (winW, winH) = windowsize
        for (x, y, window) in self.__sliding_window(stepSize=stepsize,
                                                    windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            clone = self.img.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            if savedir:
                cv2.imwrite(savedir+f'/{p}.jpg', clone)
            p += 1

    def __prediction_on_window(self, x, y, winH, winW):
        ''' This will return the prediction on the given window and the rouned
        one.

        Arguments:

        x = the inputed x coordinate where the window is
        y = the inputed y coordinate where the window is
        winH = window height
        winW = window width
        keras_img = the image to predict on as a whole

        Output:

        prediction = the full prediction float (very long)
        preditionr = this is prediction rounded to 4 signifigant figures
        '''

        keras_window = self.keras_img[y:y + winH, x:x + winW]
        if self.resize_window:
            keras_window = resize(keras_window, (64, 64))
        window_gen = ImageDataGenerator(rescale=1./255).flow(np.array([keras_window], dtype='float32'))
        prediction = self.model.predict(window_gen)[0][0]
        predictionr = round(float(self.model.predict(window_gen)[0][0]), 4)
        return keras_window, prediction, predictionr

    def __top_10_waldo_probabilities(self, cordlist, problist):
        ''' This takes in the probability list and coordinate list and sorts
        them by the top 10 probabilities.

        Arguments:

        cordlist = list of coordinates
        problist = list of probabilities

        Output:

        top_10_cords = top 10 cordinates for boxes
        top_10_probs = top 10 probabilities
        '''

        prob_idx = np.argsort(problist)[::-1]
        prob_idx = prob_idx[:10]
        top_10_cord = np.array(cordlist)[prob_idx]
        top_10_prob = np.array(problist)[prob_idx]
        top_10_cord = top_10_cord.tolist()
        top_10_prob = top_10_prob.tolist()

        return top_10_cord, top_10_prob

    def __window_check(self, winW, winH):
        ''' This is a small check to see if we need to resize the window
        to pass into the CNN.

        Arguments:

        winW = window width
        winH = window height
        '''

        if winW != 64:
            self.resize_window = True

    def find_waldo(self, savedir, modelpath, stepsize=32, windowsize=(64, 64),
                   save_window=False):
        """ This function will actually run window and classify the window with
        the model loaded therefor finding waldo and output the top 10
        probabilities in an image to the savedir variable. the window size is
        the windowsize to look in(64,64 is the best size to use unless a very
        small image), stepsize is how much the window will move per
        classification across and down.

        Arguments:

        stepSize = the size of the steps fort the window in pixels
        windowSize = this is a tuple that is the size of the window,
        has to be equal
        savedir = the directory where final image it will be saved
        modelpath = the path of the model to load to use"""

        if self.parallel:
            return print('''***Cannot run Parallel model with normal function!
            please run find_waldo_parallelize!***''')
        self.model = load_model(modelpath)
        self.model_name = os.path.split(modelpath)[1].strip('.h5')
        if self.vizualization:
            print(f'Loaded model {self.model_name}')
        self.modelpath = modelpath
        cordlist = []
        problist = []
        waldos_found = 0
        (winW, winH) = windowsize
        if self.model:
            pass
        else:
            return print('No model loaded! please run load_model.')
        if winW != winH:
            return print('Fix window size to be equal!')
        self.__window_check(winW, winH)

        for (x, y, window) in self.__sliding_window(stepSize=stepsize,
                                                    windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            keras_window, prediction, predictionr = self.__prediction_on_window(x, y, winH,
                                                                  winW)
            clone = self.img.copy()
            if prediction > self.threshold:
                if self.vizualization:
                    print(predictionr)
                    cv2.rectangle(clone, (x, y),
                                  (x + winW, y + winH), (0, 255, 0), 3)
                    cv2.putText(clone, text=f'Waldo!{predictionr}',
                                org=(x, y),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=self.scale,
                                color=(0, 255, 0), thickness=2)
                    cv2.imshow("Window", clone)
                    cv2.waitKey(1)
                    if save_window:
                        io.imsave(savedir+f'/window{p}_{self.img_name}_{self.model_name}.jpg', keras_window.astype('uint8'))
                    print(f'Found Waldo at {x},{y}')
                cordlist.append((x, y))
                problist.append(predictionr)
                waldos_found += 1
            else:
                if self.vizualization:
                    print(predictionr)
                    cv2.rectangle(clone, (x, y), (x + winW, y + winH),
                                  (0, 0, 0), 2)
                    cv2.imshow("Window", clone)
                    cv2.waitKey(1)
        self._plot_top10_waldos_found(cordlist, problist,
                                          waldos_found, winH, savedir)

    def _plot_top10_waldos_found(self, cordlist, problist,
                                 waldos_found, winW_H, savedir):
        ''' This is to plot the model and save it, it will only plot the top 10
        cordinates in the model

        Arguments:

        cordlist = list of cordinates for waldos found
        problist = list of probabillities for waldos found
        waldos_found = number of waldos found
        winW_H = is window height and length
        savedir = the save directory'''

        top_10_cord, top_10_prob = self.__top_10_waldo_probabilities(cordlist,
                                                                     problist)
        final = self.img.copy()
        for cord, prob in zip(top_10_cord, top_10_prob):
            cord = tuple(cord)
            cv2.rectangle(final, cord, (cord[0] + winW_H, cord[1] + winW_H),
                        (0, 255, 0), 2)
            cv2.putText(final, text=f'Waldo!{prob}', org=cord,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=self.scale, color=(0, 255, 0), thickness=2)
        if self.vizualization:
            print(f"Found waldo {waldos_found} times!")
            print(f"Saving image in:{savedir+'/'+waldos_found+'waldos_'+self.img_name+'_'+self.model_name+'.jpg'}")
            cv2.imwrite(savedir+f'/{waldos_found}waldos_{self.img_name}_{self.model_name}.jpg', final)
            cv2.imshow('final', final)
            print('Press any key to close window')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            if self.flask:
                cv2.imwrite(savedir, final)
            else:
                cv2.imwrite(savedir+f'/{waldos_found}waldos_{self.img_name}_{self.model_name}.jpg', final)

    def find_waldo_parallelize(self, stepsize, winW_H, modelpath, savedir):
        ''' This function is to find waldo utilizing multicore processing,
        this is a work in progress and takes some tuning, please note this is
        meant to be used with a large ec2 instance. Please note also for a
        smaller step size, and window size, this causes a much higher core size
        and can ultimately end up slowing down the process if not enough cores
        are avalible visualization is not a possibility currently

        Arguments:

        stepSize = the size of the steps fort the window in pixels
        winW_H = this the size of the window's width and height(has to be equal)
        savedir = the directory where final image it will be saved
        modelpath = the path of the model to load to use'''

        if not self.parallel:
            return print('''Please reinstantiate the WaldoFinder
            object with self.parallel as True''')
        if self.model:
            self.model = None
            tf.keras.backend.clear_session()
        self.model_name = os.path.split(modelpath)[1].strip('.h5')

        array_height = (int((self.keras_img.shape[0]-winW_H)/stepsize)*stepsize)+winW_H
        array_length = (int((self.keras_img.shape[1]-winW_H)/stepsize)*stepsize)+winW_H
        keras_img_confined = self.keras_img[:array_height, :array_length, :]
        n_total_rows = ((array_height-winW_H)/stepsize)+1
        n_rows_per_core = 1
        num_cores_needed = int(n_total_rows / n_rows_per_core)
        n_rows_per_core, num_cores_needed = find_number_of_cores(self.num_cores, n_rows_per_core, n_total_rows)
        diffrence_formula = (stepsize*n_rows_per_core)
        cpu_pool = mp.Pool(num_cores_needed)
        parallel_array = []
        for core in range(num_cores_needed):
            sub_array_height_start = core*diffrence_formula
            sub_array_height_end = sub_array_height_start + diffrence_formula
            slice_img = keras_img_confined[sub_array_height_start:sub_array_height_end, :, :]
            parallel_array.append([core, slice_img])
        waldo_func = partial(parallelize_waldo_finder, winW_H, stepsize,
                             modelpath, self.threshold,
                             diffrence_formula, n_rows_per_core)
        results = cpu_pool.map(waldo_func, parallel_array)
        cpu_pool.close()
        cpu_pool.join()
        final = self.img.copy()
        cordlist =[]
        problist = []
        waldos_found = 0
        for cords, probs, waldos in results:
            for cord in cords:
                cordlist.append(cord)
            for prob in probs:
                problist.append(prob)
            waldos_found += waldos
        self._plot_top10_waldos_found(cordlist, problist,
                                      waldos_found, winW_H, savedir)


if __name__ == "__main__":
    # pass
    # waldofind._test_sliding_window((64,64),128,resized=True,savedir=GIFpath)
    # testinglst1 = ['test1.jpg','test2.jpg']
    # testinglst2 = ['test3.jpg','test4.jpg']
    # testinglst3 = ['test5.jpg','test6.jpg']
    # holdoutlst1 = ['holdout1.jpg','holdout2.jpg','holdout3.jpg']
    # holdoutlst2 = ['holdout4.jpg','holdout5.jpg']
    # holdoutlst3 = ['holdout6.jpg','holdout7.jpg']
    # holdoutlst4 = ['holdout8.jpg','holdout9.jpg']

    imgpath = os.path.join(IMGSpath, 'test3.jpg')
    waldofind = WaldoFinder(imgpath, parallel=True)
    # waldofind.find_waldo(savedir=FOUNDWALDOpath,
    #                      modelpath=os.path.join(MODELpath, 'model_v4.h5'),
    #                      stepsize=32, windowsize=(64, 64))
    waldofind.find_waldo_parallelize(32, 64, os.path.join(MODELpath, 'model_v4.h5'), savedir=FOUNDWALDOpath)