import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def find_number_of_cores(current_num_cores, n_rows_per_core, n_total_rows):
    ''' This function will basically find the number of cores to use for
    parallel processing. This is used within the waldo class mainly

    Arguments:

    current_num_core = this is the current nubmer of avalible cores to do
    processing
    n_rows_per_core = this is the number of rows which will be processed per
    core
    n_total_rows = This is the number of total rows

    Parameters:

    num_cores_needed = this is number of cores needed
    cores_not_suffcient = this is wether the cores are suffcient or not, used
    in the while loop to end it when num_cores_needed are suffcient.

    Output:

    n_rows_per_core = the number of rows each core will process
    num_cores_needed = the number of cores that will be used'''
    num_cores_needed = int(n_total_rows / n_rows_per_core)
    if num_cores_needed <= current_num_cores:
        if (n_total_rows % n_rows_per_core) != 0:
            cores_not_sufficient = True
        else:
            cores_not_sufficient = False
    else:
        cores_not_sufficient = True

    while cores_not_sufficient:
        if num_cores_needed <= current_num_cores:
            if (n_total_rows % n_rows_per_core) == 0:
                cores_not_sufficient = False
                break
        n_rows_per_core += 1
        num_cores_needed = int(n_total_rows / n_rows_per_core)

    return n_rows_per_core, num_cores_needed


def parallelize_waldo_finder(winW_H, stepsize, modelpath,
                             threshold, diffrence_formula, slice_img_tup):
    '''This is the sliding window and prediction function made for the
    map function of parallel processing

    Arguments:

    slice_img_tuple = a tuple where the core is the first object and the slice
    of the image array is the second object
    winW_H = this is the windowsize
    stepsize = this is the stepsize of the window
    model = this is the model
    threshold = this is the threshold of probabillity to detrmine if it is 
    Waldo diffrence_formula = this is the formula of the distance between
    the start of the array vs the end of the array
    '''
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    model = load_model(modelpath)
    cordlist = []
    problist = []
    waldos_found = 0
    window_cords = cord_maker(stepsize, slice_img_tup[1])
    for y, x in window_cords:
        window = slice_img_tup[1][y:(y + winW_H), x:(x + winW_H)]
        if window.shape[0] != winW_H or window.shape[1] != winW_H:
            continue
        if winW_H != 64:
            window = resize(window, (64, 64))
        window_gen = ImageDataGenerator(rescale=1./255).flow(np.array([window],
                                                             dtype='float32'))
        prediction = model.predict(window_gen)[0][0]
        predictionr = round(float(prediction), 4)
        if prediction >= threshold:
            y_correct = y+(diffrence_formula*slice_img_tup[0])
            cordlist.append((x, y_correct))
            problist.append(predictionr)
            waldos_found += 1
    return [cordlist, problist, waldos_found]


def cord_maker(stepsize, slice_img):
    ''' This is make a list of all the possible cordinates for the windows,
    this was to cut down time even more.

    Arguments:

    stepsize = this is the stepsize of the window
    slice_img = this is the sliced image to get the shape of the array
    '''

    import numpy as np
    y, x, _ = slice_img.shape
    y = np.arange(0, y, 32)
    x = np.arange(0, x, 32)

    return np.array(np.meshgrid(y, x)).T.reshape(-1, 2)


if __name__ == "__main__":
    pass
