from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.keras.backend.clear_session() 
from tensorflow import keras
from tensorflow.keras import layers

# All this is doing it loading in the models and exporting them to models which OpenCV can use.(A .pb format)

model1 = keras.models.load_model('../model/model_v1.h5')
model1.save('../model/model1pb', save_format='tf')
tf.keras.backend.clear_session()
model2 = keras.models.load_model('../model/model_v2.h5')
model2.save('../model/model2pb', save_format='tf')
tf.keras.backend.clear_session()

# This below is the old dpreciated way of doing this
# keras.experimental.export_saved_model(model, '../model/model1pb.pb')