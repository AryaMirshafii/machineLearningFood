# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.layers.convolutional import *



import os
import os.path as path

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

from keras.preprocessing.image import ImageDataGenerator



MODEL_NAME = 'foodPredictions'

def build_model():
    # Init the cnn
    classifier = Sequential()

    #Convolution with first layer
    classifier.add(Conv2D(32, (3, 3), input_shape = (224, 224, 3), activation = 'relu'))

    #Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    #Adding a second layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    #Flattening
    classifier.add(Flatten())

    #Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 7, activation = 'softmax', name = "output"))

    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

 
def train_model(model):
    #Preprocessing of images


    train_datagen = ImageDataGenerator(rescale = 1./255,
    shear_range = 0.2,                                                              # Shear angle in counter-clockwise direction
    zoom_range = 0.2,                                                               # range for random zoom
    horizontal_flip = True)                                                         #  randomly flip inputs horizontaly
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory('foodData/training_set',
    target_size = (224, 224),
    batch_size = 50,
    class_mode = 'categorical')
    test_set = test_datagen.flow_from_directory('foodData/test_set',
    target_size = (224, 224),
    batch_size = 50,
    class_mode = 'categorical')


    model.fit_generator(training_set,
    steps_per_epoch = 140,                                                             # number of images in training set
    epochs = 800,     #change back to 25
    validation_data = test_set,
    validation_steps = 140) 




def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")



def main():
    if not path.exists('out'):
        os.mkdir('out')

    

    model = build_model()
    model.summary()
    train_model(model)

    
    export_model(tf.train.Saver(), model, ["conv2d_1_input"], "output/Softmax")


if __name__ == '__main__':
    main()





