import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.datasets import mnist

import numpy as np
from matplotlib import pyplot as plt
import time

# this is necesary for NVIDIA-Turing Tooling
# otherwise GPU Memory will be flooded and 
# cuDNN results in an error
conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.Session(config=conf)
K.set_session(sess)

# check if we have a GPU available
assert (len(K.tensorflow_backend._get_available_gpus()) > 0), "No GPU available"

# some configuration
K.set_image_dim_ordering('th')
K.set_floatx('float32')

##################################################################################
# Preparation of Dataset
##################################################################################
# load the mnist dataset using the fancy keras function
print("loading mnist data...")
(x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnistdata')
print("...mnist data loaded: {}".format(x_train.shape))

#plt.imshow(x_train[0])

# Prepare the dataset
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
print("reshaping data to {}".format(x_train.shape))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print("Reshaped y data {}".format(y_train.shape))

##################################################################################
# Building the model and training
##################################################################################
model = Sequential()

early_stop = EarlyStopping(patience=2)
 
model.add(Convolution2D(16, (3, 3), activation='relu', input_shape=(1,28,28), name='input_conv'))
model.add(Dropout(0.3))
model.add(Convolution2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.4))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax', name='output_softmax'))
 
# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Fit model on training data
start = time.time()
model.fit(x_train, y_train, 
          batch_size=32, epochs=20, verbose=1, shuffle=True)
end = time.time()
print("Elapsed Time: {}".format(end-start))

# evaluate the trained model
score = model.evaluate(x_test, y_test, verbose=1)
print(score)

##################################################################################
# Save the trained model as checkpoint
##################################################################################
# Get the name of input- and output-nodes
model_input = model.input.name
model_output = model.output.name
print(model_input)
print(model_output)

# save the model to directory
sess = K.get_session()
saver = tf.train.Saver(tf.all_variables())
saver.save(sess, './modeldir/mnistModel')

