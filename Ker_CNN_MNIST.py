import os
import numpy as np
import pandas as pd
import pylab
from scipy.misc import imread
from sklearn.metrics import accuracy_score

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Activation, Dense, Convolution2D, Reshape, MaxPooling2D, Dropout, Flatten, InputLayer ###last 2?

#To stop potential randomness
seed = 128
rng = np.random.RandomState(seed=seed)

#Set paths for further
root_dir = os.path.abspath('../TF_1')
data_dir = os.path.join(root_dir,'data')
sub_dir = os.path.join(root_dir,'sub')

#Check if exists..

#Read the datasets and convert them to usable form
train = pd.read_csv(os.path.join(data_dir,'Train','train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'Test.csv'))

sample_submission = pd.read_csv(os.path.join(data_dir,'Sample_Submission.csv'))

temp = []
for img_name in train.filename:
    file_path = os.path.join(data_dir,'Train','Images','train',img_name)
    img = imread(file_path,flatten=True)
    img = img.astype('float32')
    temp.append(img)

train_x = np.stack(temp)
train_x = train_x / 255.0 #ranges from 0 to 1
train_x = train_x.reshape(-1,784).astype('float32')

temp = []
for img_name in test.filename:
    file_path = os.path.join(data_dir,'Train','Images','test',img_name)
    img = imread(file_path,flatten=True)
    img = img.astype('float32')
    temp.append(img)

test_x = np.stack(temp)
test_x = test_x / 255.0
test_x = test_x.reshape(-1,784).astype('float32')

train_y = keras.utils.np_utils.to_categorical(train.label.values)

#Split the data into 70:30
split_size = int(train_x.shape[0]*0.7)
train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]


#reshape the data
train_x_temp = train_x.reshape(-1,28,28,1)
val_x_temp = val_x.reshape(-1,28,28,1)

#Define vars
input_shape = (784,)
input_reshape = (28,28,1)

conv_num_filters = 5
conv_filter_size = 5

pool_size = (2,2)

hidden_num_units = 50
output_num_units = 10

epochs = 5
batch_size = 128

model = Sequential([
    InputLayer(input_shape= input_reshape),

    Convolution2D(25,5,5,activation='relu'),
    MaxPooling2D(pool_size=pool_size),

    Convolution2D(25,5,5,activation='relu'),
    MaxPooling2D(pool_size=pool_size),

    Convolution2D(25,4,4,activation='relu'),
    Flatten(),

    Dense(output_dim=hidden_num_units,activation='relu'),

    Dense(output_dim=output_num_units, input_dim=hidden_num_units, activation='softmax')

])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

trained_model_conv = model.fit(x=train_x_temp,y=train_y,batch_size=batch_size, epochs=epochs,validation_data=(val_x_temp,val_y))

#Now we visualize model predictions
pred = model.predict_classes(test_x)

test_name = rng.choice(test.filename)
test_path = os.path.join(data_dir,'Train','Images','test',test_name)

img = imread(test_path, flatten=True)

test_Index = int(test_name.split('.')[0]) - train.shape[0]

print "Prediction is: ", pred[test_Index]

pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()

sample_submission.filename = test.filename
sample_submission.label = pred
sample_submission.to_csv(os.path.join(sub_dir,'sub02.csv'),index=False)
