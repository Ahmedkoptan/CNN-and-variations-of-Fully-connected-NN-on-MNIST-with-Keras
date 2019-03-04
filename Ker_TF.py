import os
import numpy as np
import pandas as pd
from scipy.misc import imread #from matplotlib.pyplot import imread from matplotlib.pyplot import imshow
from sklearn.metrics import accuracy_score


import tensorflow as tf
import keras

import pylab

#Setting a seed value to control model randomness
seed = 128
rng = np.random.RandomState(seed)

#setting directory paths for safekeeping
root_dir = os.path.abspath('../TF_1')
data_dir = os.path.join(root_dir,'data')
sub_dir = os.path.join(root_dir,'sub')
#Check if these paths exist
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)

#Reading dataset from .csv format containing filename and labels
train = pd.read_csv(os.path.join(data_dir, 'Train','train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'Test.csv'))

sample_submission = pd.read_csv(os.path.join(data_dir,'Sample_Submission.csv'))

#train.head() To see what the train.csv table looks like

"""
#To see what our images look like
img_name = rng.choice(train['filename'])
file_path = os.path.join(data_dir, 'Train', 'Images','train',img_name )

img = imread(file_path, flatten=True)

pylab.imshow(img,cmap='gray')
pylab.axis('off')
pylab.show()
"""

#Store all the images as numpy arrays
temp = []
for img_name in train.filename:
    img_path = os.path.join(data_dir,'Train','Images','train',img_name)
    img = imread(img_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

train_x = np.stack(temp)

train_x = train_x / 255.0 #now ranges from 0 to 1
train_x = train_x.reshape(-1,784).astype('float32')

temp = []
for img_name in test.filename:
    img_path = os.path.join(data_dir,'Train','Images','test',img_name)
    img = imread(img_path,flatten=True)
    img = img.astype('float32')
    temp.append(img)

test_x = np.stack(temp)

test_x = test_x / 255.0
test_x = test_x.reshape(-1,784).astype('float32')

train_y = keras.utils.np_utils.to_categorical(train.label.values) #### NEW

#We split our data into 70 : 30 for training vs validation
split_size = int(train_x.shape[0]*0.7)
train_x , val_x = train_x[:split_size],train_x[split_size:]
train_y ,val_y = train_y[:split_size],train_y[split_size:]

train.label.ix[split_size:] ####WHAT THE FUCK IS THIS DOING HERE?


#Now we define Neural Network Architecture

#Variables
input_num_units = 784
hidden_num_units = 50
output_num_units = 10

epochs = 5
batch_size = 128

#Import Keras Modules
from keras.models import Sequential
from keras.layers import Dense

#Create model
model = Sequential([Dense(output_dim = hidden_num_units,input_dim = input_num_units,activation = 'relu'),
                    Dense(output_dim = output_num_units, input_dim=hidden_num_units, activation='softmax'),
                    ])

#Compile the model with necessary attributes
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Now we train the model
model.fit(x = train_x, y = train_y, batch_size = batch_size, epochs = epochs, validation_data=(val_x,val_y))


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


