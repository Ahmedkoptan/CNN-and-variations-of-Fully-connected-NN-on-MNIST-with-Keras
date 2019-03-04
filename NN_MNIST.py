import os
import numpy as np
import pandas as pd
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
import pylab
from sklearn.metrics import accuracy_score
import tensorflow as tf

#To stop potential randomness
seed=128
rng=np.random.RandomState(seed)

#Set directory paths for safekeeping
root_dir=os.path.abspath('TF_1/..')
data_dir=os.path.join(root_dir,'data')
sub_dir=os.path.join(root_dir,'sub')

#check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)

#Reading datasets (filename, label)
train = pd.read_csv(os.path.join(data_dir, 'Train', 'train.csv'))
test = pd.read_csv(os.path.join(data_dir,'Test.csv'))
sample_submission = pd.read_csv(os.path.join(data_dir,'Sample_Submission.csv'))

train.head()

"""
#Sample a random image and display it to see what data looks like
img_name = rng.choice(train.filename)
filepath = os.path.join(data_dir,'Train','Images','train',img_name)

img=imread(filepath)

pylab.imshow(img,cmap='gray')
pylab.axis('off')
pylab.show()
"""

#For easier data manipulation, we store all of our images as numpy arrays
temp = []
for img_name in train.filename:
    img_path = os.path.join(data_dir,'Train','Images','train',img_name)
    img = imread(img_path)
    img = img.astype('float32')
    temp.append(img)

train_x = np.stack(temp)

temp = []
for img_name in test.filename:
    img_path = os.path.join(data_dir,'Train','Images','test',img_name)
    img = imread(img_path)
    img = img.astype('float32')
    temp.append(img)

test_x = np.stack(temp)

split_size = int(train_x.shape[0]*0.7)
train_x , val_x = train_x[:split_size] , train_x[split_size:]
train_y , val_y = train.label.values[:split_size] , train.label.values[split_size:]

#Defining some helper functions to be used later
def dense_to_one_hot(labels_dense,num_classes = 10):
    """Convert class labels from scalars to one-hot vectors"""
    """ My Implementation
        y_oh = np.zeros(len(labels_dense),num_classes) #can use np.max(labels_dense)+1 as the second argument instead
        y_oh[np.arange(len(labels_dense)),labels_dense] = 1
        return y_oh
    """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x  / unclean_batch_x.max() #since minimum = 0 in grayscale/ image values, therefore range = max, and max is
    #the same for all features (i.e. pixels)
    #We didn't subtract mean since we want the range to be from 0 to 1, not -0.5 to 0.5
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length,batch_size)
    batch_x = eval(dataset_name+'_x')[[batch_mask]].reshape(-1,input_num_units)
    batch_x = preproc(batch_x)

    if dataset_name == 'train':
        batch_y = eval(dataset_name).ix[batch_mask,'label'].values ###Don't understand this line
        batch_y = dense_to_one_hot(batch_y)

    return batch_x,batch_y




## NN Architecture
# number of neurons in each layer
input_num_units = 28*28
hidden_num_units = 500
output_num_units = 10


# define placeholders ####
x = tf.placeholder(tf.float32, [None,input_num_units])
y = tf.placeholder(tf.float32, [None,output_num_units])

#set remaining variables
epochs = 5
batch_size = 128
learning_rate = 0.01

#define weights and biases of the NN
weights = {
    'hidden' :tf.Variable(tf.random_normal([input_num_units,hidden_num_units],seed=seed)),
    'output' :tf.Variable(tf.random_normal([hidden_num_units,output_num_units],seed=seed))
}
biases = {
    'hidden' :tf.Variable(tf.random_normal([hidden_num_units],seed=seed)),
    'output' :tf.Variable(tf.random_normal([output_num_units],seed=seed))
}

#Now create our neural networks computational graph
hidden_layer = tf.add(tf.matmul(x,weights['hidden']),biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)
output_layer = tf.matmul(hidden_layer,weights['output']) + biases['output'] ###

#We need to define cost of Neural Network
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer,labels=y))

#We set the optimizer (backprop algorithm), we use Adam, an efficient variant of gradient descent link: https://arxiv.org/abs/1412.6980
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)




#After we defined NN architecture, we now initilize all of our variables
init = tf.global_variables_initializer()




#Now we we create a session and run our NN in the session, and evaluate our model accuracy on validation set
with tf.Session() as sess:
    sess.run(init)
    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(train.shape[0]/batch_size)
        for i in range(total_batch):
            batch_x,batch_y = batch_creator(batch_size,train_x.shape[0],'train')
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
            avg_cost += c /total_batch

        print "Epoch:",(epoch+1),"cost = ","{:.5f}".format(avg_cost) ####

    print "\nTraining complete!"

    #find predictions on validation set
    pred_temp = tf.equal(tf.argmax(output_layer,1),tf.arg_max(y,1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp,"float"))
    print "Validation accuracy ", accuracy.eval({x:val_x.reshape(-1,input_num_units),y:dense_to_one_hot(val_y)})

    predict = tf.argmax(output_layer,1)
    pred = predict.eval({x:test_x.reshape(-1,input_num_units)})


#To test our model by visualizing
img_name = rng.choice(test.filename)
filepath = os.path.join(data_dir, 'Train', 'Images', 'test', img_name)

img = imread(filepath, flatten=True)

test_index = int(img_name.split('.')[0]) - 49000

print "Prediction is: ", pred[test_index]

pylab.imshow(img, cmap='gray')
pylab.axis('off')
pylab.show()

#Creating a submission
sample_submission.filename = test.filename

sample_submission.label = pred

sample_submission.to_csv(os.path.join(sub_dir, 'sub01.csv'), index=False)







