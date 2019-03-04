Implementing a Convolutional Neural Network and variations of a fully connected Neural Network on MNIST handwritten digits with Keras (Tensorflow as backend). 

The file Ker_TF.py implements a draft of the fully connected Neural Network model with a relatively weaker accuracy

The file Ker_TF_Tweak.py implements variations of the fully connected Neural Network model with either wider network, deeper, or both

The file Ker_CNN_MNIST.py implements a Convolutional Neural Network with 3 Convolution layers followed by max pooling and then 2 fully connected layers

The file NN_MNIST uses only Tensorflow to implement a fully connected Neural Network

All files use accuracy as their metric, and they print out the accuracy and loss on both Validation and Training sets for every model. The files display a random testing image and its prediction as a test for each model at the end of each file.

While the fully connected wide and deep Neural Network with 25 epochs was able to achieve 99.36% accuracy on the Validation set, its training time was much more significant than the CNN, since it used only 5 epochs for training. The CNN achieved 98.10 on the Validation set.