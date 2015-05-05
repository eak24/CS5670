import numpy as np
from cs4670.classifier_trainer import ClassifierTrainer
from cs4670.gradient_check import eval_numerical_gradient
from cs4670.classifiers.convnet import *
from cs4670.data_utils import load_CIFAR10
from math import log, exp
from random import random
from random import randint as irandrange

#best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
#X_train[:50], y_train[:50], X_val, y_val, model, two_layer_convnet,
#reg=0.001, momentum=0.9, learning_rate=0.0001, batch_size=10, num_epochs=10,
#verbose=True)

randrange = lambda x,y: random()*(y-x)+x
loge = lambda x: log(x)/log(exp(1))
#best_model, loss_history, train_acc, val_acc = trainer.train(X_train, y_train, X_val, y_val,
#                                             model, two_layer_net,
#                                             num_epochs=5, reg=1.0,
#                                             momentum=0.9, learning_rate_decay = 0.95,
#                                             learning_rate=1e-5, verbose=True)
#model = init_two_layer_convnet(filter_size=7)
#def init_two_layer_convnet(weight_scale=1e-3, bias_scale=0, input_shape=(3, 32, 32),
#                           num_classes=10, num_filters=32, filter_size=5):]\

reg = loge(0.001)
reg = (reg-3,reg+3)
momentum = loge(0.9)
momentum = (momentum-3,momentum+3)
learning_rate_decay = loge(.95)
learning_rate_decay = (learning_rate_decay-3,learning_rate_decay+3)
learning_rate = loge(1e-5)
learning_rate = (learning_rate-3,learning_rate+3)
filter_size = (2,3)
num_filters = (16,48)

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs4670/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    return X_train, y_train, X_val, y_val, X_test, y_test


print 'Loading data'
# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape
model = init_two_layer_convnet()

X = np.random.randn(100, 3, 32, 32)
y = np.random.randint(10, size=100)

loss, _ = two_layer_convnet(X, model, y, reg=0)

# Sanity check: Loss should be about log(10) = 2.3026
print 'Sanity check loss (no regularization): ', loss

# Sanity check: Loss should go up when you add regularization
loss, _ = two_layer_convnet(X, model, y, reg=1)
print 'Sanity check loss (with regularization): ', loss
from random import randint 
from pickle import dump
import warnings
i = irandrange(0,2<<20)
dumpster = open('%x.performance'%i,'w')
best_acc = 0
print 'Starting main loop'
while True:
	with warnings.catch_warnings():
		warnings.simplefilter("error")
		try:
			trainer = ClassifierTrainer()
			model = init_my_new_convnet(filter_size=5)#init_my_convnet(filter_size=5)
			_reg = exp(randrange(*reg))
			_learning_rate = exp(randrange(*learning_rate))
			print 'Running new model'
			print 'reg\t\tlr\t\t'
			print '%.2e\t%.2e'%(_reg,_learning_rate)
			best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
		          X_train[:1000], y_train[:1000], X_val, y_val, model, my_new_convnet,update = 'rmsprop',
		          reg=_reg, momentum=0.9, learning_rate=_learning_rate, batch_size=50, num_epochs=1,
		          acc_frequency=50, verbose=True)
			if max(val_acc_history)>best_acc:
				best_acc=max(val_acc_history)
				print 'Best so far, with an accuracy of %f'%best_acc
			dumpster.seek(0)
			dumpster.write('%e %e %e\n',%(_reg,_learning_rate,max(val_acc_history)))
		except Warning,w:
			print 'Got warning %s, abandon ship!'%w
			continue
