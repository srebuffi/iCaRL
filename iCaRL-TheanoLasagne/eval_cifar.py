# THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu1,floatX=float32,lib.cnmem=0.09' python
from __future__ import print_function
import sys
import os
import time
import string
import random
import pickle
import numpy as np
import theano
import theano.tensor as T
import lasagne
from scipy.spatial.distance import cdist
import utils_cifar100

######### Modifiable Settings ##########
batch_size = 128            # Batch size
n          = 5              # Set the depth of the architecture: n = 5 -> 32 layers (See He et al. paper)
nb_val     = 0              # Validation samples per class
nb_cl      = 10             # Number of classes per group
nb_increm  = 10             # Number of groups of classes (total)
increm     = 2              # Groups of classes already processed
eval_groups= [0,1]          # Groups on which performances are evaluated. NB: max(eval_groups) < increment
src_order  = 'order.npy'    # Ordering of classes
src_network= 'net_incr'+str(increm)+'_of_'+str(nb_increm)+'.npz'
src_intermed= 'intermed_incr'+str(increm)+'_of_'+str(nb_increm)+'.npz'
src_clmeans= 'cl_means.npy' # File for class means
########################################

# Load the dataset
print("\n")
print("Loading data...")
data = utils_cifar100.load_data(nb_val)
X_valid_total = data['X_test']
Y_valid_total = data['Y_test']

# Load the ordering of classes and class_means
order       = np.load(src_order)
class_means = np.load(src_clmeans)

# Loading the tested group of classes
X_valid_cumul = []
Y_valid_cumul = []
for group in eval_groups:
    actual_cl    = order[range(group*nb_cl,(group+1)*nb_cl)]
    indices_test = np.array([i in actual_cl for i in Y_valid_total])
    X_valid      = X_valid_total[indices_test]
    Y_valid      = Y_valid_total[indices_test]
    X_valid_cumul.append(X_valid)
    Y_valid_cumul.append(Y_valid)

X_valid_cumul    = np.concatenate(X_valid_cumul)
Y_valid_cumul    = np.concatenate(Y_valid_cumul)


# Initialization
top1_acc_list = np.zeros((1,3,1))
    
# Prepare Theano variables for inputs and targets
input_var  = T.tensor4('inputs')
target_var = T.matrix('targets')

# Loading neural network 
print("Building model and compiling functions...")
[network,intermed] = utils_cifar100.build_cnn(input_var, n)
with np.load(src_network) as f:
     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)
with np.load(src_intermed) as f:
     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(intermed, param_values)

# Create a validation/testing function
test_prediction          = lasagne.layers.get_output(network, deterministic=True)
test_prediction_intermed = lasagne.layers.get_output(intermed, deterministic=True)
test_loss                = lasagne.objectives.binary_crossentropy(test_prediction,target_var)
test_loss                = test_loss.mean()
val_fn                   = theano.function([input_var, target_var], [test_loss,test_prediction,test_prediction_intermed])
    
# Calculate validation error of model on the first nb_cl classes:
print('Computing accuracy on the selected batches of classes...')
top1_acc_list   = utils_cifar100.accuracy_measure(X_valid_cumul, Y_valid_cumul, class_means, val_fn, top1_acc_list, 0, 0, 'selected')

# Final save of the data        
np.save('top1_acc_list_icarl_cl'+str(nb_cl)+'_on_'+str(eval_groups)+'_at_'+str(increm),top1_acc_list)
