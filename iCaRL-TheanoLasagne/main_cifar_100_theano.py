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
nb_cl      = 10             # Classes per group 
nb_protos  = 20             # Number of prototypes per class at the end: total protoset memory/ total number of classes
epochs     = 70             # Total number of epochs 
lr_old     = 2.             # Initial learning rate
lr_strat   = [49, 63]       # Epochs where learning rate gets decreased
lr_factor  = 5.             # Learning rate decrease factor
wght_decay = 0.00001        # Weight Decay
nb_runs    = 10             # Number of runs (random ordering of classes at each run)
np.random.seed(1993)        # Fix the random seed
########################################

# Load the dataset
print("Loading data...")
data = utils_cifar100.load_data(nb_val)
X_train_total = data['X_train']
Y_train_total = data['Y_train']
if nb_val != 0:
    X_valid_total = data['X_valid']
    Y_valid_total = data['Y_valid']
else:
    X_valid_total = data['X_test']
    Y_valid_total = data['Y_test']

# Initialization
dictionary_size     = 500-nb_val
top1_acc_list_cumul = np.zeros((100/nb_cl,3,nb_runs))
top1_acc_list_ori   = np.zeros((100/nb_cl,3,nb_runs))

# Launch the different runs 
for iteration_total in range(nb_runs):
    
    # Select the order for the class learning 
    order = np.arange(100)
    np.random.shuffle(order)
    np.save('order', order)
    
    # Prepare Theano variables for inputs and targets
    input_var  = T.tensor4('inputs')
    target_var = T.matrix('targets')
    
    # Create neural network model
    print('Run {0} starting ...'.format(iteration_total))
    print("Building model and compiling functions...")
    [network,intermed] = utils_cifar100.build_cnn(input_var, n)
    
    prediction  = lasagne.layers.get_output(network)
    loss        = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss        = loss.mean()
    all_layers  = lasagne.layers.get_all_layers(network)
    l2_penalty  = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * wght_decay
    loss        = loss + l2_penalty
    
    # Create a training function
    params   = lasagne.layers.get_all_params(network, trainable=True)
    lr       = lr_old
    sh_lr    = theano.shared(lasagne.utils.floatX(lr))
    updates  = lasagne.updates.momentum(loss, params, learning_rate=sh_lr, momentum=0.9)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    
    # Create a validation/testing function
    test_prediction          = lasagne.layers.get_output(network, deterministic=True)
    test_prediction_intermed = lasagne.layers.get_output(intermed, deterministic=True)
    test_loss                = lasagne.objectives.binary_crossentropy(test_prediction,target_var)
    test_loss                = test_loss.mean()
    val_fn                   = theano.function([input_var, target_var], [test_loss,test_prediction,test_prediction_intermed])
    
    # Create a feature mapping function
    pred_map     = lasagne.layers.get_output(intermed, deterministic=True)
    function_map = theano.function([input_var], [pred_map])
    
    # Initialization of the variables for this run
    X_valid_cumuls    = []
    X_protoset_cumuls = []
    X_train_cumuls    = []
    Y_valid_cumuls    = []
    Y_protoset_cumuls = []
    Y_train_cumuls    = []
    alpha_dr_herding  = np.zeros((100/nb_cl,dictionary_size,nb_cl),np.float32)
    
    # The following contains all the training samples of the different classes 
    # because we want to compare our method with the theoretical case where all the training samples are stored
    prototypes = np.zeros((100,dictionary_size,X_train_total.shape[1],X_train_total.shape[2],X_train_total.shape[3]))
    for orde in range(100):
        prototypes[orde,:,:,:,:] = X_train_total[np.where(Y_train_total==order[orde])]
    
    for iteration in range(100/nb_cl):
        # Save data results at each increment
        np.save('top1_acc_list_cumul_icarl_cl'+str(nb_cl),top1_acc_list_cumul)
        np.save('top1_acc_list_ori_icarl_cl'+str(nb_cl),top1_acc_list_ori)
        
        # Prepare the training data for the current batch of classes
        actual_cl        = order[range(iteration*nb_cl,(iteration+1)*nb_cl)]
        indices_train_10 = np.array([i in order[range(iteration*nb_cl,(iteration+1)*nb_cl)] for i in Y_train_total])
        indices_test_10  = np.array([i in order[range(iteration*nb_cl,(iteration+1)*nb_cl)] for i in Y_valid_total])
        X_train          = X_train_total[indices_train_10]
        X_valid          = X_valid_total[indices_test_10]
        X_valid_cumuls.append(X_valid)
        X_train_cumuls.append(X_train)
        X_valid_cumul    = np.concatenate(X_valid_cumuls)
        X_train_cumul    = np.concatenate(X_train_cumuls)
        Y_train          = Y_train_total[indices_train_10]
        Y_valid          = Y_valid_total[indices_test_10]
        Y_valid_cumuls.append(Y_valid)
        Y_train_cumuls.append(Y_train)
        Y_valid_cumul    = np.concatenate(Y_valid_cumuls)
        Y_train_cumul    = np.concatenate(Y_train_cumuls)
        
        # Add the stored exemplars to the training data
        if iteration==0:
            X_valid_ori = X_valid
            Y_valid_ori = Y_valid
        else:
            X_protoset = np.concatenate(X_protoset_cumuls)
            Y_protoset = np.concatenate(Y_protoset_cumuls)
            X_train    = np.concatenate((X_train,X_protoset),axis=0)
            Y_train    = np.concatenate((Y_train,Y_protoset))
        
        # Launch the training loop
        sh_lr.set_value(lasagne.utils.floatX(lr_old))
        print("\n")
        print('Batch of classes number {0} arrives ...'.format(iteration+1))
        for epoch in range(epochs):
            # Shuffle training data
            train_indices = np.arange(len(X_train))
            np.random.shuffle(train_indices)
            X_train = X_train[train_indices,:,:,:]
            Y_train = Y_train[train_indices]
            # In each epoch, we do a full pass over the training data:
            train_err     = 0
            train_batches = 0
            start_time    = time.time()
            for batch in utils_cifar100.iterate_minibatches(X_train, Y_train, batch_size, shuffle=True, augment=True):
                inputs, targets_prep = batch
                targets = np.zeros((inputs.shape[0],100),np.float32)
                targets[range(len(targets_prep)),targets_prep.astype('int32')] = 1.
                old_train = train_err
                if iteration == 0:
                    train_err += train_fn(inputs, targets)
                
                # Distillation
                if iteration>0:
                    prediction_old = func_pred(inputs)[0]
                    targets[:,np.array(order[range(0,iteration*nb_cl)])] = prediction_old[:,np.array(order[range(0,iteration*nb_cl)])]
                    train_err += train_fn(inputs, targets)
                
                if (train_batches%100) == 1:
                    print(train_err-old_train)
                
                train_batches += 1
                
            # And a full pass over the validation data:
            val_err     = 0
            top5_acc    = 0
            top1_acc    = 0
            val_batches = 0
            for batch in utils_cifar100.iterate_minibatches(X_valid, Y_valid, min(500,len(X_valid)), shuffle=False):
                inputs, targets_prep = batch
                targets = np.zeros((inputs.shape[0],100),np.float32)
                targets[range(len(targets_prep)),targets_prep.astype('int32')] = 1.
                err,pred,pred_inter = val_fn(inputs, targets)
                pred_ranked = pred.argsort(axis=1).argsort(axis=1)
                for i in range(inputs.shape[0]):
                    top5_acc = top5_acc+np.float((pred_ranked[i,targets_prep[i]]>=(100-5)))/inputs.shape[0]
                    top1_acc = top1_acc+np.float((pred_ranked[i,targets_prep[i]]>=(100-1)))/inputs.shape[0]
                
                val_err     += err
                val_batches += 1
            # Then we print the results for this epoch:
            print("Batch of classes {} out of {} batches".format(
                iteration + 1, 100/nb_cl))
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  top 1 accuracy:\t\t{:.2f} %".format(
                top1_acc / val_batches * 100))
            print("  top 5 accuracy:\t\t{:.2f} %".format(
                top5_acc / val_batches * 100))
            # adjust learning rate
            if (epoch+1) in lr_strat:
                new_lr = sh_lr.get_value() * 1. / lr_factor
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))
        
        # Duplicate current network to distillate info
        if iteration==0:
            [network2,intermed2] = utils_cifar100.build_cnn(input_var, n)
            prediction_distil    = lasagne.layers.get_output(network2, deterministic=True)
            prediction_features  = lasagne.layers.get_output(intermed2, deterministic=True)
            func_pred      = theano.function([input_var], [prediction_distil])
            func_pred_feat = theano.function([input_var], [prediction_features])
        
        params_values = lasagne.layers.get_all_param_values(network)
        lasagne.layers.set_all_param_values(network2, params_values)

        # Save the network
        np.savez('net_incr'+str(iteration+1)+'_of_'+str(100/nb_cl)+'.npz', *lasagne.layers.get_all_param_values(network))
        np.savez('intermed_incr'+str(iteration+1)+'_of_'+str(100/nb_cl)+'.npz', *lasagne.layers.get_all_param_values(intermed))
        
        ### Exemplars 
        nb_protos_cl = int(np.ceil(nb_protos*100./nb_cl/(iteration+1)))
        # Herding
        print('Updating exemplar set...')
        for iter_dico in range(nb_cl):
            # Possible exemplars in the feature space and projected on the L2 sphere
            mapped_prototypes = function_map(np.float32(prototypes[iteration*nb_cl+iter_dico]))
            D = mapped_prototypes[0].T
            D = D/np.linalg.norm(D,axis=0)
                        
            # Herding procedure : ranking of the potential exemplars
            mu  = np.mean(D,axis=1)
            alpha_dr_herding[iteration,:,iter_dico] = alpha_dr_herding[iteration,:,iter_dico]*0
            w_t = mu
            iter_herding     = 0
            iter_herding_eff = 0
            while not(np.sum(alpha_dr_herding[iteration,:,iter_dico]!=0)==min(nb_protos_cl,500)) and iter_herding_eff<1000:
                tmp_t   = np.dot(w_t,D)
                ind_max = np.argmax(tmp_t)
                iter_herding_eff += 1
                if alpha_dr_herding[iteration,ind_max,iter_dico] == 0:
                    alpha_dr_herding[iteration,ind_max,iter_dico] = 1+iter_herding
                    iter_herding += 1
                w_t = w_t+mu-D[:,ind_max]
            
        # Prepare the protoset
        X_protoset_cumuls = []
        Y_protoset_cumuls = []
        
        # Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
        print('Computing mean-of_exemplars and theoretical mean...')
        class_means = np.zeros((64,100,2))
        for iteration2 in range(iteration+1):
            for iter_dico in range(nb_cl):
                current_cl = order[range(iteration2*nb_cl,(iteration2+1)*nb_cl)]
                
                # Collect data in the feature space for each class
                mapped_prototypes = function_map(np.float32(prototypes[iteration2*nb_cl+iter_dico]))
                D = mapped_prototypes[0].T
                D = D/np.linalg.norm(D,axis=0)
                # Flipped version also
                mapped_prototypes2 = function_map(np.float32(prototypes[iteration2*nb_cl+iter_dico][:,:,:,::-1]))
                D2 = mapped_prototypes2[0].T
                D2 = D2/np.linalg.norm(D2,axis=0)
                
                # iCaRL
                alph = alpha_dr_herding[iteration2,:,iter_dico]
                alph = (alph>0)*(alph<nb_protos_cl+1)*1.
                X_protoset_cumuls.append(prototypes[iteration2*nb_cl+iter_dico,np.where(alph==1)[0]])
                Y_protoset_cumuls.append(order[iteration2*nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))
                alph = alph/np.sum(alph)
                class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
                class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])
                
                # Normal NCM
                alph = np.ones(dictionary_size)/dictionary_size
                class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
                class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])
        
        np.save('cl_means', class_means)

        # Calculate validation error of model on the first nb_cl classes:
        print('Computing accuracy on the original batch of classes...')
        top1_acc_list_ori   = utils_cifar100.accuracy_measure(X_valid_ori, Y_valid_ori, class_means, val_fn, top1_acc_list_ori, iteration, iteration_total, 'original')
        
        # Calculate validation error of model on the cumul of classes:
        print('Computing cumulative accuracy...')
        top1_acc_list_cumul = utils_cifar100.accuracy_measure(X_valid_cumul, Y_valid_cumul, class_means, val_fn, top1_acc_list_cumul, iteration, iteration_total, 'cumul of')
     

# Final save of the data        
np.save('top1_acc_list_cumul_icarl_cl'+str(nb_cl),top1_acc_list_cumul)
np.save('top1_acc_list_ori_icarl_cl'+str(nb_cl),top1_acc_list_ori)