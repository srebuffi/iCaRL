import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import numpy as np
import scipy
import cPickle
import os
from scipy.spatial.distance import cdist
import scipy.io
import sys
# Syspath for the folder with the utils files
#sys.path.insert(0, "/data/sylvestre")

import utils_resnet
import utils_icarl
import utils_data

######### Modifiable Settings ##########
batch_size = 128            # Batch size
nb_cl      = 100             # Classes per group 
nb_groups  = 10             # Number of groups
top        = 5              # Choose to evaluate the top X accuracy 
is_cumul   = 'cumul'        # Evaluate on the cumul of classes if 'cumul', otherwise on the first classes
gpu        = '0'            # Used GPU
########################################

######### Paths  ##########
# Working station 
devkit_path = '/home/srebuffi'
train_path  = '/data/datasets/imagenets72'
save_path   = '/data/srebuffi/backup/'

###########################

# Load ResNet settings
str_mixing = str(nb_cl)+'mixing.pickle'
with open(str_mixing,'rb') as fp:
    mixing = cPickle.load(fp)

str_settings_resnet = str(nb_cl)+'settings_resnet.pickle'
with open(str_settings_resnet,'rb') as fp:
    order       = cPickle.load(fp)
    files_valid = cPickle.load(fp)
    files_train = cPickle.load(fp)

# Load class means
str_class_means = str(nb_cl)+'class_means.pickle'
with open(str_class_means,'rb') as fp:
      class_means = cPickle.load(fp)

# Loading the labels
labels_dic, label_names, validation_ground_truth = utils_data.parse_devkit_meta(devkit_path)

# Initialization
acc_list = np.zeros((nb_groups,3))

for itera in range(nb_groups):
    print("Processing network after {} increments\t".format(itera))
    # Evaluation on cumul of classes or original classes
    if is_cumul == 'cumul':
        eval_groups = np.array(range(itera+1))
    else:
        eval_groups = [0]
    
    print("Evaluation on batches {} \t".format(eval_groups))
    # Load the evaluation files
    files_from_cl = []
    for i in eval_groups:
        files_from_cl.extend(files_valid[i])
    
    inits,scores,label_batch,loss_class,file_string_batch,op_feature_map = utils_icarl.reading_data_and_preparing_network(files_from_cl, gpu, itera, batch_size, train_path, labels_dic, mixing, nb_groups, nb_cl, save_path) 
    
    with tf.Session(config=config) as sess:
        # Launch the prefetch system
        coord   = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(inits)
        
        # Evaluation routine
        stat_hb1     = []
        stat_icarl = []
        stat_ncm     = []
        
        for i in range(int(np.ceil(len(files_from_cl)/batch_size))):
            sc, l , loss,files_tmp,feat_map_tmp = sess.run([scores, label_batch,loss_class,file_string_batch,op_feature_map])
            mapped_prototypes = feat_map_tmp[:,0,0,:]
            pred_inter    = (mapped_prototypes.T)/np.linalg.norm(mapped_prototypes.T,axis=0)
            sqd_icarl     = -cdist(class_means[:,:,0,itera].T, pred_inter.T, 'sqeuclidean').T
            sqd_ncm       = -cdist(class_means[:,:,1,itera].T, pred_inter.T, 'sqeuclidean').T
            stat_hb1     += ([ll in best for ll, best in zip(l, np.argsort(sc, axis=1)[:, -top:])])
            stat_icarl   += ([ll in best for ll, best in zip(l, np.argsort(sqd_icarl, axis=1)[:, -top:])])
            stat_ncm     += ([ll in best for ll, best in zip(l, np.argsort(sqd_ncm, axis=1)[:, -top:])])
    
    print('Increment: %i' %itera)
    print('Hybrid 1 top '+str(top)+' accuracy: %f' %np.average(stat_hb1))
    print('iCaRL top '+str(top)+' accuracy: %f' %np.average(stat_icarl))
    print('NCM top '+str(top)+' accuracy: %f' %np.average(stat_ncm))
    acc_list[itera,0] = np.average(stat_icarl)
    acc_list[itera,1] = np.average(stat_hb1)
    acc_list[itera,2] = np.average(stat_ncm)
    
    # Reset the graph to compute the numbers ater the next increment
    tf.reset_default_graph()


np.save('results_top'+str(top)+'_acc_'+is_cumul+'_cl'+str(nb_cl),acc_list)
