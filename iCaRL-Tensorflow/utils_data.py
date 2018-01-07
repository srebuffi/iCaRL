import tensorflow as tf
import numpy as np
import os
import scipy.io
import sys
try:
    import cPickle
except:
    import _pickle as cPickle

def parse_devkit_meta(devkit_path):
    meta_mat                = scipy.io.loadmat(devkit_path+'/meta.mat')
    labels_dic              = dict((m[0][1][0], m[0][0][0][0]-1) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
    label_names_dic         = dict((m[0][1][0], m[0][2][0]) for m in meta_mat['synsets'] if m[0][0][0][0] >= 1 and m[0][0][0][0] <= 1000)
    label_names             = [tup[1] for tup in sorted([(v,label_names_dic[k]) for k,v in labels_dic.items()], key=lambda x:x[0])]    
    fval_ground_truth       = open(devkit_path+'/data/ILSVRC2012_validation_ground_truth.txt','r')
    validation_ground_truth = [[int(line.strip()) - 1] for line in fval_ground_truth.readlines()]
    fval_ground_truth.close()
    
    return labels_dic, label_names, validation_ground_truth

def read_data(prefix, labels_dic, mixing, files_from_cl):
    image_list = sorted(map(lambda x: os.path.join(prefix, x),
                        filter(lambda x: x.endswith('JPEG'), files_from_cl)))
    prefix2 = []
    
    for file_i in image_list:
        tmp = file_i.split(prefix+'/')[1].split("_")[0]
        prefix2.append(tmp)
    
    prefix2     = np.array(prefix2)
    labels_list = np.array([mixing[labels_dic[i]] for i in prefix2])
    
    assert(len(image_list) == len(labels_list))
    images             = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels             = tf.convert_to_tensor(labels_list, dtype=tf.int32)
    input_queue        = tf.train.slice_input_producer([images, labels], shuffle=True, capacity=2000)
    image_file_content = tf.read_file(input_queue[0])
    label              = input_queue[1]
    image              = tf.image.resize_images(tf.image.decode_jpeg(image_file_content, channels=3), [256, 256])
    image              = tf.random_crop(image, [224, 224, 3])
    image              = tf.image.random_flip_left_right(image)
    
    return image, label

def read_data_test(prefix,labels_dic, mixing, files_from_cl):
    image_list = sorted(map(lambda x: os.path.join(prefix, x),
                        filter(lambda x: x.endswith('JPEG'), files_from_cl)))
    prefix2=[]
    files_list=[]
    
    for file_i in image_list:
        tmp = file_i.split(prefix+'/')[1].split("_")[0]
        prefix2.append(tmp)
        tmp = file_i.split(prefix+'/')[1]
        files_list.append(tmp)
    
    prefix2     = np.array(prefix2)
    labels_list = np.array([mixing[labels_dic[i]] for i in prefix2])
    
    assert(len(image_list) == len(labels_list))
    images              = tf.convert_to_tensor(image_list, dtype=tf.string)
    files               = tf.convert_to_tensor(files_list, dtype=tf.string)
    labels              = tf.convert_to_tensor(labels_list, dtype=tf.int32)
    input_queue         = tf.train.slice_input_producer([images, labels,files], shuffle=False, capacity=2000)
    image_file_content  = tf.read_file(input_queue[0])
    label               = input_queue[1]
    file_string         = input_queue[2]
    image               = tf.image.resize_images(tf.image.decode_jpeg(image_file_content, channels=3), [224, 224])
    
    return image, label,file_string

def prepare_files(train_path, mixing, order, labels_dic, nb_groups, nb_cl, nb_val):
    files=os.listdir(train_path)
    prefix=[]
    
    for file_i in files:
        tmp = file_i.split("_")[0]
        prefix.append(tmp)
    
    prefix = np.array(prefix)
    labels_old=np.array([mixing[labels_dic[i]] for i in prefix])
    
    files_train = []
    files_valid = []
    
    for _ in range(nb_groups):
      files_train.append([])
      files_valid.append([])
    
    files=np.array(files)
    
    for i in range(nb_groups):
      for i2 in range(nb_cl):
        tmp_ind=np.where(labels_old == order[nb_cl*i+i2])[0]
        np.random.shuffle(tmp_ind)
        files_train[i].extend(files[tmp_ind[0:len(tmp_ind)-nb_val]])
        files_valid[i].extend(files[tmp_ind[len(tmp_ind)-nb_val:]])
    
    return files_train, files_valid
