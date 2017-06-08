# iCaRL: incremental Class and Representation Learning

## Applied on ImageNet using Tensorflow and ResNet 18

##### Requirements
- Tensorflow (version 1.1)
- Scipy (working with 0.19)
- ImageNet downloaded with a train folder containing directly all the images (no subfolder for each class)

#### Training

##### Launching the code
Execute ``main_resnet_tf.py`` to launch the code. Settings can easily be changed by hardcoding them in the parameters section of the code.

PS: if your data presents a different folder architecture, you can change it in ``utils_data.py``

##### Output files
- ``mixing.pickle``: ordering of the classes after the initial random mixing 
- ``settings_resnet.pickle``: images file names organized in batches of classes and validation/training separation
- ``model-iteration-i.pickle``: storing of the network parameters after training each increment of classes
- ``Xclass_means.pickle`` : 4D tensor. Xclass_means[:,i,0,j] corresponds to the mean-of-examplars of the class i after the j-th increment of classes. Xclass_means[:,i,1,j] corresponds to the corresponding the theoretical class mean used by the NCM.
- ``Xfiles_protoset.pickle`` : list with Xfiles_protoset[i] containing the ranked exemplars for the class i. At each increment of classes, a smaller subset of the exemplars of the previous classes is used as the memory is fixed.

##### Nota Bene
- It is possible to use any other standard network architecture: just code the architecture in the utils_resnet.py file and replace 'ResNet18' in the other files by the name of the new architecture. 
- To test if the code works for you, set the number of epochs to 1 and you should get around 48 % trainings set accuracy displayed on the screen during the first epoch of the first group and around 20 % during the first epoch of the second group

#### Testing

##### Launching the code
As we save the classifier, exemplars and weights after each increment, we can evaluate and compare the performances after each increment of classes. Execute ``valid_resnet.py`` if you want the cumulative performances after each increment. Or, execute ``valid_resnet2.py`` if you want to choose the state of the network (how many increments already done) and on which batches of classes the classifier should be evaluated. Settings can easily be changed by hardcoding them in the parameters section of the code.

##### Output file
- ``results_topX_acc_Y_clZ.npy``: accuracy file with each line corresponding to an increment. 1st column is with iCaRL, 2nd column with Hybrid 1 and 3rd column is the theoretical case of NCM.

##### Nota Bene
Here the validation code evaluates the performances on the validation files chosen randomly in the training code as a held-out set.
