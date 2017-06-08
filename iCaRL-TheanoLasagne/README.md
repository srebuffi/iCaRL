# iCaRL: incremental Class and Representation Learning

## Applied on CIFAR 100 using Theano + Lasagne and ResNet 32

##### Requirements
- Theano (version 0.9.0 , commit 56da8ca8775ce610595a61c3984c160aabd6ec7b tested)
- Lasagne (version 0.2.dev1 , commit 7c542651dd8e75736e02b67876fb5c9b4171b9a4 tested)
- Numpy (working with 1.11.1)
- Scipy (working with 0.18)
- CIFAR 100 downloaded 

##### Launching the code
Execute ``main_cifar_100_theano.py`` to launch the training code. Settings can easily be changed by hardcoding them in the parameters section of the code. ``eval_cifar.py`` evaluates the performances of a trained network on different groups of classes. 

PS: before running the main file, the path to the data location can be changed in ``utils_cifar100.py`` 

##### Output files
- ``top1_acc_list_cumul_icarl_clX.npy``: 3D numpy tensor for Top 1 validation cumulative accuracy on a held-out validation set. 1st dimension: each value corresponds to the cumulative accuracy after an increment of classes. 2nd dimension for different methods: iCaRL, hybrid 1 and NCM. 3rd dimension for the different runs with different orderings of classes.
- ``top1_acc_list_ori_icarl_clX.npy``: 3D numpy tensor for Top 1 accuracy on the first group of classes. 1st dimension: each value corresponds to the accuracy on the first group of classes after an increment of classes. 2nd dimension for different methods: iCaRL, hybrid 1 and NCM. 3rd dimension for the different runs with different orderings of classes.

