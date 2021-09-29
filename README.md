# iCaRL: Incremental Classifier and Representation Learning

Tensorflow and Theano + Lasagne codes for the paper https://arxiv.org/abs/1611.07725

## Disclaimer

The code is now very out-dated and not supported by the current Tensorflow so this repo should be considered as an indication on how we coded iCaRL rather than a runnable code.

## Abstract 

A major open problem on the road to artificial intelligence is the development of incrementally learning systems that learn about more and more concepts over time from a stream of data. In this work, we introduce a new training strategy, iCaRL, that allows learning in such a class-incremental way: only the training data for a small number of classes has to be present at the same time and new classes can be added progressively. iCaRL learns strong classifiers and a data representation simultaneously. This distinguishes it from earlier works that were fundamentally limited to fixed data representations and therefore incompatible with deep learning architectures. We show by experiments on CIFAR-100 and ImageNet ILSVRC 2012 data that iCaRL can learn many classes incrementally over a long period of time where other strategies quickly fail.

## If you consider citing us

    @inproceedings{ rebuffi-cvpr2017,
       author = { Sylvestre-Alvise Rebuffi and Alexander Kolesnikov and Georg Sperl and Christoph H. Lampert },
       title = {{iCaRL:} Incremental Classifier and Representation Learning},
       booktitle = CVPR,
       year = 2017,
    }
