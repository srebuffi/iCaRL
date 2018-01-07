import tensorflow as tf
import numpy as np
import cPickle


def relu(x, name, alpha):
    if alpha > 0:
        return tf.maximum(alpha * x, x, name=name)
    else:
        return tf.nn.relu(x, name=name)


def get_variable(name, shape, dtype, initializer, trainable=True, regularizer=None):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape=shape, dtype=dtype,
                              initializer=initializer, regularizer=regularizer, trainable=trainable,
                              collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
    return var


def conv(inp, name, size, out_channels, strides=[1, 1, 1, 1],
         dilation=None, padding='SAME', apply_relu=True, alpha=0.0,bias=True,
         initializer=tf.contrib.layers.xavier_initializer_conv2d()):
    
    batch_size = inp.get_shape().as_list()[0]
    res1 = inp.get_shape().as_list()[1]
    res2 = inp.get_shape().as_list()[1]
    in_channels = inp.get_shape().as_list()[3]
    
    with tf.variable_scope(name):
        W = get_variable("W", shape=[size, size, in_channels, out_channels], dtype=tf.float32,
                         initializer=initializer, regularizer=tf.nn.l2_loss)
        b = get_variable("b", shape=[1, 1, 1, out_channels], dtype=tf.float32,
                         initializer=tf.zeros_initializer(),trainable=bias)
        
        if dilation:
            assert(strides == [1, 1, 1, 1])
            out = tf.add(tf.nn.atrous_conv2d(inp, W, rate=dilation, padding=padding), b, name='convolution')
            out.set_shape([batch_size, res1, res2, out_channels])
        else:
            out = tf.add(tf.nn.conv2d(inp, W, strides=strides, padding=padding), b, name='convolution')
        
        if apply_relu:
            out = relu(out, alpha=alpha, name='relu')
    
    return out


def softmax(target, axis, name=None):
    max_axis = tf.reduce_max(target, axis, keep_dims=True)
    target_exp = tf.exp(target - max_axis)
    normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
    softmax = target_exp / normalize
    return softmax


def batch_norm(inp, name, phase, decay=0.9):
    
    channels = inp.get_shape().as_list()[3]
    
    with tf.variable_scope(name):
        moving_mean = get_variable("mean", shape=[channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0), trainable=False)
        moving_variance = get_variable("var", shape=[channels], dtype=tf.float32, initializer=tf.constant_initializer(1.0), trainable=False)
        
        offset = get_variable("offset", shape=[channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        scale = get_variable("scale", shape=[channels], dtype=tf.float32, initializer=tf.constant_initializer(1.0), regularizer=tf.nn.l2_loss)
        
        mean, variance = tf.nn.moments(inp, axes=[0, 1, 2], shift=moving_mean)
        
        mean_op = moving_mean.assign(decay * moving_mean + (1 - decay) * mean)
        var_op = moving_variance.assign(decay * moving_variance + (1 - decay) * variance)
        
        assert(phase in ['train', 'test'])
        if phase == 'train':
            with tf.control_dependencies([mean_op, var_op]):
                return tf.nn.batch_normalization(inp, mean, variance, offset, scale, 0.01, name='norm')
        else:
            return tf.nn.batch_normalization(inp, moving_mean, moving_variance, offset, scale, 0.01, name='norm')


def pool(inp, name, kind, size, stride, padding='SAME'):
    
    assert kind in ['max', 'avg']
    
    strides = [1, stride, stride, 1]
    sizes = [1, size, size, 1]
    
    with tf.variable_scope(name):
        if kind == 'max':
            out = tf.nn.max_pool(inp, sizes, strides=strides, padding=padding, name=kind)
        else:
            out = tf.nn.avg_pool(inp, sizes, strides=strides, padding=padding, name=kind)
    
    return out


def ResNet18(inp, phase, num_outputs=1000, alpha=0.0):
    def residual_block(inp, phase, alpha=0.0,nom='a',increase_dim=False,last=False):
        input_num_filters = inp.get_shape().as_list()[3]
        if increase_dim:
            first_stride = [1, 2, 2, 1]
            out_num_filters = input_num_filters*2
        else:
            first_stride = [1, 1, 1, 1]
            out_num_filters = input_num_filters
        
        layer = conv(inp, 'resconv1'+nom, size=3, strides=first_stride, out_channels=out_num_filters, alpha=alpha, padding='SAME')
        layer = batch_norm(layer, 'batch_norm_resconv1'+nom, phase=phase)
        layer = conv(layer, 'resconv2'+nom, size=3, strides=[1, 1, 1, 1], out_channels=out_num_filters, apply_relu=False,alpha=alpha, padding='SAME')
        layer = batch_norm(layer, 'batch_norm_resconv2'+nom, phase=phase)
        
        if increase_dim:
                projection = conv(inp, 'projconv'+nom, size=1, strides=[1, 2, 2, 1], out_channels=out_num_filters, alpha=alpha, apply_relu=False,padding='SAME',bias=False)
                projection = batch_norm(projection, 'batch_norm_projconv'+nom, phase=phase)
                if last:
                    block = layer + projection
                else:
                    block = layer + projection
                    block = tf.nn.relu(block, name='relu')
        else:
            if last:
                block = layer + inp
            else:
                block = layer + inp
                block = tf.nn.relu(block, name='relu')
        
        return block
    
    # First conv
    #layer = batch_norm(inp, 'batch_norm_0', phase=phase)
    layer = conv(inp,"conv1",size=7,strides=[1, 2, 2, 1], out_channels=64, alpha=alpha, padding='SAME')
    layer = batch_norm(layer, 'batch_norm_1', phase=phase)
    layer = pool(layer, 'pool1', 'max', size=3, stride=2)
    
    # First stack of residual blocks
    for letter in 'ab':
        layer = residual_block(layer, phase, alpha=0.0,nom=letter)
    
    # Second stack of residual blocks
    layer = residual_block(layer, phase, alpha=0.0,nom='c',increase_dim=True)
    for letter in 'd':
        layer = residual_block(layer, phase, alpha=0.0,nom=letter)
    
    # Third stack of residual blocks
    layer = residual_block(layer, phase, alpha=0.0,nom='e',increase_dim=True)
    for letter in 'f':
        layer = residual_block(layer, phase, alpha=0.0,nom=letter)
    
    # Fourth stack of residual blocks
    layer = residual_block(layer, phase, alpha=0.0,nom='g',increase_dim=True)
    layer = residual_block(layer, phase, alpha=0.0,nom='h',increase_dim=False,last=True)
    
    layer = pool(layer, 'pool_last', 'avg', size=7, stride=1,padding='VALID')
    layer = conv(layer, name='fc', size=1, out_channels=num_outputs, padding='VALID', apply_relu=False, alpha=alpha)[:, 0, 0, :]
    
    return layer


def get_weight_initializer(params):
    
    initializer = []
    
    scope = tf.get_variable_scope()
    scope.reuse_variables()
    for layer, value in params.items():
        op = tf.get_variable('%s' % layer).assign(value)
        initializer.append(op)
    return initializer


def save_model(name, scope, sess):
    variables = tf.get_collection(tf.GraphKeys.WEIGHTS, scope=scope)
    d = [(v.name.split(':')[0], sess.run(v)) for v in variables]
    
    cPickle.dump(d, open(name, 'wb'))
