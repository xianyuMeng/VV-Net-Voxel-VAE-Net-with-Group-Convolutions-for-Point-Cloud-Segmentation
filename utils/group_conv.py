from __future__ import print_function
from __future__ import division
from __future__ import absolute_import 

import os
import sys
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import tf_util
import permutation

def cayley(kernel):
    '''
    Args:
        kernel : 3-d tensor
    
    cayley diagram
        F-reflect-transpose-reflect-transpose
         -transpose-reflect-transpose
    '''
    
    ksize = kernel.get_shape()[-1].value
    r = lambda x : tf.matmul(
        x,
        tf.reverse(tf.eye(ksize), [-1])
    )
    
    t = lambda x : tf.transpose(x)
    
    # apply operation along depth dimension
    dsize = kernel.get_shape()[0].value
    _apply_op = lambda x, op : tf.stack([op(x[cnt])for cnt in range (dsize)])
    
    reflect = lambda x : _apply_op(x, r)
    transpose = lambda x : _apply_op(x, t)
    
    out_kernel = [
    kernel,
    reflect(kernel),
    transpose(reflect(kernel)),
    reflect(transpose(reflect(kernel))),
    transpose(reflect(transpose(reflect(kernel)))),
    transpose(kernel),
    reflect(transpose(kernel)),
    transpose(reflect(transpose(kernel)))]
    
    out_kernel = tf.stack(out_kernel, axis = 0)
    # return size = (ksize, ksize, ksize, # cayley diagram)
    return tf.transpose(out_kernel, perm = (1,2,3,0))




def group_conv_cayley(
    inputs,
    scope,
    kernel_size = 3,
    stride = [1, 1, 1],
    padding = 'VALID',
    stddev = 1e-3,
    use_xavier = True,
    weight_decay = 0.0,
    activation_fn = tf.nn.relu,
    bn = True,
    bn_decay = None,
    is_training = None):
    '''
    group convolution defined on symmetry group p4m
    Args:
    inputs : (Batch_size, D, W, H, l)
    num_input_channels : l ( i.e. the length of latent vector in RBF-VAE module)
    num_output_channels : 1 
    kernel_size : odd number
    '''
    num_in_channels = inputs.get_shape()[-1].value
    weight_shape = (kernel_size, kernel_size, kernel_size, num_in_channels, 1)
    weight = tf_util._variable_with_weight_decay(
        name = scope +  '/gconv_weights',
        shape = weight_shape,
        stddev = stddev,
        wd = weight_decay,
        use_xavier = use_xavier)
    weight_list = tf.split(weight, num_in_channels, axis = -2)
    weight_list = [tf.squeeze(ww) for ww in weight_list]
    # weight_list : list of (ksize, ksize, ksize) tensor

    permute_order = permutation.VVInt()	
    permute_in = permutation.VInt([cnt for cnt in range(kernel_size)])
    permutation.permute(permute_in, kernel_size, permute_order)
    permute_order = list(list(pp) for pp in permute_order)

    gconv_kernel = []
    for pp in permute_order:
        gconv_kernel.append([cayley(tf.transpose(weight_list[cnt], pp)) for cnt in range(num_in_channels)])


    # gconv_kernel -> list [6][num_in_channels][(ksize, ksize, ksize, #cayley diagram)]
    gconv_kernel = [
        tf.stack(gconv_kernel[cnt], axis = -2) for cnt in range(len(permute_order))
    ]
    # gconv_kernel -> list [6][(ksize, ksize, ksize,num_in_channels, #cayley diagram)] 
    gconv_kernel = tf.concat(gconv_kernel, axis = -1)

    # gconv_kernel -> tensor of shape (ksize, ksize, num_in_channels, 2 * #cayley diagram)

    bias = tf.get_variable(
        name = scope + "/bias",
        shape = [1],
        initializer = tf.constant_initializer(0.0))
    bias_share = tf.reshape(
        tf.stack([bias for cnt in range(2 * 8)]),
        [-1])

    output = tf.nn.conv3d(
        input = inputs,
        filter = gconv_kernel,
        strides = (1, stride[0], stride[1], stride[2], 1),
        padding = padding)
    if bn:
        output = tf_util.batch_norm_for_conv3d(
            output,
            is_training = is_training,
            bn_decay = bn_decay,
            scope = "bn")
        output = tf.nn.bias_add(output, bias_share)
        if activation_fn is not None:
            output = activation_fn(output)

    return output





