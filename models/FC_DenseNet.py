from __future__ import division

import os
import time

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def preact_conv(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.2):
    """
    Basic pre-activation layer for DenseNets
    Apply successivly BatchNormalization, ReLU nonlinearity, Convolution and
    Dropout (if dropout_p > 0) on the inputs
    """
    preact = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    conv = slim.conv2d(preact,
                       n_filters,
                       kernel_size,
                       activation_fn=None,
                       normalizer_fn=None)
    if dropout_p != 0.0:
        conv = slim.dropout(conv, keep_prob=(1.0 - dropout_p))
    return conv


def dense_block(stack, n_layers, growth_rate, dropout_p, scope=None):
    """
    dense_block for DenseNet and FC-DenseNet
    Arguments:
    stack: input 4D tensor
    n_layers: number of internal layers
    growth_rate: number of feature maps per internal layer
    Returns:
    stack: current stack of feature maps (4D tensor)
    new_features: 4D tensor containing only the new feature maps generated
      in this block
    """
    with tf.name_scope(scope) as sc:
        new_features = []

        for j in range(n_layers):
            # Compute new feature maps
            layer = preact_conv(stack, growth_rate, dropout_p=dropout_p)
            new_features.append(layer)
            # Stack new layer
            stack = tf.concat([stack, layer], axis=-1)
            new_features = tf.concat(new_features, axis=-1)

    return stack, new_features


def transition_down(inputs, n_filters, dropout_p=0.2, scope=None):
    """
    Transition Down (TD) for FC-DenseNet
    Apply 1x1 BN + ReLU + conv then 2x2 max pooling
    """
    with tf.name_scope(scope) as sc:
        l = preact_conv(inputs,
                        n_filters,
                        kernel_size=[1, 1],
                        dropout_p=dropout_p)
        l = slim.pool(l, [2, 2], stride=[2, 2], pooling_type='MAX')
    return l


def transition_up(block_to_upsample,
                  skip_connection,
                  n_filters_keep,
                  scope=None):
    """
    Transition Up for FC-DenseNet
    Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection
    """
    with tf.name_scope(scope) as sc:
        # Upsample
        l = slim.conv2d_transpose(block_to_upsample,
                                  n_filters_keep,
                                  kernel_size=[3, 3],
                                  stride=[2, 2],
                                  activation_fn=None)
        # Concatenate with skip connection
        l = tf.concat([l, skip_connection], axis=-1)

    return l


def build_fc_densenet56(inputs,
                        number_of_classes,
                        weights_directory,
                        backbone_name="ResNet101",
                        is_training=True):
    return build_fc_densenet_aux(5, 12, 4, inputs, number_of_classes)


def build_fc_densenet67(inputs,
                        number_of_classes,
                        weights_directory,
                        backbone_name="ResNet101",
                        is_training=True):
    return build_fc_densenet_aux(5, 16, 5, inputs, number_of_classes)


def build_fc_densenet103(inputs,
                         number_of_classes,
                         weights_directory,
                         backbone_name="ResNet101",
                         is_training=True):
    return build_fc_densenet_aux(5, 12, [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4],
                                 inputs, number_of_classes)


def build_fc_densenet_aux(n_pool,
                          growth_rate,
                          n_layers_per_block,
                          inputs,
                          number_of_classes,
                          n_filters_first_conv=48,
                          dropout_p=0.2,
                          scope=None):
    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError

    with tf.variable_scope(scope, 'fc_densenet', [inputs]) as sc:
        stack = slim.conv2d(inputs,
                            n_filters_first_conv, [3, 3],
                            scope='first_conv',
                            activation_fn=None)

        n_filters = n_filters_first_conv
        skip_connection_list = []

        for i in range(n_pool):
            # Dense Block
            stack, _ = dense_block(stack,
                                   n_layers_per_block[i],
                                   growth_rate,
                                   dropout_p,
                                   scope='denseblock%d' % (i + 1))
            n_filters += growth_rate * n_layers_per_block[i]
            # At the end of the dense block, the current stack is stored in the skip_connections list
            skip_connection_list.append(stack)

            # Transition Down
            stack = transition_down(stack,
                                    n_filters,
                                    dropout_p,
                                    scope='transitiondown%d' % (i + 1))

        skip_connection_list = skip_connection_list[::-1]

        # Dense Block
        # We will only upsample the new feature maps
        stack, block_to_upsample = dense_block(stack,
                                               n_layers_per_block[n_pool],
                                               growth_rate,
                                               dropout_p,
                                               scope='denseblock%d' %
                                               (n_pool + 1))

        for i in range(n_pool):
            # Transition Up ( Upsampling + concatenation with the skip connection)
            n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
            stack = transition_up(block_to_upsample,
                                  skip_connection_list[i],
                                  n_filters_keep,
                                  scope='transitionup%d' % (n_pool + i + 1))

            # Dense Block
            # We will only upsample the new feature maps
            stack, block_to_upsample = dense_block(
                stack,
                n_layers_per_block[n_pool + i + 1],
                growth_rate,
                dropout_p,
                scope='denseblock%d' % (n_pool + i + 2))

        net = slim.conv2d(stack,
                          number_of_classes, [1, 1],
                          activation_fn=None,
                          scope='logits')
        return net, None
