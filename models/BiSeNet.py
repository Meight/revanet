import tensorflow as tf
from tensorflow.contrib import slim
from utils.backbones import BackboneBuilder


def upsampling(inputs, scale):
    return tf.image.resize_bilinear(
        inputs,
        size=[tf.shape(inputs)[1] * scale,
              tf.shape(inputs)[2] * scale])


def conv_upscale_block(inputs, n_filters, kernel_size=None, scale=2):
    if kernel_size is None:
        kernel_size = [3, 3]

    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d_transpose(net,
                                n_filters,
                                kernel_size=[3, 3],
                                stride=[scale, scale],
                                activation_fn=None)
    return net


def conv2d_block(inputs, n_filters, kernel_size=None, strides=1):
    if kernel_size is None:
        kernel_size = [3, 3]

    net = slim.conv2d(inputs,
                      n_filters,
                      kernel_size,
                      stride=[strides, strides],
                      activation_fn=None,
                      normalizer_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    return net


def attention_refinement_module(inputs, n_filters):
    # Global average pooling.
    net = tf.reduce_mean(inputs, [1, 2], keepdims=True)

    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = slim.batch_norm(net, fused=True)
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    return net


def feature_fusion_module(input_1, input_2, n_filters):
    inputs = tf.concat([input_1, input_2], axis=-1)
    inputs = conv2d_block(inputs, n_filters=n_filters, kernel_size=[3, 3])

    # Global average pooling
    net = tf.reduce_mean(inputs, [1, 2], keepdims=True)

    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.sigmoid(net)

    net = tf.multiply(inputs, net)

    net = tf.add(inputs, net)

    return net


def build_bisenet(inputs,
                  number_of_classes,
                  weights_directory,
                  backbone_name="ResNet101",
                  is_training=True):
    logits, end_points, scope, init_fn = BackboneBuilder(
        backbone_name=backbone_name,
        is_training=is_training,
        weights_directory=weights_directory).build(inputs=inputs)

    # Context path.
    net_4 = attention_refinement_module(end_points['pool4'], n_filters=512)

    net_5 = attention_refinement_module(end_points['pool5'], n_filters=2048)

    global_channels = tf.reduce_mean(net_5, [1, 2], keepdims=True)
    net_5_scaled = tf.multiply(global_channels, net_5)

    # Spatial path.
    spatial_net = conv2d_block(inputs,
                               n_filters=64,
                               kernel_size=[3, 3],
                               strides=2)
    spatial_net = conv2d_block(spatial_net,
                               n_filters=128,
                               kernel_size=[3, 3],
                               strides=2)
    spatial_net = conv2d_block(spatial_net,
                               n_filters=256,
                               kernel_size=[3, 3],
                               strides=2)

    # Path combination.
    net_4 = upsampling(net_4, scale=2)
    net_5_scaled = upsampling(net_5_scaled, scale=4)

    context_net = tf.concat([net_4, net_5_scaled], axis=-1)

    net = feature_fusion_module(input_1=spatial_net,
                                input_2=context_net,
                                n_filters=number_of_classes)
    net = upsampling(net, scale=8)

    net = slim.conv2d(net,
                      number_of_classes, [1, 1],
                      activation_fn=None,
                      scope='logits')

    return net, init_fn
