import tensorflow as tf
from tensorflow.contrib import slim
from utils.backbones import BackboneBuilder


def conv_upscale_block(inputs, n_filters, kernel_size=[3, 3], scale=2):
    """
    Basic deconv block for GCN
    Apply Transposed Convolution for feature map upscaling
    """
    net = slim.conv2d_transpose(inputs,
                                n_filters,
                                kernel_size=[3, 3],
                                stride=[2, 2],
                                activation_fn=None)
    return net


def boundary_refinement_block(inputs, n_filters, kernel_size=[3, 3]):
    """
    Boundary Refinement Block for GCN
    """
    net = slim.conv2d(inputs,
                      n_filters,
                      kernel_size,
                      activation_fn=None,
                      normalizer_fn=None)
    net = tf.nn.relu(net)
    net = slim.conv2d(net,
                      n_filters,
                      kernel_size,
                      activation_fn=None,
                      normalizer_fn=None)
    net = tf.add(inputs, net)
    return net


def global_conv_block(inputs, n_filters=21, size=3):
    """
    Global Conv Block for GCN
    """

    net_1 = slim.conv2d(inputs,
                        n_filters, [size, 1],
                        activation_fn=None,
                        normalizer_fn=None)
    net_1 = slim.conv2d(net_1,
                        n_filters, [1, size],
                        activation_fn=None,
                        normalizer_fn=None)

    net_2 = slim.conv2d(inputs,
                        n_filters, [1, size],
                        activation_fn=None,
                        normalizer_fn=None)
    net_2 = slim.conv2d(net_2,
                        n_filters, [size, 1],
                        activation_fn=None,
                        normalizer_fn=None)

    net = tf.add(net_1, net_2)

    return net


def build_gcn(inputs,
              number_of_classes,
              weights_directory,
              backbone_name="ResNet101",
              weight_decay=1e-5,
              is_training=True,
              upscaling_method="bilinear"):
    logits, end_points, scope, init_fn = BackboneBuilder(
        backbone_name=backbone_name,
        is_training=is_training,
        weights_directory=weights_directory).build(inputs=inputs)

    res = [
        end_points['pool5'], end_points['pool4'], end_points['pool3'],
        end_points['pool2']
    ]

    down_5 = global_conv_block(res[0], n_filters=21, size=3)
    down_5 = boundary_refinement_block(down_5,
                                       n_filters=21,
                                       kernel_size=[3, 3])
    down_5 = conv_upscale_block(down_5,
                                n_filters=21,
                                kernel_size=[3, 3],
                                scale=2)

    down_4 = global_conv_block(res[1], n_filters=21, size=3)
    down_4 = boundary_refinement_block(down_4,
                                       n_filters=21,
                                       kernel_size=[3, 3])
    down_4 = tf.add(down_4, down_5)
    down_4 = boundary_refinement_block(down_4,
                                       n_filters=21,
                                       kernel_size=[3, 3])
    down_4 = conv_upscale_block(down_4,
                                n_filters=21,
                                kernel_size=[3, 3],
                                scale=2)

    down_3 = global_conv_block(res[2], n_filters=21, size=3)
    down_3 = boundary_refinement_block(down_3,
                                       n_filters=21,
                                       kernel_size=[3, 3])
    down_3 = tf.add(down_3, down_4)
    down_3 = boundary_refinement_block(down_3,
                                       n_filters=21,
                                       kernel_size=[3, 3])
    down_3 = conv_upscale_block(down_3,
                                n_filters=21,
                                kernel_size=[3, 3],
                                scale=2)

    down_2 = global_conv_block(res[3], n_filters=21, size=3)
    down_2 = boundary_refinement_block(down_2,
                                       n_filters=21,
                                       kernel_size=[3, 3])
    down_2 = tf.add(down_2, down_3)
    down_2 = boundary_refinement_block(down_2,
                                       n_filters=21,
                                       kernel_size=[3, 3])
    down_2 = conv_upscale_block(down_2,
                                n_filters=21,
                                kernel_size=[3, 3],
                                scale=2)

    net = boundary_refinement_block(down_2, n_filters=21, kernel_size=[3, 3])
    net = conv_upscale_block(net, n_filters=21, kernel_size=[3, 3], scale=2)
    net = boundary_refinement_block(net, n_filters=21, kernel_size=[3, 3])

    net = slim.conv2d(net,
                      number_of_classes, [1, 1],
                      activation_fn=None,
                      scope='logits')

    return net, init_fn
