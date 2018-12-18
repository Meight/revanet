import os

from models.backbones import resnet_v2
from tensorflow.contrib import slim

AVAILABLE_BACKBONES = {
    'ResNet50': {
        'arguments_scope': resnet_v2.resnet_arg_scope,
        'model': resnet_v2.resnet_v2_50,
        'scope': 'resnet_v2_50',
        'weights_url': 'https://s3.amazonaws.com/pretrained-weights/resnet_v2_50.ckpt'
    },
    'ResNet101': {
        'arguments_scope': resnet_v2.resnet_arg_scope,
        'model': resnet_v2.resnet_v2_101,
        'scope': 'resnet_v2_101',
        'weights_url': 'https://s3.amazonaws.com/pretrained-weights/resnet_v2_101.ckpt'
    },
    'ResNet152': {
        'arguments_scope': resnet_v2.resnet_arg_scope,
        'model': resnet_v2.resnet_v2_152,
        'scope': 'resnet_v2_152',
        'weights_url': 'https://s3.amazonaws.com/pretrained-weights/resnet_v2_152.ckpt'
    },
}


class BackboneBuilder:
    def __init__(self, backbone_name, weights_directory, is_training=True):
        if backbone_name not in AVAILABLE_BACKBONES:
            raise ValueError('Backbone {} is not currently available.'.format(backbone_name))

        self.backbone_name = backbone_name
        self.is_training = is_training
        self.weights_directory = weights_directory

    def build(self, inputs):
        model = AVAILABLE_BACKBONES[self.backbone_name]['model']
        scope = AVAILABLE_BACKBONES[self.backbone_name]['scope']

        logits, end_points = model(inputs,
                                   is_training=self.is_training,
                                   scope=scope)

        init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join(self.weights_directory,
                                                                      scope + '.ckpt'),
                                                 var_list=slim.get_model_variables(scope),
                                                 ignore_missing_vars=True)

        return logits, end_points, scope, init_fn
