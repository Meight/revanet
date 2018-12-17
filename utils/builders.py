from tensorflow.contrib import slim
from frontends import resnet_v2
import os


available_backbones = {
    'ResNet50': {
        'arguments_scope': resnet_v2.resnet_arg_scope,
        'model': resnet_v2.resnet_v2_50,
        'scope': 'resnet_v2_50',
    },
    'ResNet101': {
        'arguments_scope': resnet_v2.resnet_arg_scope,
        'model': resnet_v2.resnet_v2_101,
        'scope': 'resnet_v2_101',
    },
    'ResNet152': {
        'arguments_scope': resnet_v2.resnet_arg_scope,
        'model': resnet_v2.resnet_v2_152,
        'scope': 'resnet_v2_152',
    },
}


class BackboneBuilder:
    def __init__(self, backbone_name, is_training=True, weights_directory='models'):
        if backbone_name not in available_backbones:
            raise ValueError('Backbone {} is not currently available.'.format(backbone_name))

        self.backbone_name = backbone_name
        self.is_training = is_training
        self.weights_directory = weights_directory

    def build(self, inputs):
        model = available_backbones[self.backbone_name]['model']
        scope = available_backbones[self.backbone_name]['scope']

        logits, end_points = model(inputs,
                                   is_training=self.is_training,
                                   scope=scope)

        init_fn = slim.assign_from_checkpoint(model_path=os.path.join(self.weights_directory,
                                                                      scope + '.ckpt'),
                                              var_list=slim.get_model_variables(scope),
                                              ignore_missing_vars=True)

        return logits, end_points, scope, init_fn