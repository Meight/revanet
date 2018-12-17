from pathlib import Path

from tensorflow.contrib import slim
from frontends import resnet_v2
import os

from models.BiSeNet import build_bisenet

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

available_models = {
    'BiSeNet': build_bisenet
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


class ModelBuilder:
    def __init__(self,
                 number_of_classes,
                 input_size,
                 backbone_name="ResNet101",
                 is_training=True,
                 weights_directory='model'):
        self.input_size = input_size
        self.number_of_classes = number_of_classes
        self.is_training = is_training
        self.backbone_name = backbone_name
        self.weights_directory = weights_directory

    def build(self, model_name, inputs):
        if model_name not in available_models.keys():
            raise ValueError('Requested model {} is not available.'.format(model_name))

        self.download_backbone_weights(backbone_name=self.backbone_name)

        return available_models[model_name](inputs,
                                            number_of_classes=self.number_of_classes,
                                            preset_model=model_name,
                                            backbone_name=self.backbone_name,
                                            is_training=self.is_training)

    def download_backbone_weights(self, backbone_name, only_if_not_exists=True):
        if only_if_not_exists and not Path(self.weights_directory).exists():
            from utils.download import download_model_weights
            download_model_weights(model_name=backbone_name, weights_directory=self.weights_directory)
