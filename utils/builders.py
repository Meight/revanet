from pathlib import Path

from tensorflow.contrib import slim
from frontends import resnet_v2
import os

from models.BiSeNet import build_bisenet

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

AVAILABLE_MODELS = {
    'BiSeNet': build_bisenet
}


class BackboneBuilder:
    def __init__(self, backbone_name, is_training=True, weights_directory='models'):
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
        """
        Builds a model using the backbone the builder was provided with when instantiated.

        :param model_name:  The name of the model to build.
        :param inputs:      The inputs of the built model.
        :return:            The network and its initialization function.
        """
        if model_name not in AVAILABLE_MODELS.keys():
            raise ValueError('Requested model {} is not available.'.format(model_name))

        self.download_backbone_weights(backbone_name=self.backbone_name)

        return AVAILABLE_MODELS[model_name](inputs,
                                            number_of_classes=self.number_of_classes,
                                            preset_model=model_name,
                                            backbone_name=self.backbone_name,
                                            is_training=self.is_training)

    def download_backbone_weights(self, backbone_name, only_if_not_exists=True):
        """
        Downloads the pretrained weights for the requested model and saves them into the weights directory
        provided to the builder's instance.

        :param backbone_name:       The name of the model of which to retrieve the pretrained weights.
        :param only_if_not_exists:  Whether or not to download the weights if they already exist in the weights
                                    directory.
        """
        if only_if_not_exists and not Path(self.weights_directory).exists():
            from utils.download import download_model_weights
            download_model_weights(model_name=backbone_name, weights_directory=self.weights_directory)
