from pathlib import Path

from models.BiSeNet import build_bisenet
from models.DeepLabv3_plus import build_deeplabv3_plus
from models.GCN import build_gcn

AVAILABLE_MODELS = {
    'BiSeNet': build_bisenet,
    'DeepLabv3_plus': build_deeplabv3_plus,
    'GCN': build_gcn
}


class ModelBuilder:
    def __init__(self,
                 number_of_classes,
                 input_size,
                 backbone_name="ResNet101",
                 is_training=True,
                 weights_directory='models/checkpoints'):
        self.input_size = input_size
        self.number_of_classes = number_of_classes
        self.is_training = is_training
        self.backbone_name = backbone_name
        self.weights_directory = weights_directory

    def build(self, model_name, inputs):
        """
        Builds a model using the backbone the builder was provided with when
        instantiated.

        :param model_name:  The name of the model to build.
        :param inputs:      The inputs of the built model.
        :return:            The network and its initialization function.
        """
        if model_name not in AVAILABLE_MODELS.keys():
            raise ValueError(
                'Requested model {} is not available.'.format(model_name))

        self.download_backbone_weights(backbone_name=self.backbone_name)

        return AVAILABLE_MODELS[model_name](
            inputs=inputs,
            number_of_classes=self.number_of_classes,
            backbone_name=self.backbone_name,
            is_training=self.is_training,
            weights_directory=self.weights_directory)

    def download_backbone_weights(self, backbone_name,
                                  only_if_not_exists=True):
        """
        Downloads the pretrained weights for the requested model and saves them into the weights directory
        provided to the builder's instance.

        :param backbone_name:       The name of the model of which to retrieve the pretrained weights.
        :param only_if_not_exists:  Whether or not to download the weights if they already exist in the weights
                                    directory.
        """
        if only_if_not_exists and not Path(self.weights_directory).exists():
            from utils.download import download_model_weights
            download_model_weights(model_name=backbone_name,
                                   weights_directory=self.weights_directory)
