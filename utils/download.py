"""
Helper function to download the weights associated to a given model.
It's essentially just wgetting the provided URL.
"""

import subprocess

from utils.backbones import AVAILABLE_BACKBONES


def download_model_weights(model_name, weights_directory='models'):
    """
    Downloads the weights of the model with a given name.

    :param model_name: Name of the model of which we're willing to retrieve the weights.
    """
    if model_name not in AVAILABLE_BACKBONES.keys():
        raise ValueError('Unknown weights URL for request model {}.'.format(model_name))

    subprocess.check_output(['wget',
                             AVAILABLE_BACKBONES[model_name]['weights_url'],
                             "-P",
                             weights_directory])
