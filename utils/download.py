import subprocess

from utils.models import AVAILABLE_BACKBONES


def download_model_weights(model_name, weights_directory='models'):
    if model_name not in AVAILABLE_BACKBONES.keys():
        raise ValueError('Unknown weights URL for request model {}.'.format(model_name))

    subprocess.check_output(['wget',
                             AVAILABLE_BACKBONES[model_name]['weights_url'],
                             "-P",
                             weights_directory])
