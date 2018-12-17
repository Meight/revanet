import subprocess

weights_url = {
    'ResNet50': 'https://s3.amazonaws.com/pretrained-weights/resnet_v2_50.ckpt',
    'ResNet101': 'https://s3.amazonaws.com/pretrained-weights/resnet_v2_101.ckpt',
    'ResNet152': 'https://s3.amazonaws.com/pretrained-weights/resnet_v2_152.ckpt',
}


def download_model_weights(model_name, weights_directory='models'):
    if model_name not in weights_url.keys():
        raise ValueError('Unknown weights URL for request model {}.'.format(model_name))

    subprocess.check_output(['wget',
                             'https://s3.amazonaws.com/pretrained-weights/resnet_v2_50.ckpt',
                             "-P",
                             weights_directory])
