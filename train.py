from __future__ import print_function

import argparse
import os
import random
import time

import numpy as np
import tensorflow as tf

from builders import model_builder
from utils import utils, segmentation
from utils.augmentation import augment_data
from utils.files import retrieve_dataset_information
from utils.naming import FilesFormatterFactory
from utils.utils import build_images_association_dictionary, gather_multi_label_data, \
    get_available_annotation_resized_tensors_for_image
from utils.validation import SegmentationEvaluator


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--is-multi-label',
                    action='store_true',
                    default=False,
                    help='Whether or not to interpret the task as multi-label classification.')
parser.add_argument('--learning-rate',
                    type=float,
                    default=0.0001,
                    help='Learning rate to use')
parser.add_argument('--number-of-epochs',
                    type=int,
                    default=300,
                    help='Number of epochs to train for')
parser.add_argument('--first-epoch',
                    type=int,
                    default=0,
                    help='Start counting epochs from this number')
parser.add_argument('--save-weights-every',
                    type=int,
                    default=5,
                    help='How often to save checkpoints (epochs)')
parser.add_argument('--validate-every',
                    type=int,
                    default=1,
                    help='How often to perform validation (epochs)')
parser.add_argument('--image',
                    type=str,
                    default=None,
                    help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue-training',
                    action='store_true',
                    default=False,
                    help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset-name',
                    type=str,
                    default="CamVid",
                    help='Dataset you are using.')
parser.add_argument('--input-size',
                    type=int,
                    default=512,
                    help='Box six of input image to network')
parser.add_argument('--batch-size',
                    type=int,
                    default=1,
                    help='Number of images in each batch')
parser.add_argument('--number-of-validation-images',
                    type=int,
                    default=20,
                    help='The number of images to used for validations')
parser.add_argument('--model-name',
                    type=str,
                    default="FC-DenseNet56",
                    help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--backbone',
                    type=str,
                    default="ResNet101",
                    help='The backbone to use. See frontend_builder.py for supported models')
parser.add_argument('--results-directory',
                    type=str,
                    default='/projets/thesepizenberg/deep-learning/segmentation-suite',
                    help='Path to the directory where the results are to be stored.')
args = parser.parse_args()

IS_MULTI_LABEL_CLASSIFICATION: bool(args.is_multi_label)
INPUT_SIZE = int(args.input_size)

# No augmentation available for multi-label classification.
is_dataset_augmented = not IS_MULTI_LABEL_CLASSIFICATION and (args.h_flip
                                                              or args.v_flip
                                                              or (args.brightness is not None)
                                                              or (args.rotation is not None))

DATASET_NAME = str(args.dataset_name)
RESULTS_DIRECTORY = str(args.results_directory)
MODEL_NAME = str(args.model_name)
BACKBONE_NAME = str(args.backbone_name)

NUMBER_OF_EPOCHS = int(args.number_of_epochs)
BATCH_SIZE = int(args.batch_size)
LEARNING_RATE = float(args.learning_rate)
NUMBER_OF_VALIDATION_IMAGES = int(args.number_of_validation_images)
SAVE_WEIGHTS_EVERY = int(args.save_weights_every)
VALIDATE_EVERY = int(args.validate_every)

RANDOM_SEED = 2018
PRINT_INFO_EVERY = 30  # Period (in epochs) of prints.

TRAINING_PARAMETERS = {
    'model': MODEL_NAME,
    'epochs': NUMBER_OF_EPOCHS,
    'learning_rate': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'validation_steps': NUMBER_OF_VALIDATION_IMAGES,
    'input_size': INPUT_SIZE,
    'augmented': is_dataset_augmented
}

validation_measures = ['scores', 'scores_per_class', 'precision', 'recall', 'f1', 'iou']

files_formatter_factory = FilesFormatterFactory(mode='training',
                                                dataset_name=DATASET_NAME,
                                                model_name=MODEL_NAME,
                                                backbone_name=BACKBONE_NAME,
                                                training_parameters=TRAINING_PARAMETERS,
                                                verbose=True,
                                                results_folder=RESULTS_DIRECTORY)

DATASET_INFORMATION_FILE_PATH = os.path.join(args.dataset, "information.csv")
class_names_list, class_colors = retrieve_dataset_information(dataset_path=DATASET_INFORMATION_FILE_PATH)
class_colors_dictionary = dict(zip(class_names_list, class_colors))
number_of_classes = len(class_colors)

segmentation_evaluator = SegmentationEvaluator(validation_measures, number_of_classes)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Compute your softmax cross entropy loss
input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3])
output_tensor = tf.placeholder(tf.float32, shape=[None, None, None, number_of_classes])

predictions_tensor, init_fn = model_builder.build_model(model_name=MODEL_NAME,
                                                        frontend=BACKBONE_NAME,
                                                        net_input=input_tensor,
                                                        num_classes=number_of_classes,
                                                        crop_width=INPUT_SIZE,
                                                        crop_height=INPUT_SIZE,
                                                        is_training=True)

weights_shape = (BATCH_SIZE, INPUT_SIZE, INPUT_SIZE)
unc = tf.where(tf.equal(tf.reduce_sum(output_tensor, axis=-1), 0),
               tf.zeros(shape=weights_shape),
               tf.ones(shape=weights_shape))

if not IS_MULTI_LABEL_CLASSIFICATION:
    print('Using softmax cross entropy.')
    adapted_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions_tensor,
                                                              labels=output_tensor)
else:
    print('Using sigmoid cross entropy.')
    adapted_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions_tensor,
                                                           labels=output_tensor)

loss = tf.reduce_mean(tf.losses.compute_weighted_loss(weights=tf.cast(unc, tf.float32),
                                                      losses=adapted_loss))

opt = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE,
                                decay=0.995,
                                momentum=0.9).minimize(loss,
                                                       var_list=[var for var in tf.trainable_variables()])

session.run(tf.global_variables_initializer())

utils.count_parameters()

if init_fn is not None:
    init_fn(session)

model_checkpoint_name = "checkpoints/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
checkpoint_formatter = files_formatter_factory.get_checkpoint_formatter(saver=tf.train.Saver(max_to_keep=1000))
summary_formatter = files_formatter_factory.get_summary_formatter()

if args.continue_training:
    print('Loaded latest model checkpoint.')
    checkpoint_formatter.restore(session, model_checkpoint_name)

train_output_names, validation_output_names, test_output_names = None, None, None
paths = None

if not IS_MULTI_LABEL_CLASSIFICATION:
    train_input_names, train_output_names, validation_input_names, \
    validation_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)
else:
    paths = gather_multi_label_data(dataset_directory=DATASET_NAME)

    train_input_names = paths['train'].keys()
    validation_input_names = paths['val'].keys()
    test_input_names = paths['test'].keys()

average_measures_per_epoch = {
    'loss': [],
    'iou': [],
    'scores': []
}

random.seed(RANDOM_SEED)
validation_images_count = min(NUMBER_OF_VALIDATION_IMAGES, len(validation_input_names))
validation_indices = random.sample(range(len(validation_input_names)), validation_images_count)

if not IS_MULTI_LABEL_CLASSIFICATION:
    images_association = build_images_association_dictionary(train_input_names, train_output_names)
else:
    images_association = None

assert IS_MULTI_LABEL_CLASSIFICATION or images_association is not None

for epoch in range(args.epoch_start_i, args.num_epochs):
    current_losses = []
    samples_seen = 0

    # Equivalent to shuffling
    input_indices = np.random.permutation(len(train_input_names))
    number_of_training_images = len(train_input_names)

    number_of_steps = len(train_output_names) // BATCH_SIZE
    start_time = time.time()
    epoch_start_time = time.time()
    for current_step_index in range(number_of_steps):
        input_image_batch = []
        output_image_batch = []

        # Collect a batch of images.
        for current_batch_index in range(BATCH_SIZE):
            current_index = (current_step_index * BATCH_SIZE + current_batch_index) % number_of_training_images
            input_image_index = input_indices[current_index]

            input_image_name = train_input_names[input_image_index]
            input_image = utils.load_image(input_image_name)

            if not IS_MULTI_LABEL_CLASSIFICATION:
                output_image_name = random.choice(images_association[input_image_name])
                output_image = utils.load_image(output_image_name)
            else:
                n_encoded_masks = get_available_annotation_resized_tensors_for_image((INPUT_SIZE, INPUT_SIZE),
                                                                                     paths['train'][input_image_index],
                                                                                     class_colors_dictionary)
                output_image = random.choice(n_encoded_masks)

            with tf.device('/cpu:0'):
                if not IS_MULTI_LABEL_CLASSIFICATION:
                    input_image, output_image = augment_data(input_image, output_image, input_size=INPUT_SIZE)
                else:
                    # Output tensor is already resized at this point, only resize input image.
                    input_image, _ = utils.resize_to_size(INPUT_SIZE, desired_size=INPUT_SIZE)

                input_image = np.float32(input_image) / 255.0
                output_image = np.float32(segmentation.image_to_one_hot(annotation=output_image,
                                                                        class_colors=class_colors))

                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image_batch.append(np.expand_dims(output_image, axis=0))

        if BATCH_SIZE == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

        # Perform training.
        _, current = session.run([opt, loss],
                                 feed_dict={input_tensor: input_image_batch, output_tensor: output_image_batch})
        current_losses.append(current)
        samples_seen += BATCH_SIZE

        if samples_seen % PRINT_INFO_EVERY == 0:
            print('[{} - {} #{}] Seen samples: {}, current loss: {}, time spent on batch: {}'.format(
                args.model, args.frontend, epoch, samples_seen, current, time.time() - start_time))
            start_time = time.time()

    average_measures_per_epoch['loss'].append(np.mean(current_losses))

    if validation_indices != 0 and epoch % SAVE_WEIGHTS_EVERY == 0:
        checkpoint_formatter.save(session=session,
                                  current_epoch=epoch)

    if epoch % VALIDATE_EVERY == 0:
        segmentation_evaluator.initialize_history()

        for image_index in validation_indices:
            input_image = np.float32(utils.load_image(validation_input_names[image_index]))
            ground_truth = utils.load_image(validation_output_names[image_index])
            input_image, ground_truth = utils.resize_to_size(input_image, ground_truth, desired_size=INPUT_SIZE)

            input_image = np.expand_dims(input_image, axis=0) / 255.0
            ground_truth = segmentation.one_hot_to_image(segmentation.image_to_one_hot(ground_truth, class_colors))

            valid_indices = np.where(np.sum(ground_truth, axis=-1) != 0)
            ground_truth = ground_truth[valid_indices, :]

            output_image = session.run(predictions_tensor, feed_dict={input_tensor: input_image})

            output_image = np.array(output_image[0, :, :, :])
            output_image = output_image[valid_indices, :]
            output_image = segmentation.one_hot_to_image(output_image)

            segmentation_evaluator.evaluate(prediction=output_image,
                                            annotation=ground_truth)

            file_name = utils.file_path_to_name(validation_input_names[image_index])

        summary_formatter.update(current_epoch=epoch,
                                 measures_dictionary=segmentation_evaluator.get_averaged_measures(current_epoch=epoch))

        epoch_time = time.time() - epoch_start_time
        remain_time = epoch_time * (args.num_epochs - 1 - epoch)
        m, s = divmod(remain_time, 60)
        h, m = divmod(m, 60)

        if s != 0:
            print("Remaining training time: {:02d}:{:02d}:{:02d}.".format(h, m, s))
        else:
            print("Training completed.")