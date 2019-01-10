from __future__ import print_function

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils import utils, segmentation
from utils.arguments import ratio
from utils.augmentation import augment_data
from utils.dataset import check_dataset_correctness
from utils.files import retrieve_dataset_information
from utils.models import ModelBuilder
from utils.naming import FilesFormatterFactory
from utils.utils import gather_multi_label_data, \
    get_available_annotation_resized_tensors_for_image, prepare_data
from utils.validation import SegmentationEvaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--is-multi-label-segmentation',
                    action='store_true',
                    default=False,
                    help='Whether or not to interpret the task as multi-label classification.')
parser.add_argument('--prediction-validation-threshold',
                    action='store',
                    default=0.5,
                    type=ratio,
                    help='Whether or not a threshold should be applied to validate predictions during multi-label'
                         'classification.')
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
parser.add_argument('--continue-training',
                    action='store_true',
                    default=False,
                    help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset-name',
                    type=str,
                    default='voc-chh',
                    help='Dataset you are using.')
parser.add_argument('--input-size',
                    type=int,
                    default=512,
                    help='Box six of input image to network')
parser.add_argument('--batch-size',
                    type=int,
                    default=1,
                    help='Number of images in each batch')
parser.add_argument('--training-ratio',
                    type=ratio,
                    default=1.0,
                    help='The ratio of training samples to use to perform actual training.')
parser.add_argument('--validation-ratio',
                    type=ratio,
                    default=1.0,
                    help='The ratio of validation samples to use to perform actual validation.')
parser.add_argument('--model-name',
                    type=str,
                    default="FC-DenseNet56",
                    help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--backbone-name',
                    type=str,
                    default="ResNet101",
                    help='The backbone to use. See frontend_builder.py for supported models')
parser.add_argument('--results-directory',
                    type=str,
                    default='/projets/thesepizenberg/deep-learning/revanet/',
                    help='Path to the directory where the results are to be stored.')
parser.add_argument('--train-folder',
                    type=str,
                    default='train',
                    help='Name of the folder in which the training samples are to be found.')
parser.add_argument('--train-annotations-folder',
                    type=str,
                    default='train_annotations',
                    help='Name of the folder containing the annotations corresponding to the training samples.')
parser.add_argument('--validation-folder',
                    type=str,
                    default='validation',
                    help='Name of the folder in which the validation samples are to be found.')
parser.add_argument('--validation-annotations-folder',
                    type=str,
                    default='validation_annotations',
                    help='Name of the folder containing the annotations corresponding to the validation samples.')
args = parser.parse_args()

IS_MULTI_LABEL_CLASSIFICATION = bool(args.is_multi_label_segmentation)
VALIDATION_THRESHOLD = float(args.prediction_validation_threshold)
INPUT_SIZE = int(args.input_size)

# No augmentation available for multi-label classification.
is_dataset_augmented = False

DATASET_NAME = str(args.dataset_name)
DATASET_PATH = Path('datasets', DATASET_NAME)
TRAIN_PATH = Path(DATASET_PATH, str(args.train_folder))
TRAIN_ANNOTATIONS_PATH = Path(DATASET_PATH, str(args.train_annotations_folder))
VALIDATION_PATH = Path(DATASET_PATH, str(args.validation_folder))
VALIDATION_ANNOTATIONS_PATH = Path(DATASET_PATH, str(args.validation_annotations_folder))
RESULTS_DIRECTORY = str(args.results_directory)

# Ensure that all required folders and files exist and are well defined.
check_dataset_correctness(dataset_name=DATASET_NAME,
                          dataset_path=DATASET_PATH,
                          train_path=TRAIN_PATH,
                          train_annotations_path=TRAIN_ANNOTATIONS_PATH,
                          validation_path=VALIDATION_PATH,
                          validation_annotations_path=VALIDATION_ANNOTATIONS_PATH)

MODEL_NAME = str(args.model_name)
BACKBONE_NAME = str(args.backbone_name)
CONTINUE_TRAINING = bool(args.continue_training)

NUMBER_OF_EPOCHS = int(args.number_of_epochs)
FIRST_EPOCH = int(args.first_epoch)
BATCH_SIZE = int(args.batch_size)
LEARNING_RATE = float(args.learning_rate)
TRAINING_RATIO = float(args.training_ratio)
VALIDATION_RATIO = float(args.validation_ratio)
SAVE_WEIGHTS_EVERY = int(args.save_weights_every)
VALIDATE_EVERY = int(args.validate_every)

RANDOM_SEED = 2018
PRINT_INFO_EVERY = 5  # Period (in steps) of prints.

TRAINING_PARAMETERS = {
    'epochs': NUMBER_OF_EPOCHS,
    'learning_rate': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'training_ratio': TRAINING_RATIO,
    'validation_ratio': VALIDATION_RATIO,
    'input_size': INPUT_SIZE,
    'train_images_path': TRAIN_PATH,
    'train_annotations_path': TRAIN_ANNOTATIONS_PATH,
    'validation_images_path': VALIDATION_PATH,
    'validation_annotations_path': VALIDATION_ANNOTATIONS_PATH
}

if IS_MULTI_LABEL_CLASSIFICATION:
    validation_measures = ['exact_match_ratio', 'hamming_score']
else:
    validation_measures = ['accuracy', 'accuracy_per_class', 'precision', 'recall', 'f1', 'iou']

files_formatter_factory = FilesFormatterFactory(mode='training',
                                                dataset_name=DATASET_NAME,
                                                model_name=MODEL_NAME,
                                                backbone_name=BACKBONE_NAME,
                                                training_parameters=TRAINING_PARAMETERS,
                                                train_path=TRAIN_PATH,
                                                verbose=True,
                                                results_folder=RESULTS_DIRECTORY)

class_names_list, class_colors = retrieve_dataset_information(dataset_path=DATASET_PATH)
class_colors_dictionary = dict(zip(class_names_list, class_colors))
number_of_classes = len(class_colors)

segmentation_evaluator = SegmentationEvaluator(validation_measures, number_of_classes)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3])
output_tensor = tf.placeholder(tf.float32, shape=[None, None, None, number_of_classes])

model_builder = ModelBuilder(number_of_classes=number_of_classes,
                             input_size=INPUT_SIZE,
                             backbone_name=BACKBONE_NAME,
                             is_training=True)
predictions_tensor, init_fn = model_builder.build(model_name=MODEL_NAME, inputs=input_tensor)

# total_summary = tf.summary.image('images',
#                                  tf.concat(axis=2, values=[tf.cast(input_tensor, tf.uint8),
#                                                            tf.py_func(tf.cast(one_hot_to_image, tf.uint8),
#                                                                       [output_tensor],
#                                                                       tf.uint8),
#                                                            tf.py_func(tf.cast(one_hot_to_image, tf.uint8),
#                                                                       [predictions_tensor],
#                                                                       tf.uint8)]),
#                                  max_outputs=20)  # Concatenate row-wise.
# summary_writer = tf.summary.FileWriter(RESULTS_DIRECTORY,
#                                        graph=tf.get_default_graph())

if not IS_MULTI_LABEL_CLASSIFICATION:
    weights_shape = (BATCH_SIZE, INPUT_SIZE, INPUT_SIZE)
    unc = tf.where(tf.equal(tf.reduce_sum(output_tensor, axis=-1), 0),
                   tf.zeros(shape=weights_shape),
                   tf.ones(shape=weights_shape))

    adapted_loss = tf.nn.softmax_cross_entropy_with_logits_v2
    loss = tf.reduce_mean(tf.losses.compute_weighted_loss(weights=tf.cast(unc, tf.float32),
                                                          losses=adapted_loss(logits=predictions_tensor,
                                                                              labels=output_tensor)))
else:
    adapted_loss = tf.nn.sigmoid_cross_entropy_with_logits
    loss = tf.reduce_mean(adapted_loss(logits=predictions_tensor, labels=output_tensor))

opt = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE,
                                decay=0.995,
                                momentum=0.9).minimize(loss,
                                                       var_list=[var for var in tf.trainable_variables()])

session.run(tf.global_variables_initializer())

print('Parameters:', utils.count_parameters())

if init_fn is not None:
    init_fn(session)

model_checkpoint_name = "checkpoints/latest_model_{}_{}.ckpt".format(MODEL_NAME, DATASET_NAME)
checkpoint_formatter = files_formatter_factory.get_checkpoint_formatter(saver=tf.train.Saver(max_to_keep=1000))
summary_formatter = files_formatter_factory.get_summary_formatter()
logs_formatter = files_formatter_factory.get_logs_formatter()

paths = None
subset_associations = None

if not IS_MULTI_LABEL_CLASSIFICATION:
    subset_associations = prepare_data(TRAIN_PATH, TRAIN_ANNOTATIONS_PATH, VALIDATION_PATH, VALIDATION_ANNOTATIONS_PATH)
else:
    paths = gather_multi_label_data(dataset_directory=DATASET_PATH)

    train_input_names = list(paths['train'].keys())
    validation_input_names = list(paths['val'].keys())
    test_input_names = list(paths['test'].keys())
    train_output_names = list(paths['train'].values())
    validation_output_names = list(paths['val'].values())

train_image_paths = list(subset_associations['train'].keys())
validation_image_paths = list(subset_associations['validation'].keys())

random.seed(RANDOM_SEED)
number_of_training_samples = len(train_image_paths)
number_of_used_training_samples = int(TRAINING_RATIO * number_of_training_samples)
training_indices = random.sample(range(number_of_training_samples),
                                 max(1, number_of_used_training_samples))
number_of_validation_samples = len(validation_image_paths)
number_of_used_validation_samples = int(VALIDATION_RATIO * number_of_validation_samples)
validation_indices = random.sample(range(number_of_validation_samples),
                                   max(1, number_of_used_validation_samples))

ADDITIONAL_INFO = {
    'results_directory': RESULTS_DIRECTORY,
    'model': MODEL_NAME,
    'backbone': BACKBONE_NAME,
    'validation_every': VALIDATE_EVERY,
    'saving_weights_every': SAVE_WEIGHTS_EVERY,
    'random_seed': RANDOM_SEED,
    'is_multi_label_classification': IS_MULTI_LABEL_CLASSIFICATION,
    'validation_threshold': VALIDATION_THRESHOLD,
    'training_samples': len(train_image_paths),
    'used_training_samples': number_of_training_samples,
    'validation_samples': number_of_validation_samples,
    'used_validation_samples': number_of_used_validation_samples,
    'validation_measures': validation_measures,
    'train_annotations_folder': TRAIN_ANNOTATIONS_PATH
}

logs_formatter.write(additional_info=ADDITIONAL_INFO)

if CONTINUE_TRAINING:
    print('Loaded latest model checkpoint.')
    checkpoint_formatter.restore(session, model_checkpoint_name)

average_measures_per_epoch = {
    'loss': [],
    'iou': [],
    'scores': []
}

for epoch in range(FIRST_EPOCH, NUMBER_OF_EPOCHS):
    current_losses = []
    samples_seen = 0

    input_indices = np.random.permutation(training_indices)

    number_of_steps = number_of_used_training_samples // BATCH_SIZE
    start_time = time.time()
    epoch_start_time = time.time()

    for current_step_index in range(number_of_steps):
        input_image_batch = []
        output_image_batch = []

        # Collect a batch of images.
        for current_batch_index in range(BATCH_SIZE):
            current_index = (current_step_index * BATCH_SIZE + current_batch_index) % number_of_used_training_samples
            input_image_index = input_indices[current_index]

            input_image_path = train_image_paths[input_image_index]
            input_image = utils.load_image(input_image_path)

            output_image_path = random.choice(subset_associations['train'][input_image_path])

            if not IS_MULTI_LABEL_CLASSIFICATION:
                output_image = utils.load_image(output_image_path)
            else:
                n_encoded_masks = get_available_annotation_resized_tensors_for_image((INPUT_SIZE, INPUT_SIZE),
                                                                                     output_image_path,
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
        # summary_writer.add_summary(summary, epoch)
        current_losses.append(current)
        samples_seen += BATCH_SIZE

        if samples_seen % PRINT_INFO_EVERY == 0:
            print('[{} - {} #{}/{}] Seen samples: {}, current loss: {}, time spent on batch: {:0.2f}'.format(
                MODEL_NAME, BACKBONE_NAME, epoch, NUMBER_OF_EPOCHS, samples_seen, current, time.time() - start_time))
            start_time = time.time()

    average_measures_per_epoch['loss'].append(np.mean(current_losses))

    if validation_indices != 0 and epoch % SAVE_WEIGHTS_EVERY == 0:
        checkpoint_formatter.save(session=session,
                                  current_epoch=epoch)

    if epoch % VALIDATE_EVERY == 0:
        segmentation_evaluator.initialize_history()

        for image_index in validation_indices:
            input_image_path = validation_image_paths[image_index]
            input_image = np.float32(utils.load_image(input_image_path))

            output_image_path = subset_associations['validation'][input_image_path][0]

            if not IS_MULTI_LABEL_CLASSIFICATION:
                ground_truth = utils.load_image(output_image_path)

                input_image, ground_truth = utils.resize_to_size(input_image, ground_truth, desired_size=INPUT_SIZE)
                ground_truth = segmentation.one_hot_to_image(segmentation.image_to_one_hot(ground_truth, class_colors))

                input_image = np.expand_dims(input_image, axis=0) / 255.0

                valid_indices = np.where(np.sum(ground_truth, axis=-1) != 0)
                ground_truth = ground_truth[valid_indices, :]

                output_image = session.run(predictions_tensor, feed_dict={input_tensor: input_image})

                output_image = np.array(output_image[0, :, :, :])
                output_image = output_image[valid_indices, :]
                output_image = segmentation.one_hot_to_image(output_image)

                segmentation_evaluator.evaluate(prediction=output_image,
                                                annotation=ground_truth)
            else:
                n_encoded_masks = get_available_annotation_resized_tensors_for_image(INPUT_SIZE, INPUT_SIZE,
                                                                                     output_image_path,
                                                                                     class_colors_dictionary)

                ground_truth = random.choice(n_encoded_masks)  # (INPUT_SIZE, INPUT_SIZE, NUMBER_OF_CLASSES)
                input_image, _ = utils.resize_to_size(input_image, desired_size=INPUT_SIZE)

                input_image = np.expand_dims(input_image, axis=0) / 255.0

                output_image = session.run(predictions_tensor, feed_dict={input_tensor: input_image})

                output_image = np.array(output_image[0, :, :, :])

                if VALIDATION_THRESHOLD != 0:
                    output_image = segmentation.apply_threshold_to_prediction(output_image,
                                                                              threshold=VALIDATION_THRESHOLD)
                else:
                    output_image = segmentation.one_hot_to_image(output_image)

                segmentation_evaluator.evaluate(prediction=output_image,
                                                annotation=ground_truth)

        summary_formatter.update(current_epoch=epoch,
                                 measures_dictionary=segmentation_evaluator.get_averaged_measures(current_epoch=epoch))

        epoch_time = time.time() - epoch_start_time
        remain_time = epoch_time * (NUMBER_OF_EPOCHS - 1 - epoch)
        m, s = divmod(remain_time, 60)
        h, m = divmod(m, 60)

        if s != 0:
            print("Remaining training time: {:02d}:{:02d}:{:02d}.".format(int(h), int(m), int(s)))
        else:
            print("Training completed.")
