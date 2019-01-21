"""
Set of utilities for Tensorflow parallel computations.
"""
import argparse
import random
import time
from pathlib import Path

import tensorflow as tf
from utils.arguments import ratio
from utils.data_generation import get_batch_loader_for_subset
from utils.dataset import check_dataset_correctness, generate_dataset
from utils.files import retrieve_dataset_information
from utils.models import ModelBuilder
from utils.naming import FilesFormatterFactory
from utils.utils import (gather_multi_label_data,
                         get_available_annotation_resized_tensors_for_image,
                         prepare_data)
from utils.validation import SegmentationEvaluator

parser = argparse.ArgumentParser()
parser.add_argument(
    '--learning-rate',
    type=float,
    default=0.0001,
    help='Learning rate to use.')
parser.add_argument(
    '--number-of-epochs',
    type=int,
    default=75,
    help='Number of epochs to train for.')
parser.add_argument(
    '--save-weights-every',
    type=int,
    default=5,
    help='How often to save checkpoints (epochs)')
parser.add_argument(
    '--validate-every',
    type=int,
    default=1,
    help='How often to perform validation (epochs)')
parser.add_argument(
    '--continue-training',
    action='store_true',
    default=False,
    help='Whether to continue training from a checkpoint')
parser.add_argument(
    '--dataset-name',
    type=str,
    help='The name of the dataset to fetch in the /datasets folder.')
parser.add_argument(
    '--input-size', type=int, default=512, help='Square size of input images.')
parser.add_argument(
    '--batch-size', type=int, default=1, help='Number of images in each batch')
parser.add_argument(
    '--training-ratio',
    type=ratio,
    default=1.0,
    help='The ratio of training samples to use to perform actual training.')
parser.add_argument(
    '--validation-ratio',
    type=ratio,
    default=1.0,
    help='The ratio of validation samples to use to perform actual validation.'
)
parser.add_argument(
    '--model-name',
    type=str,
    default="BiSeNet",
    help='The model you are using. See model_builder.py for supported models')
parser.add_argument(
    '--backbone-name',
    type=str,
    default="ResNet101",
    help='The backbone to use. See frontend_builder.py for supported models')
parser.add_argument(
    '--results-directory',
    type=str,
    default='./',
    help='Path to the directory where the results are to be stored.')
parser.add_argument(
    '--train-folder',
    type=str,
    default='train',
    help='Name of the folder in which the training samples are to be found.')
parser.add_argument(
    '--train-annotations-folder',
    type=str,
    default='train_annotations',
    help=
    'Name of the folder containing the annotations corresponding to the training samples.'
)
parser.add_argument(
    '--validation-folder',
    type=str,
    default='validation',
    help='Name of the folder in which the validation samples are to be found.')
parser.add_argument(
    '--validation-annotations-folder',
    type=str,
    default='validation_annotations',
    help=
    'Name of the folder containing the annotations corresponding to the validation samples.'
)
args = parser.parse_args()

INPUT_SIZE = int(args.input_size)

DATASET_NAME = str(args.dataset_name)
DATASET_PATH = Path('datasets', DATASET_NAME)
TRAIN_PATH = Path(DATASET_PATH, str(args.train_folder))
TRAIN_ANNOTATIONS_PATH = Path(DATASET_PATH, str(args.train_annotations_folder))
VALIDATION_PATH = Path(DATASET_PATH, str(args.validation_folder))
VALIDATION_ANNOTATIONS_PATH = Path(DATASET_PATH,
                                   str(args.validation_annotations_folder))
RESULTS_DIRECTORY = str(args.results_directory)

# Ensure that all required folders and files exist and are well defined.
check_dataset_correctness(
    dataset_name=DATASET_NAME,
    dataset_path=DATASET_PATH,
    train_path=TRAIN_PATH,
    train_annotations_path=TRAIN_ANNOTATIONS_PATH,
    validation_path=VALIDATION_PATH,
    validation_annotations_path=VALIDATION_ANNOTATIONS_PATH)

MODEL_NAME = str(args.model_name)
BACKBONE_NAME = str(args.backbone_name)
CONTINUE_TRAINING = bool(args.continue_training)

NUMBER_OF_EPOCHS = int(args.number_of_epochs)
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

NUMBER_OF_CPUS = 1
NUMBER_OF_GPUS = 1

validation_measures = [
    'accuracy', 'accuracy_per_class', 'precision', 'recall', 'f1', 'iou'
]

ADDITIONAL_INFO = {
    'results_directory': RESULTS_DIRECTORY,
    'model': MODEL_NAME,
    'backbone': BACKBONE_NAME,
    'validation_every': VALIDATE_EVERY,
    'saving_weights_every': SAVE_WEIGHTS_EVERY,
    'random_seed': RANDOM_SEED,
    'validation_measures': validation_measures,
    'train_annotations_folder': TRAIN_ANNOTATIONS_PATH,
    'number_of_gpus': NUMBER_OF_GPUS,
    'number_of_cpus': NUMBER_OF_CPUS
}

files_formatter_factory = FilesFormatterFactory(
    mode='training',
    dataset_name=DATASET_NAME,
    model_name=MODEL_NAME,
    backbone_name=BACKBONE_NAME,
    training_parameters=TRAINING_PARAMETERS,
    train_path=TRAIN_PATH,
    verbose=True,
    results_folder=RESULTS_DIRECTORY)


def get_available_gpus():
    """
    Returns a list of the identifiers of all visible GPUs of the form
    `['/gpu:0', '/gpu:1', '/gpu:2', ...]`.
    https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def create_parallel_optimization(model_fn,
                                 input_batch,
                                 output_batch,
                                 optimizer,
                                 controller="/cpu:0"):
    devices = get_available_gpus()
    # Store the gradients per tower.
    tower_grads = []

    # Store the losses for later averaging.
    losses = []

    # mIoU.
    scores = []

    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id in enumerate(devices):
            name = 'tower_{}'.format(i)
            # Ensure the variables are created on the controller and nowhere else.
            with tf.device(assign_to_device(id,
                                            controller)), tf.name_scope(name):
                # Compute loss and gradients. They shall not be applied yet
                # as we'd lose all interest in the parallelization.
                weights_shape = (BATCH_SIZE, INPUT_SIZE, INPUT_SIZE)
                unc = tf.where(
                    tf.equal(tf.reduce_sum(output_batch, axis=-1), 0),
                    tf.zeros(shape=weights_shape),
                    tf.ones(shape=weights_shape))

                adapted_loss = tf.nn.softmax_cross_entropy_with_logits_v2
                loss = tf.reduce_mean(
                    tf.losses.compute_weighted_loss(
                        weights=tf.cast(unc, tf.float32),
                        losses=adapted_loss(
                            logits=model_fn, labels=output_batch)))

                annotations = tf.argmax(output_batch, axis=-1)
                predictions = tf.argmax(model_fn, axis=-1)
                iou, update_metric = tf.metrics.mean_iou(labels=annotations, predictions=predictions, num_classes=21, weights=tf.cast(unc, tf.float32), name="mean_iou")


                score = tf.reduce_mean(iou)
                scores.append(score)

                with tf.name_scope("compute_gradients"):
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

                losses.append(loss)

            outer_scope.reuse_variables()

    # Apply the gradients on the controlling device.
    with tf.name_scope("apply_gradients"), tf.device(controller):
        gradients = average_gradients(tower_grads)
        global_step = tf.train.get_or_create_global_step()
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
        avg_loss = tf.reduce_mean(losses)
        avg_scores = tf.reduce_mean(scores)

    return apply_gradient_op, avg_loss, avg_scores, update_metric


PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]


def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the default device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.

    See https://github.com/tensorflow/tensorflow/issues/9517.
    """

    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device

    return _assign


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    This function provides a synchronization point across all towers.

    tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
        over the devices. The inner list ranges over the different variables.
    Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            and thus synchronized across all towers.

    See https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101.
    """
    average_grads = []

    for grad_and_vars in zip(*tower_grads):

        # Each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        mean_grad = tf.reduce_mean(grads, 0)

        grad_and_var = (mean_grad, grad_and_vars[0][1])
        average_grads.append(grad_and_var)

    return average_grads


def parallel_training(dataset, training_steps_per_epoch):
    iterator = dataset.make_one_shot_iterator()

    def input_fn():
        with tf.device(None):
            return iterator.get_next()

    input_batch, output_batch = input_fn()
    input_batch = tf.cast(input_batch, tf.float32)

    model_builder = ModelBuilder(
        number_of_classes=number_of_classes,
        input_size=INPUT_SIZE,
        backbone_name=BACKBONE_NAME,
        is_training=True)

    model_fn, init_fn = model_builder.build(
        model_name=MODEL_NAME, inputs=input_batch)

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    update_op, loss, score, update_metric = create_parallel_optimization(model_fn, input_batch,
                                                   output_batch, optimizer)

    do_training(training_steps_per_epoch, update_op, loss, score, update_metric, init_fn)


#def do_validation(session, model_fn, dataset):
#    predictions_batch = 


def do_training(training_steps_per_epoch, update_op, loss, score, update_metric, init_fn=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer()) 


        if init_fn is not None:
            init_fn(session)

        checkpoint_formatter = files_formatter_factory.get_checkpoint_formatter(
            saver=tf.train.Saver(max_to_keep=1000))
        summary_formatter = files_formatter_factory.get_summary_formatter()
        logs_formatter = files_formatter_factory.get_logs_formatter()

        logs_formatter.write(additional_info=ADDITIONAL_INFO)

        try:
            step = 0
            while True:
                current_epoch = step // training_steps_per_epoch

                start_time = time.time()

                _, loss_value, _ = session.run((update_op, loss, update_metric))

                if current_epoch != 0 and current_epoch % VALIDATE_EVERY == 0:
                    pass

                if current_epoch != 0 and current_epoch % SAVE_WEIGHTS_EVERY == 0:
                    #checkpoint_formatter.save(
                    #    session=session, current_epoch=current_epoch)
                    pass

                if step % PRINT_INFO_EVERY == 0:
                    score_value = session.run(score)
                    print(
                        '[{} - {} #{}/{} - Step {}] Seen samples: {}, current loss: {}, score: {}, time spent on batch: {:0.2f}'
                        .format(MODEL_NAME, BACKBONE_NAME, current_epoch + 1, NUMBER_OF_EPOCHS, step - step // (current_epoch + 1),
                                step * BATCH_SIZE, loss_value, score_value,
                                time.time() - start_time))

                step += 1
        except tf.errors.OutOfRangeError:
            print('Training finished.')


subset_associations = prepare_data(TRAIN_PATH, TRAIN_ANNOTATIONS_PATH,
                                   VALIDATION_PATH,
                                   VALIDATION_ANNOTATIONS_PATH)

random.seed(RANDOM_SEED)

class_names_list, class_colors = retrieve_dataset_information(
    dataset_path=DATASET_PATH)
class_colors_dictionary = dict(zip(class_names_list, class_colors))
number_of_classes = len(class_colors)

segmentation_evaluator = SegmentationEvaluator(validation_measures,
                                               number_of_classes)

train_augmenter, train_batch_loader, training_steps_per_epoch = get_batch_loader_for_subset(
    number_of_epochs=NUMBER_OF_EPOCHS,
    batch_size=BATCH_SIZE,
    input_size=INPUT_SIZE,
    subset_associations=subset_associations['train'],
    class_colors=class_colors)

training_dataset = generate_dataset(
    train_augmenter,
    input_size=INPUT_SIZE,
    number_of_epochs=NUMBER_OF_EPOCHS,
    batch_size=BATCH_SIZE,
    number_of_cpus=NUMBER_OF_CPUS,
    number_of_gpus=NUMBER_OF_GPUS,
    class_colors=class_colors)

tf.reset_default_graph()
parallel_training(training_dataset, training_steps_per_epoch)
