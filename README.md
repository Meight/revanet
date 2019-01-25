# REVANet

An accessible and generic framework for semantic image segmentation built with
TensorFlow and Python.

# Getting started

Create a local virtual environment using Python's `virtualenv` module.

Note: please make sure the virtual environment you're using to run this project
contains the required dependencies listed in `requirements.txt`. You can run
a quick `pip install -r requirements.txt` to do that.

## Using a custom dataset

There's a little but necessary overhead work that this framework requires you to
do in order to be able to correctly retrieve your custom data. Your images need
to be JPEG images and their associated annotations must be RGB PNG encoded
images.

1. Create a folder named after your dataset in `datasets/`,
   e.g. `datasets/my-custom-dataset`;
2. Split your dataset into three subsets called `train`, `val` and `test`. Each
   subset consists of two folders, e.g. `train` and `train_labels`. The former
   contains the training image samples while the latter contains the associated
   annotations;
3. Provide a mapping between the different classes of your dataset and their
   associated RGB channels under the form of a CSV file typically named
   `information.csv` at the root of your dataset. The framework will parse this 
   file to know how to retrieve classes from your annotations.

Your dataset is ready!

## Training a model

_TODO._

```
  --number-of-cpus NUMBER_OF_CPUS
  --number-of-gpus NUMBER_OF_GPUS
  --prediction-validation-threshold PREDICTION_VALIDATION_THRESHOLD
                        Whether or not a threshold should be applied to
                        validate predictions during multi-label
                        classification.
  --learning-rate LEARNING_RATE
                        Learning rate to use
  --number-of-epochs NUMBER_OF_EPOCHS
                        Number of epochs to train for
  --first-epoch FIRST_EPOCH
                        Start counting epochs from this number
  --save-weights-every SAVE_WEIGHTS_EVERY
                        How often to save checkpoints (epochs)
  --validate-every VALIDATE_EVERY
                        How often to perform validation (epochs)
  --continue-training   Whether to continue training from a checkpoint
  --dataset-name DATASET_NAME
                        Dataset you are using.
  --input-size INPUT_SIZE
                        Box six of input image to network
  --batch-size BATCH_SIZE
                        Number of images in each batch
  --training-ratio TRAINING_RATIO
                        The ratio of training samples to use to perform actual
                        training.
  --validation-ratio VALIDATION_RATIO
                        The ratio of validation samples to use to perform
                        actual validation.
  --model-name MODEL_NAME
                        The model you are using. See model_builder.py for
                        supported models
  --backbone-name BACKBONE_NAME
                        The backbone to use. See frontend_builder.py for
                        supported models
  --results-directory RESULTS_DIRECTORY
                        Path to the directory where the results are to be
                        stored.
  --train-folder TRAIN_FOLDER
                        Name of the folder in which the training samples are
                        to be found.
  --train-annotations-folder TRAIN_ANNOTATIONS_FOLDER
                        Name of the folder containing the annotations
                        corresponding to the training samples.
  --validation-folder VALIDATION_FOLDER
                        Name of the folder in which the validation samples are
                        to be found.
  --validation-annotations-folder VALIDATION_ANNOTATIONS_FOLDER
                        Name of the folder containing the annotations
                        corresponding to the validation samples.
  --ignore-class-name IGNORE_CLASS_NAME
                        Name of the class that's representing the parts of an
                        image that should be ignored during evaluation and
                        training.
  --augmentation-strategy {none,light,aggressive}
                        The strategy to adopt for data augmentation during
                        training.
```

## Todo list

Ordered by priority.

- [x] allow to select a custom subset folder and override defaults for `train`
  and `validation`
- [x] naming strategy for logging files taking into account which subset has
  been selected
- [x] allow usage of a subset of the training data with guarantee of
  reproducibility, through a ratio and a random seed.
- [x] check passed training arguments make sense against the provided dataset
- [ ] add support for
  - models:
    - [x] BiSeNet
    - [ ] DeepLab
    - [ ] DenseNet
  - backbones:
    - [x] ResNet101
    - [ ] ResNet50
    - [ ] ResNet152
- [x] change the data handling to make use of the `tf.Dataset` API
- [ ] thorough readme file
- [x] utility script to check a dataset has valid format
- [ ] add substantial documentation
- [x] add support for multi-GPU processing
