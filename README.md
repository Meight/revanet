- [RevaNet](#sec-1)
  - [Getting started](#sec-1-1)
  - [Using a custom dataset](#sec-1-2)
  - [Training a model](#sec-1-3)
  - [Features to add <code>[80%]</code>](#sec-1-4)

# RevaNet<a id="sec-1"></a>

An accessible and generic framework for semantic image segmentation built with TensorFlow and Python.

## Getting started<a id="sec-1-1"></a>

Create a local virtual environment using for example `virtualenv` or [Anaconda](https://www.anaconda.com/).

Note: please make sure the virtual environment you&rsquo;re using to run this project contains the required dependencies listed in `requirements.txt`. You can run a quick `pip install -r requirements.txt` to do that.

## Using a custom dataset<a id="sec-1-2"></a>

There&rsquo;s little but necessary overhead work that this framework requires you to do in order to be able to correctly retrieve your custom data. Your images need to be JPEG images and their associated annotations must be RGB PNG encoded images.

1.  Create a folder named after your dataset in `datasets/`, e.g. `datasets/my-custom-dataset`;
2.  Split your dataset into three subsets called `train`, `val` and `test`. Each subset consists of two folders, e.g. `train` and `train_labels`. The former contains the training image samples while the latter contains the associated annotations;
3.  Provide a mapping between the different classes of your dataset and their associated RGB channels under the form of a CSV file typically named `information.csv` at the root of your dataset. The framework will parse this file to know how to retrieve classes from your annotations.

Your dataset is ready! You can train any supported model onto this dataset by running `train.py --dataset-name=my-custom-dataset`.

## Training a model<a id="sec-1-3"></a>

The training script comes with many optional arguments that allow fine tuning, pointing to custom folders, changing the data augmentation strategy, continue training from a previous checkpoint, etc.

```
usage: train_multi_gpu.py [-h] --number-of-cpus NUMBER_OF_CPUS
                          --number-of-gpus NUMBER_OF_GPUS
                          [--prediction-validation-threshold PREDICTION_VALIDATION_THRESHOLD]
                          [--learning-rate LEARNING_RATE]
                          [--number-of-epochs NUMBER_OF_EPOCHS]
                          [--first-epoch FIRST_EPOCH]
                          [--save-weights-every SAVE_WEIGHTS_EVERY]
                          [--validate-every VALIDATE_EVERY]
                          [--continue-training] --dataset-name DATASET_NAME
                          [--input-size INPUT_SIZE] [--batch-size BATCH_SIZE]
                          [--training-ratio TRAINING_RATIO]
                          [--validation-ratio VALIDATION_RATIO]
                          [--model-name {BiSeNet,DeepLabv3_plus,GCN,FC-DenseNet56,FC-DenseNet67,FC-DenseNet103,AdapNet}]
                          [--backbone-name {ResNet101}]
                          [--results-directory RESULTS_DIRECTORY]
                          [--train-folder TRAIN_FOLDER]
                          [--train-annotations-folder TRAIN_ANNOTATIONS_FOLDER]
                          [--validation-folder VALIDATION_FOLDER]
                          [--validation-annotations-folder VALIDATION_ANNOTATIONS_FOLDER]
                          [--ignore-class-name IGNORE_CLASS_NAME]
                          [--augmentation-strategy {none,light,aggressive}]

optional arguments:
  -h, --help            show this help message and exit
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
  --model-name {BiSeNet,DeepLabv3_plus,GCN,FC-DenseNet56,FC-DenseNet67,FC-DenseNet103,AdapNet}
                        The model to train (default: BiSeNet).
  --backbone-name {ResNet101}
                        The backbone to use (default: ResNet101).
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

## TODO Features to add <code>[80%]</code><a id="sec-1-4"></a>

-   [X] allow to select a custom subset folder and override defaults for `train` and `validation`;
-   [X] naming strategy for logging files taking into account which subset has been selected;
-   [X] allow usage of a subset of the training data with guarantee of reproducibility, through a ratio and a random seed;
-   [X] check passed training arguments make sense against the provided dataset
-   [ ] add support for
    -   models: <code>[5/6]</code>
        -   [X] BiSeNet
        -   [X] DeepLab v3+
        -   [X] DenseNet
        -   [X] AdapNet
        -   [X] FC-DenseNet: <code>[3/3]</code>
            -   [X] 56
            -   [X] 71
            -   [X] 103
        -   [ ] PSPNet
    -   backbones: <code>[1/3]</code>
        -   [X] ResNet101
        -   [ ] ResNet50
        -   [ ] ResNet152
-   [X] change the data handling to make use of the `tf.Dataset` API;
-   [X] thorough readme file;
-   [X] utility script to check a dataset has valid format;
-   [ ] add substantial documentation;
-   [X] add support for multi-GPU processing.
