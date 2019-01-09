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

## Todo list

- [ ] thorough readme file
- [ ] utility script to check a dataset has valid format
- [ ] check passed training arguments make sense against the provided dataset
- [ ] add substantial documentation
- [ ] add support for
  - [x] BiSeNet
  - [ ] DeepLab
  - [ ] DenseNet
- [ ] add support for multi-GPU processing
