"""
Set of utility functions to ensure the correctness of a dataset.
"""
from pathlib import Path

class BaseValidationError(ValueError):
    pass


class DatasetDoesNotExistError(BaseValidationError):
    pass


class DatasetSubsetDoesNotExistError(BaseValidationError):
    pass


def check_dataset_correctness(dataset_name,
                              dataset_path: Path,
                              train_path: Path,
                              train_annotations_path: Path,
                              validation_path: Path,
                              validation_annotations_path: Path):
    """
    Checks the correctness of a provided dataset's structure.

    :param validation_annotations_path:
    :param train_annotations_path:
    :param dataset_name:    The name of the dataset to check.
    :param dataset_path:    The path to the dataset root folder.
    :param train_path:      The name of the folder containing the training samples.
    :param validation_path: The name of the folder containing the validation samples.
    """
    if not dataset_path.exists():
        raise DatasetDoesNotExistError('Not directory was found at {} for dataset {}.'.format(dataset_path,
                                                                                              dataset_name))

    if not train_path.exists():
        raise DatasetSubsetDoesNotExistError('Train folder not found at {} for dataset {}.'.format(train_path,
                                                                                                   dataset_name))

    if not train_annotations_path.exists():
        raise DatasetSubsetDoesNotExistError('Not annotations folder found for train subset at {} for dataset {}.'
                                             .format(train_annotations_path, dataset_name))

    if not validation_path.exists():
        raise DatasetSubsetDoesNotExistError('Validation folder not found at {} for dataset {}.'.format(validation_path,
                                                                                                        dataset_name))

    if not validation_annotations_path.exists():
        raise DatasetSubsetDoesNotExistError('Not annotations folder found for validation subset at {} for dataset {}.'
                                             .format(validation_annotations_path, dataset_name))
