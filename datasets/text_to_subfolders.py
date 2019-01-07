"""
Converts text files to sub-folders of images and labels.
"""
import argparse
import multiprocessing
from pathlib import Path
from shutil import copy2

from PIL import Image
import numpy as np
from joblib import Parallel, delayed

import sys
sys.path.append("/projets/thesepizenberg/deep-learning/revanet/")

from utils.files import retrieve_dataset_information


class Processor:
    def __init__(self, source_path, target_path, class_names, class_colors_dictionary):
        self.source_path = source_path
        self.target_path = target_path
        self.class_names = class_names
        self.class_colors_dictionary = class_colors_dictionary

    def _process_annotation(self, annotation_path):
        image = Image.open(annotation_path)
        image.load()
        data = np.asarray(image, dtype=np.uint8)
        data = data.astype(np.uint8)
        result = None

        for class_index in range(len(self.class_names)):
            if result is None:
                result = np.zeros_like(data)
                result = result.astype('uint8')
                result = np.stack((result,) * 3, axis=-1)

            class_name = self.class_names[class_index]

            class_indices = np.where(data.astype(np.uint8) == class_index)

            is_class_present = class_indices[0].size != 0 or class_indices[1].size != 0

            if is_class_present:
                # Generate the colored mask based on Pascal's palette.
                colored_mask = np.zeros_like(data)
                colored_mask[class_indices] = 1
                colored_mask = np.stack((colored_mask,) * 3, axis=-1)
                red, green, blue = self.class_colors_dictionary[class_name]
                colored_mask[:, :, 0] *= red
                colored_mask[:, :, 1] *= green
                colored_mask[:, :, 2] *= blue

                result[class_indices] = colored_mask[class_indices]

        return Image.fromarray(result)

    def _process_line(self, text_file, line, annotations_save_path):
        split_line = line.split(' ')

        if len(split_line) == 2:
            image_name, annotation_name = split_line[0][1:].strip('\n'), split_line[1][1:].strip('\n')
        else:
            image_name, annotation_name = split_line[0][1:].strip('\n'), None

        image_path = Path(self.source_path, image_name)
        if annotation_name is not None:
            annotation_path = Path(self.source_path, annotation_name)
        else:
            annotation_path = None
            print('No annotation found for image {}.'.format(image_name))

        if image_path.exists():
            copy2(image_path, Path(self.target_path, text_file.stem, image_path.stem + '.jpg'))

            if annotation_path.exists():
                processed_annotation = self._process_annotation(annotation_path)
                processed_annotation.save(Path(annotations_save_path, annotation_path.stem + '.png'))
            else:
                print('Image {} not found.'.format(image_name))

    def process_text_file(self, text_file, used_cpus=3):
        print('Found text file for subset {}.'.format(text_file.stem))
        Path(self.target_path, text_file.stem).mkdir(parents=True, exist_ok=True)

        annotations_save_path = Path(self.target_path, text_file.stem + '_labels')
        annotations_save_path.mkdir(parents=True, exist_ok=True)

        with open(Path(self.source_path, text_file), 'r') as file:
            Parallel(n_jobs=multiprocessing.cpu_count() - used_cpus)(delayed(self._process_line)(text_file,
                                                                                                 line,
                                                                                                 annotations_save_path)
                                                                     for line in file.readlines())


def process_dataset(text_files_directory, dataset_name, target_dataset_root):
    subsets = ['train', 'val', 'test']
    source_path = Path(text_files_directory)
    target_path = Path(target_dataset_root, dataset_name)
    target_path.mkdir(exist_ok=True, parents=True)

    class_names, class_colors = retrieve_dataset_information(dataset_path=target_path)
    class_colors_dictionary = dict(zip(class_names, class_colors))

    if not source_path.exists():
        raise ValueError('Provided text files directory not found at {}.'.format(source_path))

    found_text_files = source_path.glob('*.txt')

    processor = Processor(source_path=source_path,
                          target_path=target_path,
                          class_names=class_names,
                          class_colors_dictionary=class_colors_dictionary)

    used_text_files = []
    for text_file in found_text_files:
        if text_file.stem in subsets:
            used_text_files.append(text_file)

    used_cpus = len(used_text_files)
    Parallel(n_jobs=used_cpus)(delayed(processor.process_text_file)(text_file, used_cpus=used_cpus)
                               for text_file in used_text_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-files-directory')
    parser.add_argument('--dataset-name')
    parser.add_argument('--target-dataset-root')
    args = parser.parse_args()

    TEXT_FILES_DIRECTORY = str(args.text_files_directory)
    DATASET_NAME = str(args.dataset_name)
    TARGET_DATASET_ROOT = str(args.target_dataset_root)

    process_dataset(TEXT_FILES_DIRECTORY, DATASET_NAME, TARGET_DATASET_ROOT)
