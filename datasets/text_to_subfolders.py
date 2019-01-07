"""
Converts text files to sub-folders of images and labels.
"""
import argparse
from pathlib import Path
from shutil import copy2

from PIL import Image
import numpy as np
from tqdm import tqdm

from utils.files import retrieve_dataset_information


def _process_annotation(annotation_path, class_names, class_colors_dictionary):
    annotation_name = annotation_path.stem
    image = Image.open(annotation_path)
    image.load()
    data = np.asarray(image, dtype=np.uint8)
    data = data.astype(np.uint8)
    result = None

    for class_index in range(len(class_names)):
        if result is None:
            result = np.zeros_like(data)
            result = result.astype('uint8')
            result = np.stack((result,) * 3, axis=-1)

        class_name = class_names[class_index]

        class_indices = np.where(data.astype(np.uint8) == class_index)

        is_class_present = class_indices[0].size != 0 or class_indices[1].size != 0

        if is_class_present:
            # Generate the colored mask based on Pascal's palette.
            colored_mask = np.zeros_like(data)
            colored_mask[class_indices] = 1
            colored_mask = np.stack((colored_mask,) * 3, axis=-1)
            red, green, blue = class_colors_dictionary[class_name]
            colored_mask[:, :, 0] *= red
            colored_mask[:, :, 1] *= green
            colored_mask[:, :, 2] *= blue

            result[class_indices] = colored_mask[class_indices]

    return Image.fromarray(result)


def process_dataset(text_files_directory, dataset_name, dataset_root):
    subsets = ['train', 'val', 'test']
    source_path = Path(text_files_directory)
    target_path = Path(dataset_root)

    class_names, class_colors = retrieve_dataset_information(dataset_path=target_path)
    class_colors_dictionary = dict(zip(class_names, class_colors))

    if not source_path.exists():
        raise ValueError('Provided text files directory not found at {}.'.format(source_path))

    found_text_files = source_path.glob('*.txt')

    for text_file in found_text_files:
        if text_file.stem in subsets:
            print('Found text file for subset {}.'.format(text_file.stem))
            Path(target_path, text_file.stem).mkdir(parents=True, exist_ok=True)

            annotations_save_path = Path(target_path, text_file.stem + '_labels')
            annotations_save_path.mkdir(parents=True, exist_ok=True)

            with open(Path(source_path, text_file), 'r') as file:
                for line in tqdm(file.readlines()):
                    split_line = line.split(' ')

                    if len(split_line) == 2:
                        image_name, annotation_name = split_line[0][1:].strip('\n'), split_line[1][1:].strip('\n')
                    else:
                        image_name, annotation_name = split_line[0][1:].strip('\n'), None

                    image_path = Path(source_path, image_name)
                    if annotation_name is not None:
                        annotation_path = Path(source_path, annotation_name)
                    else:
                        print(f'No annotation found for image {image_name}.')

                    if image_path.exists():
                        copy2(image_path, Path(target_path, text_file.stem, image_path.stem + '.jpg'))

                        if annotation_path.exists():
                            processed_annotation = _process_annotation(annotation_path,
                                                                       class_names=class_names,
                                                                       class_colors_dictionary=class_colors_dictionary)
                            processed_annotation.save(Path(annotations_save_path, annotation_path.stem + '.png'))
                    else:
                        print(f'Image {image_name} not found.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-files-directory')
    parser.add_argument('--dataset-name')
    parser.add_argument('--dataset-root')
    args = parser.parse_args()

    TEXT_FILES_DIRECTORY = str(args.text_files_directory)
    DATASET_NAME = str(args.dataset_name)
    DATASET_ROOT = str(args.dataset_root)

    process_dataset(TEXT_FILES_DIRECTORY, DATASET_NAME, DATASET_ROOT)