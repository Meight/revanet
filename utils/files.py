import csv
import glob
import os


def retrieve_dataset_information(dataset_path):
    csv_files = dataset_path.glob('*.csv')

    if not csv_files:
        raise ValueError('You must provide the dataset information in a CSV file '
                         'at its root at {}.'.format(dataset_path))

    class_names = []
    class_colors = []
    csv_file = next(csv_files)

    with open(csv_file, 'r') as dataset_info_file:
        file_reader = csv.reader(dataset_info_file, delimiter=',')
        _ = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            class_colors.append([int(row[1]), int(row[2]), int(row[3])])

    return class_names, class_colors
