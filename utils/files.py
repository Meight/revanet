import csv
import glob
import os


def retrieve_dataset_information(dataset_path):
    csv_files = glob.glob(os.path.join('datasets', dataset_path, '*.csv'))

    if not csv_files:
        raise ValueError('You must provide the dataset information in a CSV file at its root.')

    class_names = []
    class_colors = []

    with open(csv_files[0], 'r') as dataset_info_file:
        file_reader = csv.reader(dataset_info_file, delimiter=',')
        _ = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            class_colors.append([int(row[1]), int(row[2]), int(row[3])])

    return class_names, class_colors
