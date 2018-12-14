import csv
import os


def retrieve_dataset_information(dataset_info_file_path):
    filename, file_extension = os.path.splitext(dataset_info_file_path)

    if not file_extension == ".csv":
        return ValueError('You must provide the dataset information in a CSV file at its root.')

    class_names = []
    class_colors = []

    with open(dataset_info_file_path, 'r') as dataset_info_file:
        file_reader = csv.reader(dataset_info_file, delimiter=',')
        _ = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            class_colors.append([int(row[1]), int(row[2]), int(row[3])])

    return class_names, class_colors
