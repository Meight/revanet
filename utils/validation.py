from collections import OrderedDict

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


class SegmentationEvaluator:
    def __init__(self, measure_names, number_of_classes):
        for measure_name in measure_names:
            if measure_name not in self._get_available_measures_with_functions().keys():
                raise ValueError('Measure {} is not supported for validation.'.format(measure_name))

        self.active_measures = measure_names
        self.number_of_classes = number_of_classes

        self.measures_history = {}
        self.initialize_history()

    def initialize_history(self):
        for measure_name in self.active_measures:
            self.measures_history[measure_name] = []

    def evaluate(self, prediction, annotation):
        for measure_name in self.active_measures:
            measure_function = self._get_available_measures_with_functions()[measure_name]
            self.measures_history[measure_name] = measure_function(prediction, annotation)

    def get_averaged_measures(self, current_epoch):
        averaged_measures = OrderedDict({'epoch': current_epoch})

        for measure_name in self.active_measures:
            averaged_measures.update({
                measure_name: np.mean(self.measures_history[measure_name])
            })

        return averaged_measures

    def _get_available_measures_with_functions(self):
        return {
            'accuracy': self.compute_accuracy,
            'accuracy_per_class': self.compute_accuracy_per_class,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'iou': self.compute_mean_iou,
        }

    @staticmethod
    def compute_accuracy(prediction, annotation):
        total = len(annotation)
        count = 0.0

        for i in range(total):
            if prediction[i] == annotation[i]:
                count = count + 1.0

        return float(count) / float(total)

    def compute_accuracy_per_class(self, prediction, annotation):
        total = []
        for val in range(self.number_of_classes):
            total.append((annotation == val).sum())

        count = [0.0] * self.number_of_classes
        for i in range(len(annotation)):
            if prediction[i] == annotation[i]:
                count[int(prediction[i])] = count[int(prediction[i])] + 1.0

        accuracies = []
        for i in range(len(total)):
            if total[i] == 0:
                accuracies.append(1.0)
            else:
                accuracies.append(count[i] / total[i])

        return accuracies

    @staticmethod
    def compute_mean_iou(prediction, annotation):
        unique_labels = np.unique(annotation)
        num_unique_labels = len(unique_labels)

        intersection = np.zeros(num_unique_labels)
        union = np.zeros(num_unique_labels)

        for index, val in enumerate(unique_labels):
            prediction_i = prediction == val
            label_i = annotation == val

            intersection[index] = float(np.sum(np.logical_and(label_i, prediction_i)))
            union[index] = float(np.sum(np.logical_or(label_i, prediction_i)))

        return np.mean(intersection / union)
