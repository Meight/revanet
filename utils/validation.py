from collections import OrderedDict

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score,
                             jaccard_similarity_score, precision_score,
                             recall_score)


class SegmentationEvaluator:
    """
    A segmentation evaluator which can compute a set of measures when given a prediction and an associated
    annotation.

    :param measure_names:       A list of supported measures to compute every time the evaluator is fed with a
                                    couple of a prediction and its annotation.
    :param number_of_classes:   The number of classes in the dataset. Required for the computation of some
                                    of the provided measures.
    """

    def __init__(self, measure_names, number_of_classes):
        for measure_name in measure_names:
            if measure_name not in self._get_available_measures_with_functions(
            ).keys():
                raise ValueError('Measure {} is not supported for validation.'.
                                 format(measure_name))

        self.active_measures = measure_names
        self.number_of_classes = number_of_classes

        self.measures_history = {}
        self.initialize_history()

    def add_measure(self, measure_name):
        self.active_measures.append(measure_name)

    def add_custom_measure_value(self, measure_name, value):
        if not measure_name in self.active_measures:
            self.add_measure(measure_name)

        if not measure_name in self.measures_history.keys():
            self.measures_history[measure_name] = []
        else:
            self.measures_history[measure_name].append(value)

    def initialize_history(self):
        """
        Resets the history held by the evaluator. Should be done after every epoch.
        """
        for measure_name in self.active_measures:
            self.measures_history[measure_name] = []

    def evaluate(self, prediction, annotation, valid_indices=None):
        """
        Evaluates the activated measures onto the provided prediction and annotation. The computed measures
        are then stored within the internal evaluator's history and accessible at any time until the history
        gets cleared explicitly.

        :param prediction: The predicted image.
        :param annotation: The annotation image.
        """

        prediction = prediction[valid_indices].flatten()
        annotation = annotation[valid_indices].flatten()

        for measure_name in self.active_measures:
            if measure_name in self._get_available_measures_with_functions.keys():
                measure_function = self._get_available_measures_with_functions(
                )[measure_name]
                self.measures_history[measure_name].append(
                    measure_function(prediction, annotation))

    def get_averaged_measures(self, current_epoch):
        """
        Computes the average of every activated measure.

        :param current_epoch:   The epoch that corresponds to the moment the averages are computed. This epoch is
                                prepended to any measure for results formatting.
        :return: An ordered dictionary of the averaged activated measures, prepended by the epoch at which these
                 were calculated.
        """
        averaged_measures = OrderedDict({'epoch': current_epoch})

        for measure_name in self.active_measures:
            averaged_measures.update({
                measure_name:
                np.mean(self.measures_history[measure_name])
            })

        return averaged_measures

    def _get_available_measures_with_functions(self):
        """
        :return: A dictionary of the measures the evaluator is able to compute, mapping their name to their actual
                 implementation.
        """
        return {
            'accuracy': self.compute_accuracy,
            'accuracy_per_class': self.compute_accuracy_per_class,
            'precision': self.compute_precision,
            'recall': self.compute_recall,
            'f1': self.compute_f1,
            'iou': self.compute_mean_iou,
            'exact_match_ratio': self.compute_exact_match_ratio,
            'hamming_score': self.compute_hamming_score
        }

    @staticmethod
    def compute_precision(prediction, annotation):
        return precision_score(
            y_true=annotation,
            y_pred=prediction,
            average='weighted',
            labels=np.unique(annotation))

    @staticmethod
    def compute_recall(prediction, annotation):
        return recall_score(
            y_true=annotation,
            y_pred=prediction,
            average='weighted',
            labels=np.unique(annotation))

    @staticmethod
    def compute_f1(prediction, annotation):
        return f1_score(
            y_true=annotation,
            y_pred=prediction,
            average='weighted',
            labels=np.unique(annotation))

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
        unique_labels, counts = np.unique(annotation, return_counts=True)
        num_unique_labels = len(unique_labels)

        I = np.zeros(num_unique_labels)
        U = np.zeros(num_unique_labels)
        # weights = np.ones(num_unique_labels)

        for index, val in enumerate(unique_labels):
            pred_i = prediction == val
            label_i = annotation == val

            I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
            U[index] = float(np.sum(np.logical_or(label_i, pred_i)))
            # weights[index] = 1 / counts[index]

        mean_iou = np.mean(I / U)

        return mean_iou

        # return jaccard_similarity_score(annotation, prediction, sample_weight=weights)

    @staticmethod
    def compute_exact_match_ratio(prediction, annotation):
        return accuracy_score(
            y_true=annotation, y_pred=prediction, normalize=True)

    @staticmethod
    def compute_hamming_score(prediction, annotation):
        """
        Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
        https://stackoverflow.com/q/32239577/395857
        """
        acc_list = []
        for i in range(annotation.shape[0]):
            annotation_set = set(np.where(annotation[i])[0])
            prediction_set = set(np.where(prediction[i])[0])

            if len(annotation_set) == 0 and len(prediction_set) == 0:
                tmp_a = 1
            else:
                tmp_a = len(
                    annotation_set.intersection(prediction_set)) / float(
                        len(annotation_set.union(prediction_set)))

            acc_list.append(tmp_a)

        return np.mean(acc_list)
