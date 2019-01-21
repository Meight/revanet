"""
Utility classes to help manage the file naming by keeping track of the execution's parameters.

This is especially useful when running extensive experiments on various models and/or datasets as result data can
accumulate quite quickly.
"""

import os
from pathlib import Path

from colorama import Back, Fore, Style
from colorama import init as colorama_init


class FilesFormatterFactory:
    """
    A factory used to build various file formatters with different strategies.
    """
    def __init__(self, mode,
                 dataset_name=None,
                 model_name=None,
                 backbone_name=None,
                 training_parameters=None,
                 train_path: Path=Path(),
                 results_folder='results',
                 verbose=False):
        """

        :param mode:                The mode the model is in. Can be one of `training', `validation` or `test`.
        :param dataset_name:        The name of the root directory of the dataset used in the current session.
        :param model_name:          The name of the model used in the current session.
        :param backbone_name:       The name of the backbone used in the current session.
        :param training_parameters: A dictionary containing an arbitrary number of training parameters.
        :param results_folder:      The name of the root directory where to store the session's results.
        :param verbose:             Whether or not to print intermediate steps.
        """
        self.mode = mode
        self.training_parameters = training_parameters
        self.backbone_name = backbone_name
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.verbose = verbose

        self.train_folder = train_path.name

        self.results_folder = os.path.join('datasets', dataset_name, results_folder)
        self._parameters_string = self._generate_parameters_string()
        self._full_detailed_path = self._generate_full_detailed_path()

        if not self._full_detailed_path.exists():
            self._full_detailed_path.mkdir(parents=True, exist_ok=True)

    def _generate_full_detailed_path(self):
        return Path(self.results_folder,
                    self.mode,
                    self.backbone_name + '-' + self.model_name)

    def _generate_parameters_string(self):
        return '_'.join(['{}-{}'.format(self._get_initials(parameter_name), self._format_parameter(parameter_value))
                         for parameter_name, parameter_value
                         in self.training_parameters.items()
                         if parameter_name not in ['train_path',
                                                   'validation_images_path',
                                                   'validation_annotations_path']])

    @staticmethod
    def _format_parameter(parameter):
        if isinstance(parameter, Path):
            return parameter.name.upper()

        return parameter

    @staticmethod
    def _get_initials(string):
        return ''.join([x[0].upper() for x in string.split('_')])

    def generate_checkpoint_name(self, current_epoch):
        return os.path.join(self._full_detailed_path, self._parameters_string + '.ckpt')

    def generate_summary_name(self, current_epoch):
        return os.path.join(self._full_detailed_path, self._parameters_string + '.csv')

    def generate_logs_name(self):
        return os.path.join(self._full_detailed_path, self._parameters_string + '.log')

    def get_checkpoint_formatter(self, saver):
        """
        Generates a name formatter that handles the saving and restoration of checkpoint files associated to the current
        session.

        :param saver:   Instance of a `tf.Saver` that's linked to the current session.
        :return:        The generated checkpoint formatter.
        """
        return CheckpointFormatter(mode=self.mode,
                                   dataset_name=self.dataset_name,
                                   model_name=self.model_name,
                                   backbone_name=self.backbone_name,
                                   training_parameters=self.training_parameters,
                                   verbose=self.verbose,
                                   saver=saver)

    def get_summary_formatter(self):
        """
        Generates a name formatter that handles the summaries of the current session.

        :return: The generated summary formatter.
        """
        return SummaryFormatter(mode=self.mode,
                                dataset_name=self.dataset_name,
                                model_name=self.model_name,
                                backbone_name=self.backbone_name,
                                training_parameters=self.training_parameters,
                                verbose=self.verbose)

    def get_logs_formatter(self):
        return LogsFormatter(mode=self.mode,
                             dataset_name=self.dataset_name,
                             model_name=self.model_name,
                             backbone_name=self.backbone_name,
                             training_parameters=self.training_parameters,
                             verbose=self.verbose)


class SummaryFormatter(FilesFormatterFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.header_created = False
        self.current_epoch = 0

    def update(self, current_epoch, measures_dictionary, column_margin=2, precision=3, verbose=False):
        """
        Updates the summary file associated with the current ongoing session.

        :param current_epoch:       The epoch associated to the record that's about to be written.
        :param measures_dictionary: An ordered dictionary containing an arbitrary set of measures.
        :param column_margin:       The width in characters of each column in the resulting summary file.
        :param precision:           The floating point precision to print the measures in the dictionary.
        :param verbose:             Whether or not to print intermediate steps.
        """
        self.current_epoch = current_epoch
        column_width = len(max(measures_dictionary.keys(), key=len)) + column_margin

        with open(self.generate_summary_name(current_epoch=self.current_epoch), 'a+') as summary_file:
            if not self.header_created:
                summary_file.write(self._generate_header(column_names=measures_dictionary.keys(),
                                                         column_width=column_width))
                self.header_created = True

            for measure in measures_dictionary.values():
                summary_file.write('{value:<{width}.{precision}f}'.format(value=measure,
                                                                          width=column_width,
                                                                          precision=precision))

            summary_file.write('\n')

        if self.verbose or verbose:
            print('Updated session summary at {}.'.format(self.generate_summary_name(self.current_epoch)))

    @staticmethod
    def _generate_header(column_names, column_width):
        return ''.join(['{0:<{width}}'.format(column_name, width=column_width)
                        for column_name in column_names]) + '\n'


class CheckpointFormatter(FilesFormatterFactory):
    def __init__(self, saver=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saver = saver
        self.current_epoch = 0

    def save(self, session, current_epoch, verbose=False):
        """
        Saves a checkpoint file following the naming policy defined by this formatter.

        :param session:         The ongoing `tf.Session`.
        :param current_epoch:   The epoch associated to the current checkpoint file.
        :param verbose:         Whether or not print intermediate steps.
        """
        self.current_epoch = current_epoch
        self.saver.save(session, self.generate_checkpoint_name(current_epoch=self.current_epoch))

        if self.verbose or verbose:
            print('Saved checkpoints for epoch {} at {}.'.format(self.current_epoch,
                                                                 self.generate_checkpoint_name(
                                                                     current_epoch=self.current_epoch)))

    def restore(self, session, model_checkpoint_name):
        self.saver.restore(session, model_checkpoint_name)


class LogsFormatter(FilesFormatterFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        colorama_init()

    def write(self, additional_info=None):
        all_info = {**additional_info, **self.training_parameters}
        header_width = max([len(header) for header in all_info.keys()])

        for header, value in all_info.items():
            print(self._pretty_print_line(header, value, width=header_width + 2))

        if self.verbose:
            print('Logs written at {}'.format(self.generate_logs_name()))

    def _pretty_print_line(self, header, value, tabs=0, width=10):
        return '{spacing}{header:{width}}{value}'.format(spacing=' ' * tabs,
                                                         header=header.capitalize().replace('_', ' ') + ':',
                                                         width=width,
                                                         value=self._pretty_print_value(value)) + Style.RESET_ALL

    def _pretty_print_value(self, value):
        if isinstance(value, bool):
            return self._pretty_print_boolean(value)
        else:
            return Fore.YELLOW + '{}'.format(value)

    @staticmethod
    def _pretty_print_boolean(value):
        return Fore.RED + 'no' if not value else Fore.GREEN + 'yes'
