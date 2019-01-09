import argparse


def ratio(value):
    value = float(value)

    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError('{} is not a valid ratio: must be within ]0, 1].'.format(value))

    return value