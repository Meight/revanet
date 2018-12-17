def _create_validation_measures(validation_measures):
    assert not validation_measures

    measures_dictionary = {}
    for measure_name in validation_measures:
        measures_dictionary[measure_name] = []

    return measures_dictionary
