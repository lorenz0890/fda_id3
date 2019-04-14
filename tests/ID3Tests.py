import unittest
import algorithms.PreProcessor as prep
import algorithms.ID3 as id3

class IDTests(unittest.TestCase):

    def test_slpit_data_training(self):
        clean_data = prep.clean_data(prep.load_data('../data/house-votes-84.data'))
        training_set, _ = id3.split_data(clean_data, 0.2)
        assert len(training_set) <= int(len(clean_data)*0.8)+1

    def test_slpit_data_test(self):
        clean_data = prep.clean_data(prep.load_data('../data/house-votes-84.data'))
        _, test_set = id3.split_data(clean_data, 0.2)
        assert len(test_set) <= int(len(clean_data)*0.2) +1

    def test_calc_per_col_yes_probability(self):
        probs = id3.calc_per_col_yes_probability(prep.clean_data(prep.load_data('../data/house-votes-84.data')))
        for prob in probs:
            assert prob >= 0 and prob <= 1

    def test_calc_entropy(self):
        training_set, test_set = id3.split_data(prep.clean_data(prep.load_data('../data/house-votes-84.data')), 0.2)
        entropy_training = id3.calc_entropy(training_set)
        entropy_test = id3.calc_entropy(test_set)
        assert entropy_training > 0
        assert entropy_test > 0
