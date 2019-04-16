import unittest
import algorithms.PreProcessor as prep
import algorithms.ID3 as id3
import numpy as np

class IDTests(unittest.TestCase):

    def test_slpit_data_training(self):
        clean_data = prep.clean_data(prep.load_data('../data/house-votes-84.data'))
        training_set, _ = id3.split_data(clean_data, 0.2)
        assert len(training_set) <= int(len(clean_data)*0.8)+1

    def test_slpit_data_test(self):
        clean_data = prep.clean_data(prep.load_data('../data/house-votes-84.data'))
        _, test_set = id3.split_data(clean_data, 0.2)
        assert len(test_set) <= int(len(clean_data)*0.2) +1

    def test_compute_per_col_yes_probability(self):
        probs = id3.compute_per_col_yes_probability(prep.clean_data(prep.load_data('../data/house-votes-84.data')))
        for prob in probs:
            assert prob >= 0 and prob <= 1

    def test_compute_per_col_yes_probability_feature(self):
        training_set, _ = id3.split_data(prep.clean_data(prep.load_data('../data/house-votes-84.data')), 0.2)
        data_nofeature = np.delete(training_set, 2, axis=0)
        v = training_set[:, 2]
        p = training_set[:, 0]
        feature = []
        for i in range(0, len(v)):
            feature.append([p[i], v[i]])
        feature = np.array(feature)
        probs = id3.compute_per_col_yes_probability(feature)
        assert probs[0] > 0. and probs[0] <= 1.0

    def test_compute_entropy(self):
        training_set, test_set = id3.split_data(prep.clean_data(prep.load_data('../data/house-votes-84.data')), 0.2)
        entropy_training = id3.compute_entropy(training_set)
        entropy_test = id3.compute_entropy(test_set)
        assert entropy_training > 0
        assert entropy_test > 0

    def test_compute_conditional_entropy(self):
        training_set, _ = id3.split_data(prep.clean_data(prep.load_data('../data/house-votes-84.data')), 0.2)
        entropy_training = id3.compute_conditional_entropy(training_set, 1)
        print(entropy_training)
        assert entropy_training > 0

    def test_compute_gain(self):
        _, test_set = id3.split_data(prep.clean_data(prep.load_data('../data/house-votes-84.data')), 0.2)
        gain = id3.compute_gain(test_set,8)
        assert gain > 0

    def test_check_common_label_democrat(self):
        training_set, _ = id3.split_data(prep.clean_data(prep.load_data('../data/house-votes-84.data')), 0.2)
        for ind_row, _ in enumerate(training_set):
            if training_set[ind_row][0] == 'republican':
                training_set[ind_row][0] = 'democrat'
        assert id3.check_common_label(training_set) == 'democrat'

    def test_check_common_label_republican(self):
        training_set, _ = id3.split_data(prep.clean_data(prep.load_data('../data/house-votes-84.data')), 0.2)
        for ind_row, _ in enumerate(training_set):
            if training_set[ind_row][0] == 'democrat':
                training_set[ind_row][0] = 'republican'
        assert id3.check_common_label(training_set) == 'republican'

    def test_find_common_label_none(self):
        training_set, _ = id3.split_data(prep.clean_data(prep.load_data('../data/house-votes-84.data')), 0.2)
        assert id3.check_common_label(training_set) == 'heterogenous'


    def test_find_majority_label_democrat(self):
        training_set, _ = id3.split_data(prep.clean_data(prep.load_data('../data/house-votes-84.data')), 0.2)
        for ind_row, _ in enumerate(training_set):
            if training_set[ind_row][0] == 'democrat':
                training_set[ind_row][0] = 'republican'
        assert id3.find_majority_label(training_set) == 'republican'

    def test_find_majority_label_republican(self):
        training_set, _ = id3.split_data(prep.clean_data(prep.load_data('../data/house-votes-84.data')), 0.2)
        for ind_row, _ in enumerate(training_set):
            if training_set[ind_row][0] == 'democrat':
                training_set[ind_row][0] = 'republican'
        assert id3.find_majority_label(training_set) == 'republican'

    def test_ID3(self):
        training_set, test_set = id3.split_data(prep.clean_data(prep.load_data('../data/house-votes-84.data')), 0.2)
        decision_tree = id3.ID3(np.arange(17),50,training_set) # max num policies is 16
        print(decision_tree)

