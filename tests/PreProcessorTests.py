import unittest
import algorithms.PreProcessor as prep

class PreprocessorTests(unittest.TestCase):

    def test_load_data_none(self):
        data = prep.load_data('../data/house-votes-84.data')
        assert data is not None

    def test_load_data_len(self):
        data = prep.load_data('../data/house-votes-84.data')
        assert len(data) > 0

    def test_load_data_len_all(self):
        data = prep.load_data('../data/house-votes-84.data')
        for k in range(0, len(data)):
            assert len(data[k]) > 0

    def test_clean_data(self):
        cleaned_data = prep.clean_data(prep.load_data('../data/house-votes-84.data'))
        for ind_row, _ in enumerate(cleaned_data):
            for ind_col, _ in enumerate(cleaned_data[ind_row]):
                assert cleaned_data[ind_row][ind_col] != '?'
