import numpy as np
import algorithms.PreProcessor as prep
from matplotlib import pyplot


def ID3(d, n, data):
    # you will probably need additional helper functions
    #return tree
    pass


def compute_gain(S, i):
    #return gain
    pass

def split_data(data, split):
    #split is the percentage allocated for training
    #shuffel array first to guarantee random split
    shuffeled_data = np.random.permutation(data)
    split_data = np.split(shuffeled_data, [int(len(shuffeled_data)*split)], axis = 0)

    return split_data[1], split_data[0] #trainig_set, test_set

def learning_curve(d, n, training_set, test_set):
    # you will probably need additional helper functions
    #return plot
    pass



#Helper funcs:
def calc_per_col_yes_probability(data):
    # calculates the per column probabilities that a varaible, i.e. the column will have a certain value (y,n).
    # can be used on a per-dataset basisis, i.e. we can calc different per col probabilities for training and validation set
    prob_yes = []
    for ind_row, _ in enumerate(data):
        prob_yes.append(0)
        for ind_col, _ in enumerate(data[ind_row]):
            if data[ind_row][ind_col] == 'y' and ind_col > 0:
                    prob_yes[ind_row] += 1.0
        prob_yes[ind_row] /= float(len(data[ind_row]))
    return np.array(prob_yes)


def calc_entropy(data):
    probs_yes = calc_per_col_yes_probability(data)
    for ind_prob_yes, prob_yes in enumerate(probs_yes):
        probs_yes[ind_prob_yes] = prob_yes*np.log(prob_yes)
    return (-1)*np.sum(probs_yes)

def calc_conditional_entropy(data):
    probs_yes = calc_per_col_yes_probability(data)

    pass
