import numpy as np
import algorithms.PreProcessor as prep
from matplotlib import pyplot
import copy as cp

def ID3(d, n, data):
    print('recursion depth: ' + str(n))
    #check termination criteria
    if (check_common_label(data) == 'republican'):
        return "leaf republican"
    if (check_common_label(data) == 'democrat'):
        return "leaf democrat"
    if len(d) == 0 or n == 0:
        return "leaf " + find_majority_label(data)

    #find feature with biggest gain
    max_gain = -1 #initialized iwth -1 because real gain is always > 0
    max_gain_feature = None
    for i in d:
        gain = compute_gain(data, i)
        if max_gain < gain:
            max_gain = gain
            max_gain_feature = i

    #split the dataset along the max gain feature -> migrate to method
    remove_yes = []
    remove_no = []
    data_yes = cp.deepcopy(data)
    data_no = cp.deepcopy(data)
    for row_ind,_ in enumerate(data):
        if data[row_ind][max_gain_feature] == 'y':
            remove_yes.append(row_ind)
            continue
        if data[row_ind][max_gain_feature] == 'n':
            remove_no.append(row_ind)

    data_yes = np.delete(data_yes, remove_yes, axis=0)
    data_no = np.delete(data_no, remove_no, axis=0)

    #remove the max gain feature from the featzure set
    d_new = np.delete(d, np.where(d==max_gain_feature))

    #generate a node
    node = {}
    node ['policy'+str(max_gain_feature)] = {}
    node['policy'+str(max_gain_feature)]['yes']=(ID3(d_new , n-1, data_yes))
    node['policy'+str(max_gain_feature)]['no']=(ID3(d_new, n-1, data_no))
    return node

def compute_gain(S, i):
    #print('entropy' + str(compute_entropy(S)))
    #print('entropy cond ' + str(compute_conditional_entropy(S, i)))
    return compute_entropy(S) - compute_conditional_entropy(S, i)

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
def find_majority_label(data):
    rep_count = 0
    dem_count = 0
    for ind_row, _ in enumerate(data):
        if data[ind_row][0]=='republican':
            rep_count += 1
            continue
        if data[ind_row][0]=='democrat':
            dem_count += 1
            continue
    return 'democrat' if dem_count>=rep_count else 'republican'


def check_common_label(data):
    if all(party == 'republican' for party in data[:, 0]):
        return 'republican'
    if all(party == 'democrat' for party in data[:, 0]):
        return 'democrat'
    return 'heterogenous'



def compute_per_col_yes_probability(data):
    # calculates the per column probabilities that a varaible, i.e. the column will have a certain value (y,n).
    # can be used on a per-dataset basisis, i.e. we can calc different per col probabilities for training and validation set
    prob_yes = []
    data_t = data.transpose()
    for ind_row, _ in enumerate(data_t):
        prob_yes.append(0)
        for ind_col, _ in enumerate(data_t[ind_row]):
            if data_t[ind_row][ind_col] == 'y':
                    prob_yes[ind_row] += 1.0
        new_elem = prob_yes[ind_row] / float(len(data_t[ind_row]))
        if new_elem > 0.:
            prob_yes[ind_row] = new_elem
        else:
            prob_yes[ind_row] = 0.001

    prob_yes = np.array(prob_yes)
    prob_yes = prob_yes[1:]
    return np.array(prob_yes)

def compute_entropy(data):
    probs_yes = compute_per_col_yes_probability(data)
    for ind_prob_yes, prob_yes in enumerate(probs_yes):
        probs_yes[ind_prob_yes] = prob_yes*np.log(prob_yes)
    return (-1)*np.sum(probs_yes)

def compute_conditional_entropy(data, i):
    #i is column of feature, i must be greater than 0
    # source for formula: https://en.wikipedia.org/wiki/Conditional_entropy
    data_nofeature = np.delete(data, i, axis = 1)

    v= data[:,i]
    p = data[:,0]
    feature = []
    for i in range(0, len(v)):
        feature.append([p[i], v[i]])
    feature = np.array(feature)

    probs_yes_data_nofeature = compute_per_col_yes_probability(data_nofeature)
    probs_yes_feature = compute_per_col_yes_probability(feature)

    vals = []
    for ind_nofeat, prob_yes_nofeat in enumerate(probs_yes_data_nofeature):
        for ind_feat, prob_yes_feat in enumerate(probs_yes_feature):
            prob_conditional = prob_yes_nofeat * prob_yes_feat
            vals.append(prob_conditional* np.log(prob_conditional/prob_yes_feat))
    return -1*np.sum(vals)
