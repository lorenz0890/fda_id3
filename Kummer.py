import matplotlib.pyplot as plt
import numpy as np
import os
import random
import copy as cp

max_num_nodes_n = 0# just for initializing

def ID3(feature_set_indices, d, data):
    #print('recursion depth: ' + str(n))
    #check termination criteria
    global max_num_nodes_n
    max_num_nodes_n -= 1
    if (check_common_label(data) == 'republican'):
        return "republican"
    if (check_common_label(data) == 'democrat'):
        return "democrat"
    if len(feature_set_indices) == 0 or max_num_nodes_n <= 0 or d <=0:
        return find_majority_label(data)

    #find feature with biggest gain
    max_gain = -1 #initialized iwth -1 because real gain is always > 0
    max_gain_feature = None
    for i in feature_set_indices:
        gain = compute_gain(data, i)
        if max_gain < gain:
            max_gain = gain
            max_gain_feature = i

    #split the dataset along the max gain feature -> migrate to method
    remove_yes = []
    remove_no = []
    data_no = cp.deepcopy(data)
    data_yes = cp.deepcopy(data)
    for row_ind,_ in enumerate(data):
        if data[row_ind][max_gain_feature] == 'y':
            remove_yes.append(row_ind)
            continue
        if data[row_ind][max_gain_feature] == 'n':
            remove_no.append(row_ind)

    data_no = np.delete(data_no, remove_yes, axis=0)
    data_yes = np.delete(data_yes, remove_no, axis=0)

    #remove the max gain feature from the featzure set
    feature_set_indices_new = np.delete(feature_set_indices, np.where(feature_set_indices == max_gain_feature))

    #generate a node
    node = {}
    node [max_gain_feature] = {}

    node[max_gain_feature]['n']=(ID3(feature_set_indices_new, d - 1, data_no))
    node[max_gain_feature]['y']=(ID3(feature_set_indices_new, d - 1, data_yes))
    return node

def compute_gain(S, i):
    return compute_entropy(S) - compute_conditional_entropy(S, i)

def split_data(data, split):
    #split is the percentage allocated for training
    #shuffel array first to guarantee random split
    shuffeled_data = np.random.permutation(data)
    split_data = np.split(shuffeled_data, [int(len(shuffeled_data)*split)], axis = 0)
    return split_data[1], split_data[0] #trainig_set, test_set

def learning_curve(d, n, training_set, test_set, num_increments):

    #first, we split the training set into num_increments parts portions of different size
    global max_num_nodes_n
    max_num_nodes_n = n

    training_set_parts = []
    working_set = cp.deepcopy(training_set)
    part_length = 1
    while len (training_set_parts) < num_increments and part_length<len(working_set):
        training_set_parts.append(working_set[:part_length])
        part_length +=int(len(training_set)/num_increments+1)

    #now for each portion of the training set, we calculate the test and training errors
    training_errors = []
    test_errors = []
    training_set_parts_lengths = []

    training_error_mean = 0.0
    test_error_mean = 0.0
    for training_set_partial in training_set_parts:
        # make n_refs
        feature_subset_indices = np.arange(1, 17)
        max_num_nodes_n = n
        decision_tree = ID3(feature_subset_indices, d, training_set_partial)

        training_error_partial = compute_error(decision_tree, training_set_partial, feature_subset_indices)
        training_error_mean += training_error_partial/len(training_set_parts)
        training_errors.append(training_error_partial*100)

        test_error_partial = compute_error(decision_tree, test_set, feature_subset_indices)
        test_error_mean += test_error_partial/len(training_set_parts)
        test_errors.append(test_error_partial*100)

        training_set_parts_lengths.append(len(training_set_partial))


    plt.figure(figsize=(10, 10))
    plt.plot(training_set_parts_lengths, training_errors, label = 'training')
    plt.plot(training_set_parts_lengths, test_errors, label = 'test')
    plt.title('Evaluation, #features = {}, max number of nodes = {}'.format(d, n))
    plt.xlabel('training set length (rows)')
    plt.ylabel('error (% of mis-classifications)')
    plt.legend(loc='upper right')

    return plt


#Helper funcs:
def compute_error(decision_tree, data_set, d):
    err = 0.0
    for row_ind, row in enumerate(data_set):
        true = row[0]
        found = traverse_tree(decision_tree, row, d)
        if found != true : err+=1
    return err/len(data_set)


def traverse_tree(decision_tree, evaluation_row, d):
    for col_ind, col in enumerate(evaluation_row):
        if col_ind > 0 and np.isin(col_ind, d):
            try:
                if type(decision_tree) == type({}):
                    decision_sub_tree = decision_tree[col_ind][col]
                    return traverse_tree(decision_sub_tree, evaluation_row, np.delete(d, np.where(d == col_ind)))

                if type(decision_tree) == type(""):
                    return decision_tree

                raise Exception("fatal error during traversal - decision tree broken")
            except KeyError:
                continue

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


def compute_entropy(data):
    data_t = data.transpose()
    num_democrats = 0
    num_republicans = 0
    for ind_col, _ in enumerate(data_t[0]):
        if data_t[0][ind_col] == 'democrat':
            num_democrats+=1
            continue
        if data_t[0][ind_col] == 'republican':
            num_republicans += 1
            continue

    probability_democrat = 0
    probability_republican = 0
    if len(data_t[0]) > 0:
        probability_democrat = num_democrats/len(data_t[0])
        probability_republican = num_republicans/len(data_t[0])

    return -(probability_democrat*np.log2(probability_democrat) + probability_republican*np.log2(probability_republican))


def compute_conditional_entropy(data, i):
    # i is column of feature, i must be greater than 0
    # source for formula: https://en.wikipedia.org/wiki/Conditional_entropy

    v= data[:,i]
    p = data[:,0]
    feature = []
    for i in range(0, len(v)):
       feature.append([p[i], v[i]])
    feature = np.array(feature)

    num_yes = 0
    num_no = 0
    num_republican = 0
    num_democrat = 0
    num_republican_yes = 0
    num_republican_no = 0
    num_democrat_yes = 0
    num_democrat_no = 0
    for ind_row, _ in enumerate(feature):
        if feature[ind_row][0] == 'democrat':
            num_democrat +=1
            if feature[ind_row][1] == 'y':
                num_democrat_yes +=1
                num_yes += 1
            else:
                num_democrat_no = +1
                num_no += 1
        if feature[ind_row][0] == 'republican':
            num_republican += 1
            if feature[ind_row][1] == 'y':
                num_republican_yes += 1
                num_yes += 1
            else:
                num_republican_no += 1
                num_no += 1

    entropy = 1.0
    if num_republican > 0 and num_democrat > 0 and len(feature) > 0:
        probability_yes = num_yes/len(feature)
        probability_no = num_no/len(feature)

        probability_republican_condition_yes = num_republican_yes/num_republican
        probability_republican_condition_no = num_republican_yes/num_republican

        probability_democrat_condition_yes = num_democrat_yes/num_democrat
        probability_democrat_condition_no = num_democrat_no/num_democrat

        probability_democrat_joint_yes = probability_democrat_condition_yes*probability_yes
        probability_republican_joint_yes = probability_republican_condition_yes*probability_yes

        probability_democrat_joint_no = probability_democrat_condition_no*probability_no
        probability_republican_joint_no = probability_republican_condition_no*probability_no

        if probability_republican_joint_yes > 0 \
                and probability_democrat_joint_yes > 0 \
                and probability_republican_joint_no > 0 \
                and probability_democrat_joint_no > 0:
            entropy = probability_democrat_joint_yes*np.log2(probability_yes/probability_democrat_joint_yes) \
                      + probability_republican_joint_yes*np.log2(probability_yes/probability_republican_joint_yes) \
                      + probability_republican_joint_no * np.log2(probability_no / probability_republican_joint_no) \
                      + probability_democrat_joint_no * np.log2(probability_no / probability_democrat_joint_no)
    return entropy


def load_data(filename):
    #load data from disk, return np md array containing data
    filename, ftype = os.path.splitext(filename)
    raw_data = None
    if ftype == '.data':
        raw_data = np.genfromtxt((filename+ftype), delimiter=',', dtype=str
                                      )
        print(type(raw_data))
    return raw_data


def clean_data(data):
    # data cleaning form the np md array received from load_data, returns np md array but cleaned
    # uniform distributed varaible y or n replaces ?
    # we use deep copy because we avoid side effects in (somehow) functional language. costs some peformance and memory though.

    cleaned_data = cp.deepcopy(data)#[:,1:]
    for ind_row, _ in enumerate(cleaned_data):
        for ind_col, _ in enumerate(cleaned_data[ind_row]):
            if cleaned_data[ind_row][ind_col] == '?' and ind_col > 0:
                if (random.randint(0,1)) == 0:
                    cleaned_data[ind_row][ind_col] = 'y'
                else:
                    cleaned_data[ind_row][ind_col] = 'n'
    return cleaned_data

#If we want to run as a script using some test data
if __name__ == '__main__':
    try:
        #Load data
        print('load, clean, split data set house-votes-84.data')
        training_set, test_set = split_data(clean_data(load_data('house-votes-84.data')), 0.3)
        print('complete')

        #Run the learning curve script with teh required inputs from the assignment, output the plots
        print('calculate learning curve for input: d  = 7, n = 40, num_increments = 50 with output: fig1.png')
        learning_curve(d=7,n = 40, training_set=training_set, test_set=test_set, num_increments=100).savefig('fig1.png')

        print('complete\ncalculate learning curve for input: d  = 4, n = 40, num_increments = 50 with output: fig2.png')
        learning_curve(d=4,n = 40, training_set=training_set, test_set=test_set, num_increments=100).savefig('fig2.png')

        print('complete\ncalculate learning curve for input: d  = 16, n = 40, num_increments = 50 with output: fig3.png')
        learning_curve(d=16, n=40, training_set=training_set, test_set=test_set, num_increments=100).savefig('fig3.png')

        print('complete\ncalculate learning curve for input: d  = 7, n = 20, num_increments = 50 with output: fig4.png')
        learning_curve(d=7, n=20, training_set=training_set, test_set=test_set, num_increments=100).savefig('fig4.png')

        print('complete\ncalculate learning curve for input: d  = 7, n = 130, num_increments = 50 with output: fig5.png')
        learning_curve(d=7, n=130, training_set=training_set, test_set=test_set, num_increments=100).savefig('fig5.png')
        print('complete')

        #Run my own experiments regarding  good fit

        print('running experiments for finding best learning curve. output: fit_experiment_d_n.png for d from 4 to 16 and n from 2 to 32, num_increments = 50. '
              '\nthis may take a while and generate a lot of files. interrupt with ctrl+c to stop.')
        for i in range (1,17):
            n = 1

            while n <= 60:
                learning_curve(d=i, n=n, training_set=training_set, test_set=test_set, num_increments=100).savefig('fit_experiment_{}_{}.png'.format(i,n))
                n+=2
                print('complete for d = {}, n = {}'.format(i,n))
        print('complete')

    except:
        print("fatal error. is house-votes-84.data in the same location as Kummer.py?")
    exit()