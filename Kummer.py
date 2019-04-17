import matplotlib.pyplot as plt
import numpy as np
import algorithms.ID3 as id3
import algorithms.PreProcessor as prep

#If we want to run as a script using some test data
if __name__ == '__main__':
    #Load data
    training_set, test_set = id3.split_data(prep.clean_data(prep.load_data('./data/house-votes-84.data')), 0.3)

    #Run the learning curve script with teh required inputs from the assignment, output the plots
    id3.learning_curve(d=np.arange(1,8),n = 40, training_set=training_set, test_set=test_set, num_increments=100).savefig('fig1.png')
    id3.learning_curve(d=np.arange(1,5),n = 40, training_set=training_set, test_set=test_set, num_increments=100).savefig('fig2.png')
    id3.learning_curve(d=np.arange(1,17), n=40, training_set=training_set, test_set=test_set, num_increments=100).savefig('fig3.png')
    id3.learning_curve(d=np.arange(1,8), n=20, training_set=training_set, test_set=test_set, num_increments=100).savefig('fig4.png')
    id3.learning_curve(d=np.arange(1, 8), n=130, training_set=training_set, test_set=test_set, num_increments=100).savefig('fig5.png')

    #Run my own experiments regarding  good fit
    '''
    for i in range (5,17):
        n = 5
        while n <= 80:
            id3.learning_curve(d=np.arange(1, i), n=n, training_set=training_set, test_set=test_set, num_increments=100).savefig('figures/good_fit_{}{}.png'.format(i,n))
            n+=25
    '''
    exit()