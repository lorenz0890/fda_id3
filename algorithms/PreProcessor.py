import numpy as np
import os
import random
import copy as cp

'''
7. Attribute Information:
   1. Class Name: 2 (democrat, republican) #policy 0
   2. handicapped-infants: 2 (y,n)
   3. water-project-cost-sharing: 2 (y,n)
   4. adoption-of-the-budget-resolution: 2 (y,n)
   5. physician-fee-freeze: 2 (y,n)
   6. el-salvador-aid: 2 (y,n)
   7. religious-groups-in-schools: 2 (y,n)
   8. anti-satellite-test-ban: 2 (y,n)
   9. aid-to-nicaraguan-contras: 2 (y,n)
  10. mx-missile: 2 (y,n)
  11. immigration: 2 (y,n)
  12. synfuels-corporation-cutback: 2 (y,n)
  13. education-spending: 2 (y,n)
  14. superfund-right-to-sue: 2 (y,n)
  15. crime: 2 (y,n)
  16. duty-free-exports: 2 (y,n)
  17. export-administration-act-south-africa: 2 (y,n)
'''


def load_data(filename):
    #load data from disk, return np md array containing data
    filename, ftype = os.path.splitext(filename)
    raw_data = None
    if ftype == '.data':
        raw_data = np.genfromtxt((filename+ftype), delimiter=',', dtype=str
                                      )
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
