import pandas as pd
import numpy as np
import sys, gc
import matplotlib.pylab as plt
import operator
import multiprocessing as mp
import config
import matplotlib.pylab as plt
plt.ion()

from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.width', 2000)

def read_data(pct_to_keep=None, seed=None):
    '''Read pre-saved train and test datasets
    '''
    
    if pct_to_keep is None:
        train = pd.read_csv('data/cv_train.csv')
        test = pd.read_csv('data/cv_test.csv')
    else:
        if seed is None:
            np.random.seed(0)

        if pct_to_keep > 0 and pct_to_keep < 1:
            train = pd.read_csv('data/cv_train.csv', skiprows=lambda x: x>0 and np.random.random()>pct_to_keep)
            test = pd.read_csv('data/cv_test.csv', skiprows=lambda x: x>0 and np.random.random()>pct_to_keep)

    return train, test

def check_train_test_1hot(train, test, colname):
    '''Check train and test both have the same set of unique values for column colname
    '''
    train_vals = set(train[colname].unique())
    test_vals = set(test[colname].unique())
    
    union = train_vals.intersection(test_vals)
    
    if len(union) != len(train_vals): #test_vals subset of train_vals
        return False

    return True

def find_mi_threshold_parallel(train, colname, n_low = 1, n_high = None, n_jobs=1):
    '''compute mutual information if all values below a threshold and combined
       can be made smarter with binary search like algorithm
    '''

    values = train[colname].astype(str).fillna('null')
    
    value_counts = values.value_counts()

    if n_high is None:
        n_high = len(value_counts)
    
    train_df_orig = pd.DataFrame(values)
    
    manager = mp.Manager()
    mi_values = manager.dict()

    def calc_mi(value_counts, train_df_orig, colname, mi_values, i):
        top_values= list(value_counts[0:i].index)

        train_df = train_df_orig.copy()
        train_df[~train_df[colname].isin(top_values)] = 'rest'

        mi_values[i] = mutual_info_score(train_df[colname], train['HasDetections'])
    
    proc_list = []
    counter = 0
    for i in range(n_low, n_high+1): #can be made smarter
        proc = mp.Process(target=calc_mi, args=(value_counts, train_df_orig, colname, mi_values, i))
        proc.start()
        proc_list.append(proc)
        counter += 1
        
        if counter % n_jobs == 0:
            [p.join() for p in proc_list]
            proc_list = []
            
    [p.join() for p in proc_list]
        
    mi_values = dict(mi_values)
    
    return mi_values

def find_threshold(train, colname, n_iter, cutoff, debug=False):
    '''Do binary search to find appropriate cutoff
    '''
    train_col = train[colname].astype(str).fillna('null').copy()
    train_target = train['HasDetections']

    mi = mutual_info_score(train_col, train_target) #target

    value_counts = train_col.value_counts()
    N_values = len(value_counts)

    mid_point = int(N_values/2.)
    shift = int(N_values/4.) #make window size larger (3->5) to get rid of rounding issues

    counter = 0
    while counter < n_iter:
        mi_mid_point = find_mi_threshold_parallel(train, colname, n_low=mid_point-1, n_high=mid_point+1, n_jobs=3)

        mi_list = np.array([mi_mid_point[i] for i in range(mid_point-1, mid_point+2)])
        mi_list = (np.abs(mi_list - mi)/mi < cutoff).astype(int)

        if debug:
            print(f'mid_point={mid_point} - shift={shift} - values={mi_list}')

        #[False, False, False] -> to the left of transition point
        #[True, True, True] -> to the right of transition point
        #want mixed array
        truth_value = np.sum(mi_list)
        if truth_value == mi_list.shape[0]: #all Trues
            #move to left
            mid_point -= shift
        elif truth_value == 0:
            #move to right
            mid_point += shift
        else:
            #done
            for i in range(1, mi_list.shape[0]):
                if mi_list[i-1]!=mi_list[i]:
                    return mid_point -1 + i

        shift = max(int(shift/2), 1)
        counter += 1

    return None

def find_all_thresholds(train, col_list, n_iter, cutoff, debug=False):
    thresh_list = []

    for counter, colname in enumerate(col_list):
        thresh = find_threshold(train, colname, n_iter, cutoff, debug=debug)
        thresh_list.append(thresh)
        print(f'Thresh for col={colname} is {thresh}')

        #if counter == 1:
        #    return thresh_list

    return thresh_list

def create_one_hot(train, test, colnames, thresholds):
    '''Create one-hot encoded df with colnames
    '''
    assert(len(colnames)==len(thresholds))
    
    train_df = train[colnames].astype(str).fillna('null').copy()
    test_df = test[colnames].astype(str).fillna('null').copy()

    train_list, test_list = [], []
    for index, col in enumerate(colnames):
        if not check_train_test_1hot(train_df, test_df, col):
            print(f'Problem with col = {col}. Combining into "rest"')
            #continue

        value_counts = train_df[col].value_counts()
        
        top_values = list(value_counts[0:thresholds[index]].index)
        
        #two possibilities:
        #train has elements missing in test -> test has less columns than train
        #test has elements missing in train -> get assigned to 'rest' (and these elements are low in value_counts)

        train_df[~train_df[col].isin(top_values)] = f'rest_{col}'
        test_df[~test_df[col].isin(top_values)] = f'rest_{col}' #if value not in train, combined into rest
        
        train_list.append( pd.get_dummies(train_df[[col]], dummy_na=False) )
        test_list.append( pd.get_dummies(test_df[[col]], dummy_na=False) )

    train_df = pd.concat(train_list, axis=1)
    test_df = pd.concat(test_list, axis=1)
    
    return train_df, test_df

def replace_by_values(train, test, colnames):
    train_df = train[colnames].astype(str).fillna('null').copy()
    test_df = test[colnames].astype(str).fillna('null').copy()

    for index, col in enumerate(colnames):
        value_counts = train[col].value_counts()
        value_counts_sum = value_counts.sum().astype(float)

        value_counts_dict = (value_counts / value_counts_sum).to_dict()

        #null -> % of nulls (if nulls in train)
        #values in test not seen in train -> 0 (can't distinguish between them can technically 0% of time in train)
        train_df[col] = train_df[col].apply(lambda x: value_counts_dict.get(x, 0))
        test_df[col] = test_df[col].apply(lambda x: value_counts_dict.get(x, 0))

    return train_df, test_df

def build_model(model, train_df, test_df, train, test):
    '''Given a model instance, train and calculate train/test scores
    '''
    model.fit(train_df, train['HasDetections'])

    train_pred = model.predict_proba(train_df)[:,1] #prob of belonging to class 1
    test_pred = model.predict_proba(test_df)[:,1]

    train_labels = train['HasDetections']
    test_labels = test['HasDetections']

    train_score = roc_auc_score(train_labels, train_pred)
    test_score = roc_auc_score(test_labels, test_pred)

    print(f'train_score = {train_score} : test_score = {test_score}')
    
    return model, train_score, test_score


def explore_one_hot(train):
    '''Establish which cols to 1-hot encode and which ones to treat as floats
    Very simple logic for now
    '''

    unique = train.apply(lambda x: len(x.unique()), axis=0) 

    unique.sort_values(inplace=True, ascending=False) 

    categorical_cols, float_cols = [], []

    for i in unique.index:
        if i=='HasDetections' or i=='MachineIdentifier':
            continue

        if train[i].dtype=='object':
            categorical_cols.append(i)
        else:
            float_cols.append(i)

    return unique, categorical_cols, float_cols

#--------------------
