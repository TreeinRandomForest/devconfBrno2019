import pandas as pd
import numpy as np
import sys, gc
import matplotlib.pylab as plt
import operator
import multiprocessing as mp
import config

from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.width', 2000)

def read_data():
    '''Read pre-saved train and test datasets
    '''
    train = pd.read_csv('data/cv_train.csv')
    test = pd.read_csv('data/cv_test.csv')

    return train, test

def check_train_test_1hot(train, test, colname):
    '''Check train and test both have the same set of unique values for column colname
    '''
    train_vals = set(train[colname].unique())
    test_vals = set(test[colname].unique())
    
    intersection = train_vals.intersection(test_vals)
    union = train_vals.intersection(test_vals)
    
    if len(union) != len(intersection):
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
    '''Do binary search
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

        shift = int(shift/2)
        counter += 1

    return None

def create_one_hot(train, test, colnames, thresholds):
    assert(len(colnames)==len(thresholds))
    
    train_df = train[colnames].astype(str).fillna('null').copy()
    test_df = test[colnames].astype(str).fillna('null').copy()

    train_list, test_list = [], []
    for index, col in enumerate(colnames):
        value_counts = train_df[col].value_counts()
        
        top_values = list(value_counts[0:thresholds[index]].index)
        
        train_df[~train_df[col].isin(top_values)] = f'rest_{col}'
        test_df[~test_df[col].isin(top_values)] = f'rest_{col}'
        
        train_list.append( pd.get_dummies(train_df[[col]], dummy_na=False) )
        test_list.append( pd.get_dummies(test_df[[col]], dummy_na=False) )

    train_df = pd.concat(train_list, axis=1)
    test_df = pd.concat(test_list, axis=1)
    
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