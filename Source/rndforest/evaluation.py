
#########################
#        IMPORTS        #
#########################
import pandas as pd
import time
from itertools import combinations, product
from sklearn.model_selection import KFold
import numpy as np

from rndforest.DecisionTree import DecisionTree
from rndforest.rndforest_trn import train_random_forest
from rndforest.rndforest_tst import rndforest_test


#########################
#        METRICS        #
#########################

def importance_forest(forest, verbose = True):
    """Computes the importance of each attribute in a forest.
    Input:
        - forest {list(DecisionTree)}: forest to be studied.
    Output:
        Dictionary containing the attributes (key) and the frequency
        (value) of that attribute in the random forest.
    """
    importance_dic = {}
    total = 0
    for tree in forest:
        lst_tree = tree.get_importance()
        for item in lst_tree:
            total += 1
            if (item in importance_dic): 
                importance_dic[item] += 1
            else: 
                importance_dic[item] = 1
    
    for key, value in importance_dic.items():
        importance_dic[key] = value/total
        
    if verbose:
        importance = sorted(importance_dic.items(), key=lambda x:x[1], reverse=True)
        
        for key,value in importance: 
            print ("{} : {}".format(key,value))
        
    return importance_dic

def metric_importance_forest(list_dic):
    """Function to extract the mean and standard deviation of the importance.
    Inputs:
        - list_dic {list(dictionary)}: list of dictionaries containing
            the importancy for each random forest run.
    Output:
        Print of the mean and std of the importancy.
    """
    final_dic = {}
    for dic in list_dic:
        for key, val in dic.items():
            if (key in final_dic):
                final_dic[key].append(val)
            else:
                final_dic[key] = [val]

    # If we have not seen any time a value, we add a zero importancy
    max_len = 0
    for key, val in final_dic.items():
        if len(val)>max_len:
            max_len = len(val)
    
    for key, val in final_dic.items():
        if len(val)<max_len:
            number_zeros = max_len - len(val)
            final_dic[key] = final_dic[key].extend([0]*number_zeros)
    
    # Finally we build the dictionary with the means and accuracies
    metric_dictionary = {}
    for key, val in final_dic.items():
        metric_dictionary[key] = [np.mean(val), np.std(val)]
    
    importance = sorted(metric_dictionary.items(), key=lambda x:x[1][0], reverse=True)

    print("Final importance metrics:")
    for key,value in importance: 
        print ("{} : {} ".format(key,value[0]) + u"\u00B1" + " {}".format(value[1]))


#########################
#      EVALUATIONS      #
#########################

def rndforest_evaluate(df_trn, cls_atr, df_tst=None, n_trees=100, n_features=1):
    """Evaluate the random forest with a given train and test dataset."""
    
    #Train
    forest = train_random_forest(df_trn, class_atr=cls_atr, n_trees=n_trees, n_features=n_features)
    
    #Test
    if df_tst is not None:
        print("Achieved accuracy: ",rndforest_test(df_tst, forest, cls_atr))

    #Print importancies
    importance_forest(forest)
    
    return forest

def rndforest_k_fold_evaluate(df, cls_atr, n_trees=100, n_features=1, k=3):
    """
    k-fold cross validation of the random forest with a given dataset.
    """
    
    kf = KFold(n_splits = k, shuffle = True, random_state = 2)
    
    accuracy = []
    elapsed = []
    importancies = []
    f = 1
    #Getting the different train and test datasets
    for trn,tst in kf.split(df):
        train, test = df.iloc[trn], df.iloc[tst]
        
        #Train
        t = time.time()
        forest = train_random_forest(train, class_atr=cls_atr, n_trees=n_trees, n_features=n_features, fold="Fold {}/{}".format(f,k))
        elapsed.append(time.time() - t)
        
        #Test
        accuracy.append(rndforest_test(test, forest, cls_atr))

        #Importancies
        importancies.append(importance_forest(forest, verbose=False))
        f += 1
    mean_accuracy = np.mean(accuracy)
    std_accuracy = np.std(accuracy)
    mean_time = np.mean(elapsed)
    std_time = np.std(elapsed)
    print("Final accuracy: {mn_acc}".format(mn_acc=mean_accuracy)+u"\u00B1"
        +"{std_acc}".format(std_acc=std_accuracy))
    print("Final time: {mn_tm}".format(mn_tm=mean_time)+u"\u00B1"
        +"{std_tm}".format(std_tm=std_time))
    metric_importance_forest(importancies)