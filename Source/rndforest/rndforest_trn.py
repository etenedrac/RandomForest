
#########################
#        IMPORTS        #
#########################
import pandas as pd
import itertools
import random
import sys
import math

from rndforest.DecisionTree import DecisionTree


#########################
#       RNDFOREST       #
#########################

def train_random_forest(df, class_atr='Class', n_trees=100, n_features=1, classifier='CART', fold=""):
    """Performs a random forest classification with the specified classifier.
    
    This method implements the Random Forest (Beriman, 2001) algorithm.
    Inputs:
        - df {dataframe}: supervised learning dataframe.
        - class_atr {string}: name of the class variable in the dataframe.
        - n_trees {int}: number of random trees to generate.
        - n_features {int}: number of random features to consider at every splitting node.
        - classifier {string}: classifier used to perform the classification. Currently:
            'CART': CART classifier (Breiman et al., 1984).
    Output:
        Dictionary containing each of the regression trees discovered.
    """
    
    #List of final decision trees
    forest = []
    
    for i in range(n_trees):
        printProgressBar(i, n_trees, prefix=fold)
        #Creation of the bagged dataset:
        df_bagged = bagging_bootstrap(df)
        
        #Creation of the random tree using the specified method:
        if classifier=='CART':
            forest.append(DecisionTree(children=random_cart(df_bagged,class_atr,n_features)))
        else:
            sys.exit("Not implemented the {} classifier.".format(classifier))
        
        printProgressBar(i+1, n_trees, prefix=fold)
    
    return forest
        
def bagging_bootstrap(df):
    """Creation of a bootstrap sample of a given dataframe.
    
    Given a dataframe of length N, create a sample of size N by drawing N
    examples from the original data, with replacement.
    """
    df_2 = pd.DataFrame()
    for i in range(len(df)):
        df_2 = df_2.append(df.iloc[random.randint(0,len(df)-1)])
    return df_2

def random_cart(df, class_atr, n_features):
    """Generates a random tree using the CART algorithm (Breiman et al., 1984).
    
    Inputs:
        - df {dataframe}: dataframe to train.
        - class_atr {str}: name of the class variable in the dataframe.
        - n_features {int}: number of random features to consider at every splitting node.
    Output:
        Dictionary containing the classification tree discovered.
    """
    if all_instances_same_class(df,class_atr):
        tree = DecisionTree(class_atr=df[class_atr].values[0])
        return [tree]
    else:
        attributes_reduced = reduced_random_attributes(df, class_atr, n_features)
        rules, df1, df2 = best_splitting_point(df,class_atr, attributes_reduced)
        #In case no rule was found, we consider the most frequent value
        if rules is None:
            tree = DecisionTree(class_atr=df[class_atr].mode().values[0])
            return [tree]
        tree_1 = DecisionTree([rules[0]],children=random_cart(df1, class_atr, n_features))
        tree_2 = DecisionTree([rules[1]],children=random_cart(df2, class_atr, n_features))
        return [tree_1,tree_2]

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
        
def all_instances_same_class(df,class_atr):
    """Retruns true if all instances in df have the same class."""
    if len(set(df[class_atr]))==1:
        return True
    else:
        return False
    
def random_permutation(iterable, r=None):
    """Random selection from itertools.permutations(iterable, r)."""
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return list(random.sample(pool, r))
    
def reduced_random_attributes(df, class_atr, n_features):
    """Returns a dataset with the specified number of random relected attributes."""
    # Get the possible names
    possible_names = [x for x in list(df.columns.values) if x != class_atr]
    
    #Get a random permutation of the attributes:
    random_atributes = random_permutation(possible_names, n_features)

    return random_atributes

def gini(df, class_atr):
    """Computes the gini index of a given dataframe."""
    list_classes = list(set(df[class_atr]))
    len_df = len(df)
    
    gini_score = 1
    for cls in list_classes:
        gini_score -= (len(df[df[class_atr]==cls])/len_df)**2
    return gini_score

def quasi_powerset(iterable):
    "quasi_powerset([1,2,3]) --> (1,) (2,) (3,)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, math.floor(len(s)/2)+1))

def pairs_subsets(iterable):
    "pairs_subsets([1,2,3]) --> [((1),(2,3)),((2),(1,3)),((3),(1,2))]"
    subsets = set(quasi_powerset(iterable))
    pairs = []
    for elem in subsets:
        c = set(iterable) - set(elem)
        pairs.append(frozenset([frozenset(list(c)), frozenset(list(elem))]))
    pairs = set(pairs)
    pairs = list(pairs)
    for i in range(len(pairs)):
        pairs[i] = list(pairs[i])
        for j in range(len(pairs[i])):
            pairs[i][j] = list(pairs[i][j])
    return pairs
    
def best_splitting_point(df,class_atr, attributes_list):
    """Computes the best splitting point for the attribute maximizing 
    the reduction in impurity.
    
    Inputs:
        - df {dataframe}: dataframe from which we will find the splitting.
        - class_atr {str}: name of the class variable in the dataframe.
        - attributes_list {list(attributes)}: list of possible attributes.
    Outputs:
        - (1) {list(3-tuple)}: List of rules generated from the split.
        - (2) {dataframe}: dataframe containing the instances from the first split.
        - (3) {dataframe}: dataframe containing the instances from the second split.
    """
    # Get the attributes
    attr_names = attributes_list
    best_gini = float('inf')
    frmt = None
    
    for attribute in attr_names:
        #If the attribute is categorical
        if df[attribute].dtypes==object:
            values = list(set(df[attribute]))
            pairs = pairs_subsets(values)
            for pair in pairs:
                df_1 = df[df[attribute].isin(pair[0])]
                df_2 = df[df[attribute].isin(pair[1])]
                
                #Compute the gini index:
                piv_gini = (len(df_1)/len(df))*gini(df_1,class_atr) +\
                            (len(df_2)/len(df))*gini(df_2,class_atr)
                if piv_gini < best_gini:
                    best_gini = piv_gini
                    first_split = pair[0]
                    second_split = pair[1]
                    frmt = 'object'
                    final_attr = attribute
        
        #Else, the attribute is numerical
        else:
            #First we have to sort the values of the attribute
            df = df.sort_values(by=[attribute])
            #And get the differences
            differences = list((df[attribute].diff()[1:]/2))
            bias = list(df[attribute][:-1])
            midpoints = [val+bias[i] for i,val in enumerate(differences) if val>0]
            for point in midpoints:
                df_1 = df[df[attribute]<point]
                df_2 = df[df[attribute]>=point]
                
                #Compute the gini index:
                piv_gini = (len(df_1)/len(df))*gini(df_1,class_atr) +\
                            (len(df_2)/len(df))*gini(df_2,class_atr)
                if piv_gini < best_gini:
                    best_gini = piv_gini
                    first_split = point
                    frmt = 'number'
                    final_attr = attribute
                    
    if frmt is None:
        return None, df, df
    
    if frmt == 'object':
        df_1 = df[df[final_attr].isin(first_split)]
        df_2 = df[df[final_attr].isin(second_split)]
        return [(final_attr,"in",first_split),(final_attr,"in",second_split)], df_1, df_2
    else:
        df_1 = df[df[final_attr]<first_split]
        df_2 = df[df[final_attr]>=first_split]
        return [(final_attr,"<",first_split),(final_attr,">=",first_split)], df_1, df_2