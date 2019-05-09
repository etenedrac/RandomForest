
#########################
#        IMPORTS        #
#########################
import pandas as pd

from rndforest.DecisionTree import DecisionTree


#########################
#         TESTS         #
#########################

def rndforest_test(test_df, forest, class_atr):
    """Evaluate a given test dataframe in a trained forest.
    """
    df2 = test_df.copy()
    total_length = len(test_df)
    accuracy = 0

    for index, row in df2.iterrows():
        if predict_random_forest(row, forest)==row[class_atr]:
            accuracy += 1
    acc = (accuracy/total_length)
    return acc

def predict_random_forest(df_row,forest):
    """Predicts the value of a given dataframe row with a trained random forest.
    Inputs:
        - df_row {dataframe-row}: dataframe row to predict.
        - forest {list(DecisionTree)}: trained random forest classificator.
    Output:
        The class of the prediction.
    """
    predictions = [tree.predict_instance(df_row) for tree in forest]
    return max(set(predictions), key = predictions.count)