
#########################
#        IMPORTS        #
#########################
import pandas as pd
import argparse
import time

from rndforest.evaluation import rndforest_evaluate, rndforest_k_fold_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help="Dataset used to study located in the 'Data' folder (ended with .csv)",type=str, default="test.csv")
parser.add_argument('--classname', help="Name of the class of the dataset",type=str, default="Class")
parser.add_argument('--validate', help="Validate using k-fold CV (Default)", dest='val', action='store_true')
parser.add_argument('--no-validate', help="Do not do the validation", dest='val', action='store_false')
parser.add_argument('--nt', help="Number of Trees to generate (integer / Default: 100)", type=int, default=100)
parser.add_argument('--f', help="Number of festures to consider in each node (integer / Default: 1)", type=int, default=1)
parser.add_argument('--k', help="Number of folds for validation (integer / Default: 3)", type=int, default=3)
parser.set_defaults(val=True)
args, unknown = parser.parse_known_args()

path = 'Data/' + args.dataset

#Read the dataset
df = pd.read_csv(path)

# print('Computing the importancies for all the dataset...')
# print('-----------------------------------------------------------------------------')

# rndforest_evaluate(df, args.classname)

print('-----------------------------------------------------------------------------')
if args.val:
    elapsed = []
    print('Computing the CV...')
    rndforest_k_fold_evaluate(df, args.classname, n_trees=args.nt, n_features=args.f, k=args.k)
    print('-----------------------------------------------------------------------------')