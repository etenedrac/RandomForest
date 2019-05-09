Random Forest algorithm, by Albert Cardenete
--------------------------------------------

The structure of this project is as follows:

	PW2-SEL-1819-AlbertCardenete/			->The root of the project
	├── Data/
	│   ├── contraceptive.csv			->Dataset 1
	│   ├── credit.csv				->Dataset not used
	│   ├── nursery.csv				->Dataset 2
	│   └── voting.csv				->Dataset 3
	├── Documentation/
	│   ├── results_contraceptive.txt		->Final output for Dataset 1
	│   ├── results_nursery.txt			->Final output for Dataset 2
	│   ├── results_voting.txt			->Final output for Dataset 3
	│   └── REPORT_ALBERT_CARDENETE_PW2.pdf		->REPORT
	└── Source/
    		├── rndforest/
		│   ├── __init__.py			->Dummy file to informe that this is a package
		│   ├──	DecisionTree.py			->Contains the Class DecisionTree, a data structure
		│   ├── evaluation.py			->Contains functions of the evaluations and k-folds (a frontend)
		│   ├── rndforest_trn.py		->Contains the Random Forest algorithm
		│   └── rules_tst.py			->Contains the functions to test and predict
    		└── __main__.py				->Main function of the program to evaulate a given dataset