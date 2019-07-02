#!/usr/bin/env python

import argparse
import sys
#import pandas as pd
#import matplotlib
import numpy as np
import scipy as sp
#import IPython
import sklearn

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPClassifier

from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from generator import load_matrices
from timeit import default_timer as timer

from hilbert import d2xyz



def test_generator(canonical_pattern_order=True,
                   classify_root=True,
                   classify_scale=False,
                   operation_count=2,
                   process_count=2**13,
                   random_root=True,
                   random_scale=True,
                   sample_count=2**10,
                   scale_bit_min=4,
                   scale_bit_max=14,
                   stacked_curve_label=True):
    """Generate and return test data."""
    start = timer()
    test_data = load_matrices(
        canonical_pattern_order=True,
        classify_root=classify_root,
        classify_scale=classify_scale,
        operation_count=operation_count,
        process_count=process_count,
        random_root=random_root,
        random_scale=random_scale,
        sample_count=sample_count,
        scale_bit_min=scale_bit_min,
        scale_bit_max=scale_bit_max,
        stacked_curve_label=stacked_curve_label)
    print("Generated {} samples in {:.2f} seconds.".format(
        sample_count,
        timer() - start))
    return test_data

def test_runner(alpha=1,
                sample_count=2**10,
                model=None,
                neighbor_count=1,
                test_data=None):
    test_result = {}
    test_result['alpha'] = alpha
    test_result['sample_count'] = sample_count
    test_result['neighbor_count'] = neighbor_count

    current = {}
    current['data'] = test_data['data'][:sample_count, :]
    current['target'] = test_data['target'][:sample_count]
    
    X_train, X_test, y_train, y_test = train_test_split(
        current['data'], current['target'], random_state=0)

    max_iter = (1/alpha)**5

    start = timer()
    model.fit(X_train, y_train)
    test_result['fitting_time'] = timer() - start
    
    start = timer()
    test_result['training_set_score'] = model.score(X_train, y_train)
    test_result['training_set_scoring_time'] = timer() - start

    start = timer()
    test_result['test_set_score'] = model.score(X_test, y_test)
    test_result['test_set_scoring_time'] = timer() - start
    return test_result


# Note: 16K ≈ 2**14.

def run_tests(alphas=None,
              models=None,
              process_counts=None,
              sample_counts=None,
              test_suite=None):
    """Run tests and return aggregate results."""
    if alphas is None:
        alphas = [10**(-3), 10**(-2), 10**(-1), 1, 10, 100]
    if sample_counts is None:
        sample_counts = [2**10, 2**11, 2**12, 2**13, 2**14]
    if models is None:
        C = 1
        alpha = 1
        max_depth = 4
        max_features = "auto"
        max_iter = 2**30
        max_leaf_nodes = None
        n_estimators = 100
        n_neighbors = 1
        tol = 1e-3
        

        if test_suite is None:
            models = []
        elif test_suite is TestSuite.NEAREST_NEIGHBORS_PLUS_LINEAR_MODELS:
            models = [
                # Nearest neighbors
                KNeighborsClassifier(n_neighbors),
                KNeighborsRegressor(n_neighbors),

                # Linear models
                LinearRegression(),
                Ridge(alpha=alpha, solver='sag'),
                Lasso(alpha=alpha, max_iter=max_iter),
                #LogisticRegression(C=C, solver='sag'),
#ValueError: Logistic Regression supports only solvers in ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'], got lgbfs.
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    multi_class='auto',
                    solver='lbfgs'),
                LinearSVC(C=C, max_iter=max_iter),
                SGDClassifier(max_iter=max_iter, tol=tol),
                SGDRegressor(max_iter=max_iter, tol=tol)]

        elif test_suite is TestSuite.NAIVE_BAYES_PLUS_OTHERS:
            models = [
                # Naive Bayes
                GaussianNB(),
                BernoulliNB(alpha),
                MultinomialNB(alpha),

                # Decision trees
                DecisionTreeRegressor(),
                DecisionTreeClassifier(max_depth=4, random_state=0),

                # Random forests
                RandomForestClassifier(
                    max_depth,
                    max_features,
                    n_estimators,
                    random_state=2),

                # Gradient boosted decision trees
                GradientBoostingClassifier(
                    learning_rate=0.01,
                    max_depth=1,
                    max_leaf_nodes=max_leaf_nodes,
                    n_estimators=n_estimators,
                    random_state=0),

                # Support vector machines
                SVC(kernel='rbf', C=10, gamma=0.1),

                # Neural networks
                MLPClassifier(
                      activation='tanh',
                      alpha=1,
                      hidden_layer_size=[10, 10],
                      max_iter=1000,
                      random_state=0,
                      solver='lbfgs')]

    if process_counts is None:
        process_counts = [2**10, 2**11, 2**12, 2**13, 2**14]

    max_sample_count = sample_counts[-1]

    test_results = []
    tabular_test_results = []

    for process_count in process_counts:
        process_count_message = "Processes: {}".format(process_count)
        tabular_test_results.append([process_count_message])
        print(process_count_message)

        test_data = test_generator(
            sample_count=max_sample_count,
            process_count=process_count)

        header = ["sample_count"]
        header.extend([' '.join(str(model).split()) for model in models])
        tabular_test_results.append(header)
        print(";".join(header))

        score_rows = []
        timer_rows = []

        score_row_cells = []
        timer_row_cells = []

        for sample_count in sample_counts:
            score_row_cells.append(sample_count)
            timer_row_cells.append(sample_count)

            print("{};".format(sample_count), end='')
            sys.stdout.flush()

            for model in models:
                test_result = test_runner(
                    sample_count=sample_count,
                    model=model,
                    test_data=test_data)
                test_results.append(test_result)

                score = test_result['test_set_score']
                score_row_cells.append(score)

                print("{:.2f};".format(score), end='')
                sys.stdout.flush()

                timer_row_cells.append(
                   "{:.2f}".format(test_result['fitting_time']))
            print()
        score_rows.append(score_row_cells)
        timer_rows.append(timer_row_cells)

    print('\n'.join([";".join([str(x) for x in row_cells])
                   for row_cells in timer_rows]))
    return test_results


from enum import IntEnum

class TestSuite(IntEnum):
    NEAREST_NEIGHBORS = 1
    LINEAR_MODELS = 2
    NAIVE_BAYES = 3
    OTHERS = 4
    NEAREST_NEIGHBORS_PLUS_LINEAR_MODELS = 5
    NAIVE_BAYES_PLUS_OTHERS = 6

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_suite", help="Specify test suite 1-4")
    args = parser.parse_args()

    if args.test_suite is "1":
        test_suite = TestSuite.NEAREST_NEIGHBORS_PLUS_LINEAR_MODELS
    else:
        test_suite = TestSuite.NAIVE_BAYES_PLUS_OTHERS

    #process_counts = [2**10]
    process_counts = [2**4]
    process_counts = [2**2]

    #sample_counts = [2**9]
    scalars = [x**2 for x in range(2, 20, 2)]
    sample_counts = [c*x for x in process_counts for c in scalars]

    sample_counts = [2**10]
    run_tests(process_counts=process_counts,
              sample_counts=sample_counts,
              test_suite=test_suite)

