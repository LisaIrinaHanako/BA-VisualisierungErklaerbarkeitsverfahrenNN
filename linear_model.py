import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import helper_methods as helper
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from interactive_ba_preparation_master.model import load_model
from interactive_ba_preparation_master.dataset import German_Credit

# Get pretrained PyTorch model and dataset
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")

global_penalty = None
global_dual = None
global_tol = None
global_C = None
global_fit_intercept = None
global_intercept_scaling = None
global_random_state = None
global_solver = None
global_max_iter = None
global_multi_class = None
global_verbose = None
global_n_jobs = None
global_l1_ratio = None

# Function to calculate LogisticRegression
def calc_classifier(x_train, x_test, y_net_train, y_net_test,
                    penalty='l2', dual=False, tol=0.0001, 
                    C=1.0, fit_intercept=True, intercept_scaling=1, 
                    random_state=None, solver='sag', 
                    max_iter=100, multi_class='auto', verbose=0, 
                    n_jobs=None, l1_ratio=None):
    global global_penalty
    global global_dual
    global global_tol
    global global_C
    global global_fit_intercept
    global global_intercept_scaling
    global global_random_state
    global global_solver
    global global_max_iter
    global global_multi_class
    global global_verbose
    global global_n_jobs
    global global_l1_ratio
    global global_classifier

    print(global_penalty != penalty or
        global_dual != dual or 
        global_tol != tol or 
        global_C != C or 
        global_fit_intercept != fit_intercept or 
        global_intercept_scaling != intercept_scaling or 
        global_random_state != random_state or 
        global_solver != solver or 
        global_max_iter != max_iter or 
        global_multi_class != multi_class or 
        global_verbose != verbose or 
        global_n_jobs != n_jobs or 
        global_l1_ratio != l1_ratio or 
        global_classifier == None)

    if(global_penalty != penalty or
        global_dual != dual or 
        global_tol != tol or 
        global_C != C or 
        global_fit_intercept != fit_intercept or 
        global_intercept_scaling != intercept_scaling or 
        global_random_state != random_state or 
        global_solver != solver or 
        global_max_iter != max_iter or 
        global_multi_class != multi_class or 
        global_verbose != verbose or 
        global_n_jobs != n_jobs or 
        global_l1_ratio != l1_ratio or 
        global_classifier == None):

        global_penalty = penalty
        global_dual = dual 
        global_tol = tol 
        global_C = C 
        global_fit_intercept = fit_intercept 
        global_intercept_scaling = intercept_scaling 
        global_random_state = random_state 
        global_solver = solver 
        global_max_iter = max_iter 
        global_multi_class = multi_class 
        global_verbose = verbose 
        global_n_jobs = n_jobs
        global_l1_ratio = l1_ratio 

        global_classifier = LogisticRegression(penalty = penalty, dual = dual, tol = tol, 
                                                C = C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, 
                                                random_state=random_state, solver = solver, 
                                                max_iter = max_iter, multi_class=multi_class, verbose=verbose,
                                                n_jobs=n_jobs,l1_ratio=l1_ratio) 

    # Performing classification 

    global_classifier.fit(x_train, y_net_train) 
    return global_classifier 

# Funktion um Koeffizienten und Onehot-Spaltennamen (mit einzelnen num. Feature Werten) zu holen
def get_columns_and_coeff(penalty='l2', dual=False, tol=0.0001, 
                        C=1.0, fit_intercept=True, intercept_scaling=1, 
                        random_state=None, solver='sag', 
                        max_iter=100, multi_class='auto', verbose=0, 
                        warm_start=False, n_jobs=None, l1_ratio=None):
    classifier, predictions = get_classifier_and_predictions(penalty=penalty, dual=dual, tol=tol, 
                                                            C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, 
                                                            random_state=random_state, solver=solver, 
                                                            max_iter=max_iter, multi_class=multi_class, verbose=verbose, 
                                                            n_jobs=n_jobs, l1_ratio=l1_ratio)
    return ds.cols_onehot, classifier.coef_, predictions

# Funktion um Classifier und Predictions zu bestimmen
def get_classifier_and_predictions(penalty='l2', dual=False, tol=0.0001, 
                                    C=1.0, fit_intercept=True, intercept_scaling=1, 
                                    random_state=None, solver='sag', 
                                    max_iter=100, multi_class='auto', verbose=0, 
                                    warm_start=False, n_jobs=None, l1_ratio=None):
       
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    # calculate actual classifier result
    classifier = calc_classifier(x_train=x_train, x_test= x_test, y_net_train=y_net_train, y_net_test=y_net_test,
                                penalty=penalty, dual=dual, tol=tol, 
                                C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, 
                                random_state=random_state, solver=solver, 
                                max_iter=max_iter, multi_class=multi_class, verbose=verbose, 
                                n_jobs=n_jobs, l1_ratio=l1_ratio)
    predictions = classifier.predict(x_test)
    return classifier, predictions

# Funktion um Genauigkeit zu berechnen
def lin_mod_accuracy(predictions, y_net_test):
    
    acc = accuracy_score(predictions, y_net_test)
    return acc

def main():

    # get trianing and test tensors and net trained labels
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    # get predictions 
    classifier, predictions = get_classifier_and_predictions()
    
    # calculate accuracy
    avg = 0    
    acc = accuracy_score(predictions, y_net_test)
    avg = avg + acc
    
    # print accuracy average for specified parameters
    print("Accuracy of {} for Logistic Regression".format
                            (avg * 100))# / 20 * 100))
    print("\n", "-"*80, "\n")

    # print(get_columns_and_coeff())

# Calling main function 
if __name__=="__main__": 
    main() 