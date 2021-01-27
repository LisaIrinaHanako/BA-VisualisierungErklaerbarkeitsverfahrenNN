import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import helper_methods as helper
from sklearn import tree
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from interactive_ba_preparation_master.model import load_model
from interactive_ba_preparation_master.dataset import German_Credit

# Get pretrained PyTorch model and dataset
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")

# Function to calculate DecisionTreeClassifier
def calc_classifier(x_train, x_test, y_train, y_test,
                    alpha, fit_intercept, normalize, 
                    precompute, copy_X, max_iter, 
                    tol, warm_start, positive, 
                    random_state, selection):
    
    classifier = Lasso(alpha = alpha, fit_intercept = fit_intercept, normalize = normalize, 
                    precompute = precompute, copy_X=copy_X, max_iter=max_iter, 
                    tol=tol, warm_start=warm_start, positive = positive, 
                    random_state = random_state, selection=selection) 
    # Performing classification 
    classifier.fit(x_train, y_train) 
    return classifier 



def main():

    avg = 0
    alpha=1.0,
    fit_intercept=True
    normalize=False
    precompute=False 
    copy_X=True
    max_iter=1000 
    tol=0.0001
    warm_start=False
    positive=False
    random_state=None 
    selection='cyclic'

    
    # get trianing and test tensors and net trained labels
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    # calculate actual classifier result
    classifier = calc_classifier(x_train, x_test, y_train, y_test,
                    alpha, fit_intercept, normalize, 
                    precompute, copy_X, max_iter, 
                    tol, warm_start, positive, 
                    random_state, selection)
    predictions = classifier.predict(x_test)
    
    print(predictions)
    print(y_test)
    # calculate accuracy
    acc = accuracy_score(predictions, y_test)
    avg = avg + acc
    
    # print accuracy average for specified parameters
    print("Accuracy of {} for Lasso".format
                            (avg / 20 * 100))
    print("\n", "-"*80, "\n")

    helper.print_prediction(x_test, clf)


# Calling main function 
if __name__=="__main__": 
    main() 