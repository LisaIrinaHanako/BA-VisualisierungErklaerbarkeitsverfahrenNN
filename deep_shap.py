import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import helper_methods as helper
import shap
from interactive_ba_preparation_master.model import load_model
from interactive_ba_preparation_master.dataset import German_Credit
from shap import Explanation

# Get pretrained PyTorch model and dataset
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")

# Function to get a Counterfactual Explainer
def get_shap_deep_explainer(x_train):
    explainer = shap.DeepExplainer(clf, x_train)
    exp = shap.explainers._deep.PyTorchDeep(clf, x_train)
    return exp

# Function to get the Counterfactuals
def get_shap_explanation(x_test, explainer, no_CFs = 4, desired_class = 1):
    shap_values = explainer.shap_values(x_test)
    
    return shap_values

def main():

    # get trianing and test tensors and net trained labels
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    # get predictions 
    classifier = get_shap_deep_explainer(x_train)
    shap_explanation = get_shap_explanation(x_test, classifier)
    # shap.plots.bar(shap_explanation[0], show=True)

    # print(classifier())
    print(shap_explanation)
    
# Calling main function 
if __name__=="__main__": 
    main() 