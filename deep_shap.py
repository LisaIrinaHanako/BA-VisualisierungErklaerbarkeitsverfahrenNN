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
def get_shap_explanation(x_test, explainer, ranked_outputs = None, output_rank_order = "max", check_additivity = False):
    shap_values = explainer.shap_values(X = x_test,ranked_outputs = ranked_outputs, output_rank_order=output_rank_order)
    
    return shap_values

# Function to sum up all one-hot encoded column values
def get_barplot_values(vals, sample_id):
    col_vals_summed_0 = dict(zip(ds.column_names[:len(ds.column_names)], [0]*(len(ds.column_names))))
    for count, i in enumerate(vals[0][sample_id]):
        col_onehot = ds.cols_onehot[count]
        col_name = col_onehot.split(':')[0]
        col_vals_summed_0[col_name] += i

    col_vals_summed_1 = dict(zip(ds.column_names[:len(ds.column_names)], [0]*(len(ds.column_names))))
    for count, i in enumerate(vals[1][sample_id]):
        col_onehot = ds.cols_onehot[count]
        col_name = col_onehot.split(':')[0]
        col_vals_summed_1[col_name] += i

    return col_vals_summed_0, col_vals_summed_1

def main():

    # get trianing and test tensors and net trained labels
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    # get predictions 
    classifier = get_shap_deep_explainer(x_train)
    shap_explanation = get_shap_explanation(x_test, classifier)
    # shap.plots.bar(shap_explanation[0], show=True)

    # print(classifier())
    # print(shap_explanation)
    
# Calling main function 
if __name__=="__main__": 
    main() 