import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import helper_methods as helper
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from interactive_ba_preparation_master.model import load_model
from interactive_ba_preparation_master.dataset import German_Credit

# Get pretrained PyTorch model and dataset
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")

global_selected_datapoint = None
global_criterion = None
global_splitter = None
global_max_features = None
global_dt_depth = None
global_min_samples_split = None
global_min_smp_lf = None
global_max_leaf_nodes = None
global_ccp_alpha = None
global_classifier = None

# Function to calculate DecisionTreeClassifier
def calc_classifier(x_train, x_test, y_train, y_test, criterion='gini', 
                    splitter='best', max_depth=8, min_samples_split=1, min_smp_lf=0,
                    max_features=None, max_leaf_nodes=None, min_impurity_decrease=0,
                    min_impurity_split=0,ccp_alpha=0):
    global global_criterion
    global global_splitter
    global global_max_features
    global global_dt_depth
    global global_min_samples_split
    global global_min_smp_lf
    global global_max_leaf_nodes
    global global_ccp_alpha
    global global_classifier


    print(global_criterion != criterion or
        global_splitter != splitter or
        global_max_features != max_features or
        global_dt_depth != max_depth or
        global_min_samples_split != min_samples_split or
        global_min_smp_lf != min_smp_lf or
        global_max_leaf_nodes != max_leaf_nodes or
        global_ccp_alpha != ccp_alpha or
        global_classifier == None)
        
    if(global_criterion != criterion or
        global_splitter != splitter or
        global_max_features != max_features or
        global_dt_depth != max_depth or
        global_min_samples_split != min_samples_split or
        global_min_smp_lf != min_smp_lf or
        global_max_leaf_nodes != max_leaf_nodes or
        global_ccp_alpha != ccp_alpha or
        global_classifier == None):

        global_criterion = criterion
        global_splitter = splitter
        global_max_features = max_features
        global_dt_depth = max_depth
        global_min_samples_split = min_samples_split
        global_min_smp_lf = min_smp_lf
        global_max_leaf_nodes = max_leaf_nodes
        global_ccp_alpha = ccp_alpha

        global_classifier = DecisionTreeClassifier(criterion = criterion, splitter=splitter, 
                                            max_depth = max_depth, min_samples_split=min_samples_split,
                                            min_samples_leaf=min_smp_lf, max_features=max_features,
                                            max_leaf_nodes=max_leaf_nodes,
                                            min_impurity_decrease=min_impurity_decrease,
                                            min_impurity_split=min_impurity_split, ccp_alpha=ccp_alpha)
        # Performing classification 
        global_classifier.fit(x_train, y_train) 
    return global_classifier 

def get_classifier(idx, criterion='gini', splitter='best', max_depth=8,
                            min_samples_split=1, min_smp_lf=0,
                            max_features=None,
                            max_leaf_nodes=None, min_impurity_decrease=0,
                            min_impurity_split=0,ccp_alpha=0):

    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    classifier = calc_classifier(x_train, x_test, y_net_train, y_test,
                                criterion = criterion, splitter=splitter, 
                                max_depth = max_depth, min_samples_split=min_samples_split,
                                min_smp_lf=min_smp_lf, max_features=max_features,
                                max_leaf_nodes=max_leaf_nodes,
                                min_impurity_decrease=min_impurity_decrease,
                                min_impurity_split=min_impurity_split, ccp_alpha=ccp_alpha)
    return classifier

# Function to plot whole DecisionTree
def plot_result_tree(classification_results, feature_names_count):
    # plot the resulting Tree 
    # TODO: preprocessing rückgängig machen, um verständliche Ergebnisse zu zeigen
    # --> ändere nur numerische feature
    # print(helper.inverse_preprocessing(x_test[:, x for x in ))
    tree.plot_tree(decision_tree = classification_results,
                    feature_names = helper.get_feature_labels(ds, feature_names_count),
                    impurity = False,
                    class_names = list("".join(map(str, set(ds.label)))),
                    fontsize=4,
                    node_ids = True)
    plt.show()

def get_explanation_branch(classifier, predictions, datapoint, sample_id):
    # get children, feature and threshold of old tree
    feature = classifier.tree_.feature
    threshold = classifier.tree_.threshold

    # explanation_branch = classifier.tree_
    node_indicator = classifier.decision_path(datapoint)
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

    for node_id in node_index:
        # check if value of the split feature for sample 0 is below threshold
        if (datapoint[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"
        feature_name = get_feature_name(datapoint, feature, node_id)
        feature_value = datapoint[sample_id, feature[node_id]]

def get_feature_name(datapoint, feature_list, index):
    feature_names = helper.get_feature_labels(ds, len(datapoint[0,:]))
    return feature_names[feature_list[index]]


# Funktion um Genauigkeit zu berechnen
def dt_accuracy(predictions, y_net_test):
    
    acc = accuracy_score(predictions, y_net_test)
    return acc

# Main function
def main(): 

    # set parameters for classifier calculation
    avg = 0
    # rand_state = 100
    max_d = 15
    min_smp_lf = 20
    splitter = "best"
    max_f = None
    idx = 0
    criterion = 'gini'
    

    # get trianing and test tensors and net trained labels
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    # calculate the DecisionTreeClassifier
    classifier = get_classifier(idx, criterion)
    # calculate actual result
    predictions = classifier.predict(x_test)
    
    # calculate accuracy
    acc = accuracy_score(predictions, y_test)
    
    # print accuracy average for specified parameters
    print("Accuracy of {} for Tree ".format(acc))

    get_explanation_branch(classifier, predictions, x_test, sample_id=0)
    plot_result_tree(classifier, len(x_test[0,:]))


# Calling main function 
if __name__=="__main__": 
    main() 