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

# Function to calculate DecisionTreeClassifier
def calc_classifier(x_train, x_test, y_train, y_test,
                    rand_state, max_d, min_smp_lf, splitter, max_f):
    
    classifier = DecisionTreeClassifier(
            criterion = 'gini', random_state = rand_state, 
            max_depth = max_d, min_samples_leaf = min_smp_lf,
            max_features = max_f) 
    # Performing classification 
    classifier.fit(x_train, y_train) 
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
    # fig = plt.figure(figsize=(25,20))
    plt.show()


def get_explanation_branch(classifier, predictions, datapoint, sample_id):
    # get children, feature and threshold of old tree
    children_left = classifier.tree_.children_left
    children_right = classifier.tree_.children_right
    feature = classifier.tree_.feature
    threshold = classifier.tree_.threshold

    # explanation_branch = classifier.tree_
    node_indicator = classifier.decision_path(datapoint)
    leaf_id = classifier.apply(datapoint)
    
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                    node_indicator.indptr[sample_id + 1]]

    print("node indices (that sample goes through): ", node_index)
    print('Rules used to predict sample {id}:\n'.format(id=sample_id))

    for node_id in node_index:
        # continue to the next node if it is a leaf node
        # if leaf_id[sample_id] == node_id:
        #     continue

        
        # check if value of the split feature for sample 0 is below threshold
        if (datapoint[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"
        feature_name = get_feature_name(datapoint, feature, node_id)
        feature_value = datapoint[sample_id, feature[node_id]]
        print("decision node {node} : (datapoint[{sample}, (original: still open) scaled: {feature}] = {value}) "
              "{inequality} {threshold})".format(
                  node=node_id,
                  sample=sample_id,
                  feature=feature_name,
                  value=feature_value,
                  inequality=threshold_sign,
                  threshold=threshold[node_id]))
    
    print("\n class: {calculated_class}".format(calculated_class = predictions[sample_id]))
    print("-"*80)
        # # check if value of the split feature for sample 0 is below threshold
        # if (datapoint[sample_id, feature[node_id]] <= threshold[node_id]):
        #     explanation_branch.children_left[node_id] = classifier.tree_.children_left[node_id]
        #     explanation_branch.children_right[node_id] = None
        #     print(node_id, " left child: ", explanation_branch.children_left[node_id] )
        # else:
        #     explanation_branch.children_right[node_id] = classifier.tree_.children_right[node_id]
        #     explanation_branch.children_left[node_id] = None
        #     print(node_id, " right child: ", explanation_branch.children_right[node_id] )

        
def get_feature_name(datapoint, feature_list, index):
    feature_names = helper.get_feature_labels(ds, len(datapoint[0,:]))
    return feature_names[feature_list[index]]



# Main function
def main(): 

    # set parameters for classifier calculation
    avg = 0
    rand_state = 100
    max_d = 15
    min_smp_lf = 20
    splitter = "best"
    max_f = None
    
    # calculate the DecisionTreeClassifier
    # 20 times to get a hang on average accuracy --> TODO später dann rausnehmen 
    for i in range(20):

        # get trianing and test tensors and net trained labels
        x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)

        # calculate actual classifier result
        classification_results = calc_classifier(x_train, x_test, y_net_train, y_net_test,
                                                rand_state, max_d, min_smp_lf, splitter, max_f)
        predictions = classification_results.predict(x_test)
        # calculate accuracy
        acc = accuracy_score(predictions, y_test)
        avg = avg + acc
    
    # print accuracy average for specified parameters
    print("Accuracy of {} for Tree with \n {} random_state, \n {} depth, \n {} min_sample_leaf".format
                            (avg / 20 * 100,
                            rand_state,
                            max_d,
                            min_smp_lf))
    print("\n", "-"*80, "\n")

    get_explanation_branch(classification_results, predictions, x_test, sample_id=50)

    plot_result_tree(classification_results, len(x_test[0,:]))


# Calling main function 
if __name__=="__main__": 
    main() 