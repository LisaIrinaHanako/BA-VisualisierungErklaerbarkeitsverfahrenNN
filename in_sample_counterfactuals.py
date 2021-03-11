import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import helper_methods as helper
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import accuracy_score
from interactive_ba_preparation_master.model import load_model
from interactive_ba_preparation_master.dataset import German_Credit

# Get pretrained PyTorch model and dataset
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")

# Function to calculate LogisticRegression
def calc_classifier(x_train, x_test, y_train, y_test,
                    n_neighbors, weights, algorithm,
                    leaf_size, p, metric, metric_params):
    
    classifier = knn(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                    leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params) 
    classifier.fit(x_train, np.array(y_train)) 
    return classifier 

# Funktion um Koeffizienten und Onehot-Spaltennamen (mit einzelnen num. Feature Werten) zu holen
def get_columns_and_coeff():
    classifier, predictions = get_classifier_and_predictions()
    return ds.cols_onehot, classifier.coef_

# Funktion um Classifier und Predictions zu bestimmen
def get_classifier_and_predictions():
    n_neighbors=5 
    weights='uniform'
    algorithm='auto'
    leaf_size=30
    p=2
    metric='minkowski' # p=2 + minkowski = euclidean
    metric_params=None
    sample_id=0

    
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    # calculate actual classifier result
    classifier = calc_classifier(x_train, x_test, y_train, y_test,
                    n_neighbors, weights, algorithm,
                    leaf_size, p, metric, metric_params)

    neigh_dist, neigh_ind = classifier.kneighbors(x_test[sample_id].reshape(1,-1), n_neighbors, return_distance=True)

    return classifier, neigh_dist, neigh_ind

def get_cf_min_dist(neigh_dist, neigh_ind, x_test, y_test, x_train, y_train, sample_id=0):
    neighbors = neigh_ind[sample_id]
    neighbor_dist = neigh_dist[sample_id]
    ref_class = y_test[sample_id]
    min_dist = max(neighbor_dist)
    actual_cf = None

    for i, neigh in enumerate(neighbors):
        if(y_train[neigh] != ref_class):
            test_dist = neighbor_dist[i]
            if(test_dist < min_dist):
                min_dist = test_dist
                actual_cf = x_train[neigh]

    return actual_cf, min_dist


def get_cfs_df(cf, x_test, y_test, sample_id = 0):
    inv_num, inv_cat = helper.inverse_preprocessing(ds, x_test, sample_id)
    test_dp = inv_cat.tolist() + inv_num.tolist()
    # test_dp.append(y_test[sample_id])
    cf_inv_num, cf_inv_cat = helper.inverse_preprocessing_single(ds, cf)
    newdf = cf_inv_cat.tolist() + cf_inv_num.tolist()
    for j, val in enumerate(newdf):
        if test_dp[j] == newdf[j]:
            curVal= newdf[j]
            newdf[j] = '-'
    return newdf

def main():

    # get trianing and test tensors and net trained labels
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    # get predictions 
    classifier, neigh_dist, neigh_ind = get_classifier_and_predictions()
    
    print(neigh_dist)
    print(neigh_ind)

    actual_cf, min_dist = get_cf_min_dist(neigh_dist, neigh_ind, x_test, y_test, x_train, y_train)
    print(actual_cf)
    print(min_dist)

    print(get_cfs_df(actual_cf,x_test,y_test))


# Calling main function 
if __name__=="__main__": 
    main() 