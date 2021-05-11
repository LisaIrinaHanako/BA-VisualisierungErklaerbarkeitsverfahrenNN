import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import helper_methods as helper
from sklearn.neighbors import KNeighborsClassifier as knn
import gower

from sklearn.metrics import accuracy_score
from interactive_ba_preparation_master.model import load_model
from interactive_ba_preparation_master.dataset import German_Credit

# Get pretrained PyTorch model and dataset
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")

global_gower_matrix = None

# Function to calculate LogisticRegression
def calc_knn_classifier(x_train, x_test, y_train, y_test,
                    n_neighbors, weights, algorithm,
                    leaf_size, p, metric, metric_params):
    
    classifier = knn(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                    leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params) 
    classifier.fit(x_test, np.array(y_test).reshape(len(y_test),)) 
    return classifier 

# Funktion um Gower Distanz zu berechnen
def calc_gower_distance(x_test, sample_id=0):
    global global_gower_matrix
    if global_gower_matrix is None:
        df = pd.DataFrame(x_test)
        global_gower_matrix = gower.gower_matrix(df)
    
    sample_row = global_gower_matrix[sample_id]

    min_dist = global_gower_matrix[sample_row == min(sample_row[sample_row != min(sample_row)])]
    # print("gower shape für samlpe: ", sample_row.shape)
    # print("datenpunkt mit min gower Distanz für sample_id", min_dist)

    min_ind = [sample_row == min(sample_row[sample_row != min(sample_row)])]
    # print(min_ind)
    return min_dist, min_ind


# Funktion um Koeffizienten und Onehot-Spaltennamen (mit einzelnen num. Feature Werten) zu holen
def get_columns_and_coeff():
    classifier, predictions = get_classifier_and_predictions()
    return ds.cols_onehot, classifier.coef_

def call_gower_single(X,Y):
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    global global_gower_matrix
    if global_gower_matrix is None:
        df = pd.DataFrame(x_test)
        global_gower_matrix = gower.gower_matrix(df)

    sample_id_x = helper.get_id_for_dp(x_test, torch.tensor(X))
    sample_id_y = helper.get_id_for_dp(x_test, torch.tensor(Y))
    # sample_id_x=0
    # sample_id_y=0
    sample_row = global_gower_matrix[sample_id_x][sample_id_y]

    return sample_row

# Funktion um Classifier und Predictions zu bestimmen
def get_classifier_and_predictions(n_neighbors=5, weights='uniform',
                                   algorithm='auto', leaf_size=30, p=2,
                                   metric='euclidean', metric_params=None,
                                   sample_id=0, distance_metric = "euclidean"):
    global global_gower_matrix
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)

    if global_gower_matrix is None:
        df = pd.DataFrame(x_test)
        global_gower_matrix = gower.gower_matrix(df)

    if distance_metric == "gower": #gower
        df = pd.DataFrame(x_test)
        df = {'test': df}
        classifier = calc_knn_classifier(x_train, x_test, y_train, y_test,
                                            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                            leaf_size=leaf_size, p=p, 
                                            metric=call_gower_single, metric_params=None)#{"data_x":df})
    else: 
        # calculate actual classifier result
        classifier = calc_knn_classifier(x_train, x_test, y_train, y_test,
                                            n_neighbors, weights, algorithm,
                                            leaf_size, p, metric, metric_params)

    neigh_dist, neigh_ind = classifier.kneighbors(x_test[sample_id].reshape(1,-1), n_neighbors, return_distance=True)

    return neigh_dist, neigh_ind

def get_cf_min_dist(neigh_dist, neigh_ind, n_neighbors, x_test, y_test, x_train, y_train, sample_id=0):
    neighbors = neigh_ind[sample_id]
    neighbor_dist = neigh_dist[sample_id]
    ref_class = y_test[sample_id]
    max_of_min_dists = max(neighbor_dist)
    count = 0
    actual_cf = []

    for i, neigh in enumerate(neighbors):
        if(y_train[neigh] != ref_class):
            actual_cf.append(x_train[neigh])
        else:
            count += 1    

    # for i, neigh in enumerate(neighbors):
    #     if(y_train[neigh] != ref_class):
    #         test_dist = neighbor_dist[i]
    #         if len(actual_cf) > n_neighbors: 
    #             if(test_dist < max_of_min_dists):
    #                 ind_to_remove = -1
    #                 for to_remove, ind in tuple_list:
    #                     if to_remove == max_of_min_dists:
    #                         ind_to_remove = ind
    #                         break
    #                 value_to_remove = x_train[ind_to_remove]
    #                 actual_cf.remove(value_to_remove)
    #                 tuple_list.remove((value_to_remove,ind_to_remove))
    #                 actual_cf.append(x_train[neigh])
    #                 max_of_min_dists = max(actual_cf)
    #         else:
    #             tuple_list = (neigh, test_dist)
    #             actual_cf.append(x_train[neigh])
    #             max_of_min_dists = max(actual_cf)

    return actual_cf, count


def get_cfs_df(all_cfs, x_test, y_test, sample_id = 0):
    inv_num, inv_cat = helper.inverse_preprocessing(ds, x_test, sample_id)
    test_dp =  inv_num.tolist() + inv_cat.tolist() + [y_test[sample_id].item()] 
    newdf = []
    # test_dp.append(y_test[sample_id])
    for i,cf in enumerate(all_cfs):
        cf_inv_num, cf_inv_cat = helper.inverse_preprocessing_single(ds, cf)
        newdf.append(cf_inv_num.tolist() + cf_inv_cat.tolist())
        for j in range(len(newdf[i])):
            list = newdf[i]
            if test_dp[j] == list[j]:
                curVal= list[j]
                newdf[i][j] = '-'
    
    cf_df = pd.DataFrame(newdf, columns=ds.numerical_variables + ds.categorical_variables)
    return cf_df

def main():

    # get trianing and test tensors and net trained labels
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    # get predictions 
    # calc_gower_distance(x_test, sample_id=0)
    # print("first")
    neigh_dist, neigh_ind = get_classifier_and_predictions(n_neighbors=5, weights='uniform',
                                   algorithm='auto', leaf_size=30, p=2,
                                   metric='euclidean', metric_params=None,
                                   sample_id=0, distance_metric = "gower")
    n_neighbors = 4
    # print(neigh_dist)
    # print(neigh_ind)

    actual_cf, min_dist = get_cf_min_dist(neigh_dist, neigh_ind, n_neighbors, x_test, y_test, x_train, y_train)
    # print(actual_cf)
    # print(min_dist)

    print("second")
    print(get_cfs_df(actual_cf,x_test,y_test))

# Calling main function 
if __name__=="__main__": 
    main() 