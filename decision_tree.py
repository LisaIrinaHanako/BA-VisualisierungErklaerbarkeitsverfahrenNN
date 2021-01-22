import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import helper_methods as helper
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from interactive_ba_preparation_master.model import load_model
from interactive_ba_preparation_master.dataset import German_Credit

# Get pretrained PyTorch model and dataset
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")

# Function to calculate DecisionTreeClassifier
def calc_classifier(x_train, x_test, y_train): 
  
    classifier = DecisionTreeClassifier(
            criterion = 'gini', random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing classification 
    classifier.fit(x_train, y_train) 
    return classifier 


# Function to make predictions 
def prediction(x_test, clf_object): 
  
    # Predicton on test 
    y_pred = clf_object.predict(x_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 

def main(): 
    
    # get trianing and test tensors
    x_train, y_train, x_test, y_test = ds.torch() 

    # reshape all tensors to 2d
    x_train = helper.reshape(x_train)
    y_train = helper.reshape(y_train)
    x_test = helper.reshape(x_test)
    y_test = helper.reshape(y_test)

    # calculate the DecisionTreeRegressor
    classification_results = calc_classifier(x_train, x_test, y_train)
    # y_pred = prediction(x_test, classification_results)

    # plot the resulting Tree 
    tree.plot_tree(classification_results)
    plt.show()

# Calling main function 
if __name__=="__main__": 
    main() 