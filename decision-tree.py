import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from interactive_ba_preparation_master.model import load_model
from interactive_ba_preparation_master.dataset import German_Credit

# Get pretrained PyTorch model and dataset
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")

# Function to calculate DecisionTreeRegressor
def calc_regression(x_train, x_test, y_train): 
  
    regression = DecisionTreeRegressor(
            random_state = 100, 
            max_depth = 3, min_samples_leaf = 5) 
  
    # Performing regression 
    regression.fit(x_train, y_train) 
    return regression 


# Function to make predictions 
def prediction(x_test, clf_object): 
  
    # Predicton on test 
    y_pred = clf_object.predict(x_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to shape x-dimensional tensors into 2d tensors    
def reshape(tensor):
    # get originial shape of the tensor
    shape = tensor.shape
    # select and keep first dimension
    first_dim = shape[0]
    # select left dimensions to calculate size of second dimension of the result
    left_dims = len(shape[1:])
    second_dim = 1
    # calculate size of second dimension of the result
    for cur_dim in range(1,left_dims+1):
        second_dim = second_dim * shape[cur_dim]
    # reshape tensor to 2d tensor
    tensor = tensor.reshape(first_dim, second_dim)
    return tensor

def main(): 
    
    # get trianing and test tensors
    x_train, y_train, x_test, y_test = ds.torch() 

    # reshape all tensors to 2d
    x_train = reshape(x_train)
    y_train = reshape(y_train)
    x_test = reshape(x_test)
    y_test = reshape(y_test)
    
    # calculate the DecisionTreeRegressor
    regression_results = calc_regression(x_train, x_test, y_train)
    # y_pred = prediction(x_test, regression_results)

    # plot the resulting Tree 
    tree.plot_tree(regression_results)
    plt.show()

# Calling main function 
if __name__=="__main__": 
    main() 