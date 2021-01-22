import fatf.transparency.predictions.counterfactuals as fatf_cf
import numpy as np
import helper_methods as helper
from pprint import pprint
from interactive_ba_preparation_master.model import load_model
from interactive_ba_preparation_master.dataset import German_Credit

# Get pretrained PyTorch model and dataset
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")

# Helper function to get categorical indices
def get_categorical_idx():
    all_categories = ds.column_names
    categorical_vars = ds.categorical_variables
    max_length = len(categorical_vars)
    categorical_idx = []
    for i, cat in enumerate(all_categories):
        if(i >= max_length):
            break
        if(categorical_vars[i] == cat):
            categorical_idx = categorical_idx.append(i)
    return categorical_idx

# Function to get a Counterfactual Explainer
def get_counterfactual_explainer():
    cat_idx_list = get_categorical_idx()
    explainer = fatf_cf.CounterfactualExplainer(dataset = ds,
                                                categorical_indices=cat_idx_list)
    return explainer

# Function for describing attributes of the datapoint
def describe_data_point(x_test, y_test, classes, data_point_index):
    """Prints out a data point with the specified given index."""
    dp_to_explain = x_test[data_point_index, :]
    dp_to_explain_class_index = int(y_test[data_point_index,0])
    dp_to_explain_class = classes[dp_to_explain_class_index]

    feature_description_template = '    * {} (feature index {}): {}'
    features_description = []
    for i, name in enumerate(ds.column_names):
        dsc = feature_description_template.format(name, i, dp_to_explain[i])
        features_description.append(dsc)
    features_description = ',\n'.join(features_description)

    data_point_description = (
        'Explaining data point (index {}) of class {} (class index {}) with '
        'features:\n{}.'.format(data_point_index, dp_to_explain_class,
                                dp_to_explain_class_index,
                                features_description))

    print(data_point_description)

# Function to print Counterfactuals
def print_CF(dp_cfs, dp_cfs_distances, dp_cfs_predictions, dp_cfs_predictions_names):
    print('\nCounterfactuals for the data point:')
    pprint(dp_cfs)
    print('\nDistances between the counterfactuals and the data point:')
    pprint(dp_cfs_distances)
    print('\nClasses (indices and class names) of the counterfactuals:')
    pprint(dp_cfs_predictions)
    pprint(dp_cfs_predictions_names)
    

# Function to textualise the counterfactuals
def textualise(dp_X, dp_cfs, dp_y, dp_cfs_distances, dp_cfs_predictions):    
    dp_cfs_text = fatf_cf.textualise_counterfactuals(
    dp_X,
    dp_cfs,
    instance_class=dp_y,
    counterfactuals_distances=dp_cfs_distances,
    counterfactuals_predictions=dp_cfs_predictions)
    print(dp_cfs_text)

def main():     
    # get trianing and test tensors
    x_train, y_train, x_test, y_test = ds.torch() 
    # print("x test ", x_test.shape, "\n y test ", y_test.shape, "\n x train " , x_train.shape, "\n y train ", y_train.shape)

    # reshape all tensors to 2d
    x_train = helper.reshape(x_train)
    y_train = helper.reshape(y_train)
    x_test = helper.reshape(x_test)
    y_test = helper.reshape(y_test)

    # print("x test ", x_test.shape, "\n y test ", y_test.shape, "\n x train " , x_train.shape, "\n y train ", y_train.shape)
    classes = ds.label
    # TODO erstmal einen aussuchen
    dp_index = 1

    # describe selected datapiont
    describe_data_point(x_test, y_test, classes, dp_index)
    cf_explainer = get_counterfactual_explainer()
    
    # Get a Counterfactual Explanation tuple for this data point
    dp_cfs, dp_cfs_distances, dp_cfs_predictions = cf_explainer.explain_instance(x_test)
    dp_cfs_predictions_names = np.array(
        [classes[i] for i in dp_cfs_predictions])

    print_CF(dp_cfs, dp_cfs_distances, dp_cfs_predictions, dp_cfs_predictions_names)
    textualise(x_test, dp_cfs, y_test, dp_cfs_distances, dp_cfs_predictions)



    
# Calling main function 
if __name__=="__main__": 
    main() 