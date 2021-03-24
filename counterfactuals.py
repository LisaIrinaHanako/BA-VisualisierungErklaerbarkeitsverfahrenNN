import fatf.transparency.predictions.counterfactuals as fatf_cf
import numpy as np
import helper_methods as helper
from pprint import pprint
from skorch import NeuralNetClassifier
from interactive_ba_preparation_master.model import load_model
from interactive_ba_preparation_master.dataset import German_Credit

# Get pretrained PyTorch model and dataset
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")


# Function to get a Counterfactual Explainer
def get_counterfactual_explainer(dataset_to_use):
    cat_idx_list = helper.get_categorical_idx(ds)
    # print(type(clf.predict))
    explainer = fatf_cf.CounterfactualExplainer(predictive_function = clf.predict,
                                                dataset = dataset_to_use,
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

    # print(data_point_description)

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
    # print(dp_cfs_text)

def main():     
    # get trianing and test tensors
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)

    classes = list(set(ds.label))

    # TODO erstmal einen aussuchen
    dp_index = 1

    x_train_set, y_train_set, x_test_set, y_test_set = ds.numpy()

    # reshape all arrays to 2d
    x_train_set = helper.reshape(x_train_set)
    y_train_set = helper.reshape(y_train_set)
    x_test_set = helper.reshape(x_test_set)
    y_test_set = helper.reshape(y_test_set)


    # describe selected datapiont
    describe_data_point(x_test, y_test, classes, dp_index)
    cf_explainer = get_counterfactual_explainer(x_train_set)
    
    
    # Get value of y_net_test at dp_index
    y_net_test_value = np.array(y_net_test)[dp_index].item()

    # Get a Counterfactual Explanation tuple for this data point
    dp_2_cf_tuple = cf_explainer.explain_instance(instance = x_test_set[dp_index, : ], 
                                                counterfactual_class = y_net_test_value)
    dp_cfs = dp_2_cf_tuple[0] 
    dp_cfs_distances = dp_2_cf_tuple [1]
    dp_cfs_predictions = dp_2_cf_tuple [2]

    dp_cfs_predictions_names = np.array(
        [classes[i] for i in dp_cfs_predictions])

    # print and textualise counterfactual explanations
    # print_CF(dp_cfs, dp_cfs_distances, dp_cfs_predictions, dp_cfs_predictions_names)
    # textualise(x_test_set[dp_index, : ], dp_cfs, y_test_set[dp_index].item(), dp_cfs_distances, dp_cfs_predictions)

    
# Calling main function 
if __name__=="__main__": 
    main() 