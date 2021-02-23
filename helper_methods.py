from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer
import torch

# Function to get test, training and training net samples and labels
def get_samples_and_labels(ds, clf):
    
        # get trianing and test tensors
        x_train, y_train, x_test, y_test = ds.torch() 

        # reshape all tensors to 2d
        x_train = reshape(x_train)
        x_test = reshape(x_test)

        # get correct labels
        y_train = reshape(y_train)
        y_test = reshape(y_test)

        # get net labels
        y_net_train = clf(x_train)
        y_net_test = clf(x_test)

        # format net labels
        y_net_train = torch.argmax(y_net_train.detach(), dim=-1)
        y_net_test = torch.argmax(y_net_test.detach(), dim=-1)

        return x_test, y_test, x_train, y_train, y_net_test, y_net_train

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


# Function to get list of all Feature labels for a single datapoint
def get_feature_labels(ds, number_of_features):
    all_feature_labels = []
    for i in range(number_of_features):
        all_feature_labels.append(ds.index_to_label(i))
    return all_feature_labels


# Function to inverse transform preprocessing of numerical features
# TODO: die kann noch nix
def inverse_preprocessing(ds, datapoint, sample_id):
    transformer = ds.transformer
    scaler = transformer.named_transformers_['num']
    ohe = transformer.named_transformers_['ohe']
    features = datapoint[sample_id]
    num_features = features[0:len(ds.numerical_variables)]
    cat_features = features[len(num_features):]

    inversed_num = scaler.inverse_transform(num_features)
    inversed_cat = ohe.inverse_transform(cat_features.reshape(1, -1) )
    return inversed_num, inversed_cat
    

# Function to make and print predictions for given datapoints and classifier
def print_prediction(datapoint, clf_object): 
  
    # Predicton on test 
    y_pred = clf_object.predict(datapoint) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
    