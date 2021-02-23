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
def inverse_preprocessing(ds, datapoint, sample_id, feature_name):
    scaler = StandardScaler()
    
    transformer = ColumnTransformer(transformers=[("num", scaler, ds.numerical_variables), ("ohe", OneHotEncoder(), ds.categorical_variables)])
    transformer.fit_transform(ds.data)


    feature = datapoint[sample_id, feature_name]
    for i in range(len(datapoint[sample_id,:])):
        if type(datapoint[sample_id, i]) != str:
            scaler.fit(datapoint)

    if type(feature) != str:
        inversed = scaler.inverse_transform(feature)
    else:
        inversed = datapoint[sample_id, :]

    return inversed[sample_id]
    

# Function to make and print predictions for given datapoints and classifier
def print_prediction(datapoint, clf_object): 
  
    # Predicton on test 
    y_pred = clf_object.predict(datapoint) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
    

def make_tree_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
    L=len(pos)
    if len(text)!=L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = []
    labels = zip(ds.cols_onehot, )
    for k in range(L):
        annotations.append(
            dict(
                text=labels[k],
                x=pos[k][0], y=2*M-position[k][1],
                xref='x1', yref='y1',
                font=dict(color=font_color, size=font_size),
                showarrow=False)
        )
    return annotations