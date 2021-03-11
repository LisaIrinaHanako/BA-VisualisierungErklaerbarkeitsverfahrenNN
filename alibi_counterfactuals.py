import torch
import numpy
import torchvision
import helper_methods as helper
import alibi.explainers as AnchorTab

from interactive_ba_preparation_master.model import load_model
from interactive_ba_preparation_master.dataset import German_Credit

# Get pretrained PyTorch model and dataset
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")

def get_cf_explainer(x_train):
    feature_names = ds.cols_onehot
    predict_fn = lambda x: clf.predict(x)
    print(type(clf.predict))
    print(type(feature_names))
    # initialize and fit explainer by passing a prediction function and any other required arguments
    explainer = AnchorTab.AnchorTabular(predictor=predict_fn, feature_names=feature_names)
    explainer.fit(x_train)

    return explainer

def get_explanation(explainer, x_test, sample_id=0):
    # explain an instance
    explanation = explainer.explain(x_test[sample_id])
    return explanation


    
def main(): 
    # get trianing and test tensors and net trained labels
    # x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    x_train, y_train, x_test, y_test = ds.numpy()
    cf_explainer = get_cf_explainer(x_train)
    explanation = get_explanation(cf_explainer, x_test)
    print(explanation)
    
    
# Calling main function 
if __name__=="__main__": 
    main() 