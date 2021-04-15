import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import helper_methods as helper
import dice_ml
from dice_ml.utils import helpers as dice_helpers 
from interactive_ba_preparation_master.model import load_model
from interactive_ba_preparation_master.dataset import German_Credit

# Get pretrained PyTorch model and dataset
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")

# Function to get a Counterfactual Explainer
def get_counterfactual_explainer():
    ds_dataframe = pd.DataFrame(data=ds.data, columns=ds.column_names[0:-1])
    data = dice_ml.Data(dataframe=ds_dataframe, continuous_features=ds.numerical_variables, outcome_name='Kreditwürdig')
    model = dice_ml.Model(model=clf, backend='PYT') #={"model": "sklearn_model.SklearnModel", "explainer": dice_sklearn.DiceSklearn})

    explainer = dice_ml.Dice(data, model)

    return explainer

# Function to get the Counterfactuals
def get_counterfactual_explanation(x_test, explainer, sample_id = 0, no_CFs = 4, desired_class = "opposite",
                                    proximity_weight = 0.5, diversity_weight = 1.0, 
                                    yloss_type='hinge_loss', diversity_loss_type='dpp_style:inverse_dist'):
    # new_vals_num, new_vals_cat = helper.inverse_preprocessing(ds, x_test, sample_id)
    # new_vals_whole = np.hstack([new_vals_num, new_vals_cat])
    # datapoint_dict = dict(zip(ds.column_names, new_vals_whole))
    # print(datapoint_dict)
    datapoint_dict = dict(zip(ds.cols_onehot, x_test[sample_id][0:-2].numpy()))
    #region var erklärungen
# proximity_weight=0.5, --> je größer, destho näher am Datenpunkt
# diversity_weight=1.0, --> je größer, desto diverser die CFs
# categorical_penalty=0.1, --> raus lassen?
# algorithm="DiverseCF", --> raus lassen?
# features_to_vary="all", --> begrenzt Feature, die verändert werden dürfen
# yloss_type="hinge_loss", --> l2 / log / hinge --> Funktion für yloss (hinge)
# diversity_loss_type="dpp_style:inverse_dist", --> avg / dpp inverse --> Funktion für dpp_diversity (dpp)
# feature_weights="inverse_mad", --> 1/MAD raus lassen? 
# optimizer="pytorch:adam", --> raus lassen
# learning_rate=0.05, --> raus lassen (learning rate für gewählten optimizer)
# min_iter=500, --> min grad desc Iterationen
# max_iter=5000, --> max grad desc Iterationen
# project_iter=0, ??? --> raus lassen
# loss_diff_thres=1e-5, --> minimaler Unterschied, den zwei aufeinanderfolgende Loss-Werte haben dürfen (abbruchkriterium) raus lassen?
# loss_converge_maxiter=1, --> maximale Anzahl Iterationen bevor mit loss diff threshold konvergiert
# verbose=False, --> nur training
# init_near_query_instance=True, --> initialisieren nah am Datenpunkt
# tie_random=False, --> rauslassen
# stopping_threshold=0.5, --> Threshold der Wahrscheinlichkeit einer Klassenvorhersage für ein CF
# posthoc_sparsity_param=0.1, --> 
# posthoc_sparsity_algorithm="linear"
#endregion
    dice_exp = explainer.generate_counterfactuals(datapoint_dict, total_CFs=no_CFs, desired_class=desired_class,
                                                    proximity_weight = proximity_weight, diversity_weight = diversity_weight, 
                                                    yloss_type=yloss_type, diversity_loss_type=diversity_loss_type)
    return dice_exp


def get_cf_explanations_dict(predictions):
    
    df = predictions.final_cfs_df
    cfs = []
    display_sparse_df = True
    show_only_changes = True
    if df is not None and len(df) > 0:
        if predictions.posthoc_sparsity_param == None:
            # print('\nCounterfactual set (new outcome: {0})'.format(predictions.new_outcome))
            if show_only_changes is False:
                cfs.append(df)  # works only in Jupyter notebook
            else:
                newdf = df.values.tolist()
                org = predictions.org_instance.values.tolist()[0]
                for ix in range(df.shape[0]):
                    for jx in range(len(org)):
                        if newdf[ix][jx] == org[jx]:
                            newdf[ix][jx] = '-'
                        else:
                            newdf[ix][jx] = str(newdf[ix][jx])
                cfs.append(pd.DataFrame(newdf, columns=df.columns))  # works only in Jupyter notebook

            # predictions.display_df(df, show_only_changes)
        elif hasattr(predictions.data_interface, 'data_df') and display_sparse_df==True and predictions.final_cfs_sparse is not None:
            # CFs
            # print('\nDiverse Counterfactual set (new outcome: {0})'.format(predictions.new_outcome))
            df = predictions.final_cfs_df_sparse
            if show_only_changes is False:
                cfs.append(df)  # works only in Jupyter notebook
            else:
                newdf = df.values.tolist()
                org = predictions.org_instance.values.tolist()[0]
                for ix in range(df.shape[0]):
                    for jx in range(len(org)):
                        if newdf[ix][jx] == org[jx]:
                            newdf[ix][jx] = '-'
                        else:
                            newdf[ix][jx] = str(newdf[ix][jx])
                cfs.append(pd.DataFrame(newdf, columns=df.columns))  # works only in Jupyter notebook

            # predictions.display_df(df, show_only_changes)
        elif hasattr(predictions.data_interface, 'data_df') and display_sparse_df==True and predictions.final_cfs_sparse is None:
            # print('\nPlease specify a valid posthoc_sparsity_param to perform sparsity correction.. displaying Diverse Counterfactual set without sparsity correction (new outcome : %i)' %(predictions.new_outcome))
            df = predictions.final_cfs #(oder final_cfs_pred)
            if show_only_changes is False:
                cfs.append(df)  # works only in Jupyter notebook
            else:
                newdf = df.values.tolist()
                org = predictions.org_instance.values.tolist()[0]
                for ix in range(df.shape[0]):
                    for jx in range(len(org)):
                        if newdf[ix][jx] == org[jx]:
                            newdf[ix][jx] = '-'
                        else:
                            newdf[ix][jx] = str(newdf[ix][jx])
                cfs.append(pd.DataFrame(newdf, columns=df.columns))  # works only in Jupyter notebook

            # predictions.display_df(df, show_only_changes)
        elif not hasattr(predictions.data_interface, 'data_df'):# for private data
            # print('\nDiverse Counterfactual set without sparsity correction since only metadata about each feature is available (new outcome: ', predictions.new_outcome)
            df = predictions.final_cfs_df #(oder final_cfs_pred)        
            if show_only_changes is False:
                cfs.append(df)  # works only in Jupyter notebook
            else:
                newdf = df.values.tolist()
                org = predictions.org_instance.values.tolist()[0]
                for ix in range(df.shape[0]):
                    for jx in range(len(org)):
                        if newdf[ix][jx] == org[jx]:
                            newdf[ix][jx] = '-'
                        else:
                            newdf[ix][jx] = str(newdf[ix][jx])
                cfs.append(pd.DataFrame(newdf, columns=df.columns))  # works only in Jupyter notebook

            # predictions.display_df(df, show_only_changes)

        else:
            # CFs
            # print('\nDiverse Counterfactual set without sparsity correction (new outcome: ', predictions.new_outcome)
            df = predictions.final_cfs_df #(oder final_cfs_pred)
            if show_only_changes is False:
                cfs.append(df)  # works only in Jupyter notebook
            else:
                newdf = df.values.tolist()
                org = predictions.org_instance.values.tolist()[0]
                for ix in range(df.shape[0]):
                    for jx in range(len(org)):
                        if newdf[ix][jx] == org[jx]:
                            newdf[ix][jx] = '-'
                        else:
                            newdf[ix][jx] = str(newdf[ix][jx])
                cfs.append(pd.DataFrame(newdf, columns=df.columns))  # works only in Jupyter notebook

            # predictions.display_df(predictions.final_cfs_df, show_only_changes)
    return cfs

def get_cfs_df(predictions, x_test, y_test, sample_id = 0):
    df = predictions.final_cfs_df
    inv_num, inv_cat = helper.inverse_preprocessing(ds, x_test, sample_id)
    test_dp = inv_cat.tolist() + inv_num.tolist()
    test_dp.append(y_test[sample_id])
    newdf = df
    cols = ds.categorical_variables + ds.numerical_variables
    cols.append('Kreditwürdig')
    for i,row in enumerate(df):
        keep = False
        for j, val in enumerate(df[row]):
            idx = cols.index(row)
            if test_dp[idx] != df[row][j]:
                keep = True
            else:
                curVal= newdf[row][j]
                newdf = newdf.replace({row:{curVal:'-'}})
        if(not keep):
            # print("drop ", row)
            newdf = newdf.drop(row, axis = 1)

    return newdf


def main():

    # get trianing and test tensors and net trained labels
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    # get predictions 
    classifier = get_counterfactual_explainer()
    counterfactuals = get_counterfactual_explanation(x_test, classifier)
    # counterfactuals.visualize_as_dataframe(show_only_changes=True)
    # print(get_cf_explanations_dict(counterfactuals))
    # print(x_test)
    print(get_cfs_df(counterfactuals, x_test, y_test).to_dict())

# Calling main function 
if __name__=="__main__": 
    main() 