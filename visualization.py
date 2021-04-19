
#region imports
import pandas as pd
import numpy as np
import igraph
from igraph import Graph, EdgeSeq
# import cufflinks as cfl
# import plotly.express as px
# %matplotlib inline
import dash
import dash_daq as daq
from dash.dependencies import Input, Output, State
import dash_table

import dash_core_components as dcc
import dash_html_components as html

import decision_tree as dt
import linear_model as lin
import counterfactuals as cf
import in_sample_counterfactuals as inscf
import dice
import deep_shap as deeps
import lrp
import helper_methods as helper

from plotly.offline import plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# init_notebook_mode(connected=True)
# cfl.go_offline()

from interactive_ba_preparation_master.model import load_model
from interactive_ba_preparation_master.dataset import German_Credit
#endregion


# Get pretrained PyTorch model and dataset
#region global Variables
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
clf = clf.eval()
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")
app = dash.Dash()


x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
x_train_set, y_train_set, x_test_set, y_test_set = ds.numpy()


global_dt = None
global_lin_mod = None
global_cf = None
global_dice = None
dice_preditions = None
global_shap = None
global_lrp = None
global_dp_selection_index = None
global_dp_df = pd.DataFrame()
#endregion

#region Decision Tree
def show_decision_tree_path(datapoint_index, criterion='gini', splitter='best', max_depth=8,
                            min_samples_split=2, min_smp_lf=1,
                            max_features=None,
                            max_leaf_nodes=None, min_impurity_decrease=0,
                            min_impurity_split=0,ccp_alpha=0):
    global global_dt
    global global_dp_selection_index
    if global_dt == None or global_dp_selection_index == datapoint_index:
        classifier = dt.get_classifier(datapoint_index, criterion = criterion, splitter=splitter,
                                        max_depth = max_depth, min_samples_split=min_samples_split,
                                        min_smp_lf=min_smp_lf, max_features=max_features,
                                        max_leaf_nodes=max_leaf_nodes,
                                        min_impurity_decrease=min_impurity_decrease,
                                        min_impurity_split=min_impurity_split, ccp_alpha=ccp_alpha)
        global_dt = classifier

    predictions = global_dt.predict(x_test)


    feature = global_dt.tree_.feature
    threshold = global_dt.tree_.threshold
    node_indicator = global_dt.decision_path(x_test)

    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[node_indicator.indptr[datapoint_index]:
                                    node_indicator.indptr[datapoint_index + 1]]

    fig, dt_edges, dt_nodes = create_and_get_tree(node_index, feature, x_test, threshold, datapoint_index, predictions)

    visual_style = {}
    # visual_style["vertex_size"] = 200
    # plot(fig, visual_style)


    global y_net_test
    accuracy = dt.dt_accuracy(predictions, y_net_test)

    # return dt_edges, dt_nodes
    text = show_decision_tree_text([], node_index, feature, x_test, threshold, datapoint_index, predictions)
    return fig, text, accuracy

def show_decision_tree_text(explanaition_text, node_index, feature, x_test, threshold, datapoint_index, predictions):
    for node_id in node_index:
        inversed_num, inversed_cat = helper.inverse_preprocessing_single(ds, x_test[datapoint_index])

        feature_id = feature[node_id]
        feature_name_onehot = ds.cols_onehot[feature_id]
        is_cat_feature = feature_name_onehot.__contains__(":")

        threshold_value = threshold[node_id]
        pred = predictions[node_id]

        if is_cat_feature:
            feature_name = feature_name_onehot.split(":")[0]
            feature_value = inversed_cat[helper.get_idx_for_feature(feature_name, inversed_cat)]
            feature_value = feature_name_onehot.split(":")[1]
            if (x_test[datapoint_index, feature_id] <= threshold_value):
                threshold_sign = "!="
            else:
                threshold_sign = "="
            explanaition_text.append(html.Div("Knoten {node}: {feat_name} {thres_sign} {feat_val}\n".format(
                                node = node_id, feat_name= feature_name, feat_val=feature_value,
                                thres_sign=threshold_sign)))
        else:
            feature_name = feature_name_onehot
            feature_value = inversed_num[helper.get_idx_for_feature(feature_name, inversed_num)]
            if (x_test[datapoint_index, feature_id] <= threshold_value):
                threshold_sign = "<="
            else:
                threshold_sign = ">"
            explanaition_text.append(html.Div("Knoten {node}: {feat_name} = {feat_val} {thres_sign} {thres_val} \n".format(
                                    node = node_id, feat_name= feature_name, feat_val=feature_value,
                                    thres_sign=threshold_sign, thres_val= threshold_value)))

        explanaition_text.append(html.Br())


    explanaition_text.append("\n Daher ist der Datenpunkt {sample} als Klasse {prediction} bestimmt worden.".format(
                            sample = datapoint_index, prediction = pred))

    explanaition_text.append(html.Br())
    explanaition_text.append(html.Br())

    return explanaition_text

def create_and_get_tree(node_index, feature, x_test, threshold, datapoint_index, predictions):
    # create Tree basis G = Graph.Tree(nr_vertices, 2) # 2 stands for children number
    nr_vertices = 2*max(node_index)
    children_count = 2
    G = Graph.Tree(nr_vertices, children_count)
    layout_def = G.layout_reingold_tilford(mode="in", root=[0])
    number_nodes = len(node_index)

    # intitialisation of position dict
    position = {k: layout_def[k] for k in range(number_nodes)}
    edge_position = {k : layout_def[k] for k in range(number_nodes)}

    # level of the tree: root is 0, children of the root are 1,...
    level = 0.0
    x_pos = 0.0
    edge_x = 0.0
    edge_y = 50.0
    node_x_pos = [x_pos]
    node_y_pos = [level]
    edges = []
    # fill position dict with correct tree positions
    old = 0
    for k in range(number_nodes-1):
        cur = node_index[k]
        next_node = node_index[k+1]
        level = level + 1.0
        if next_node == cur + 1:
            x_old = x_pos
            x_pos = x_pos -8.0
            # add edges' middle positions
            edge_x = 0.5 * (x_old + x_pos) - 2
        else :
            x_old = x_pos
            x_pos = x_pos + 4
            # add edges' middle positions
            edge_x = 0.5 * (x_old + x_pos) + 2
        # set position values to x and y position of nodes
        position[k+1] = [5*x_pos, -100 * level]
        new = old + 1
        # add edge x pos between nodes' x pos
        edges.append((old, new))
        old = new

        # add edge y pos between nodes' y pos
        edge_y = edge_y -100
        edge_position[k+1] = [5*edge_x, edge_y]
        # update x and y pos
        node_x_pos += [5*x_pos]
        node_y_pos += [-100*level]

    edge_x_pos = []
    edge_y_pos = []
    for edge in edges:
        if edge[0] < number_nodes and edge[1] < number_nodes:
            edge_x_pos+=[position[edge[0]][0],position[edge[1]][0], None]
            edge_y_pos+=[position[edge[0]][1],position[edge[1]][1], None]

    labels = get_labels([], node_index, feature, x_test, threshold, datapoint_index, predictions)
    edge_labels = get_edge_labels([], node_index, feature, x_test, threshold, datapoint_index)
    axis = dict(showline=True,
            zeroline=True,
            showgrid=False,
            showticklabels=True,
            autorange=True)

    fig = go.Figure()
    dt_edges = go.Scatter(x=edge_x_pos,
                       y=edge_y_pos,
                       mode='lines',
                       text=edge_labels,
                       line=dict(color='rgb(210,210,210)', width=2),
                       hoverinfo='none'
                       )
    fig.add_trace(dt_edges)


    dt_nodes = go.Scatter(x=node_x_pos,
                      y=node_y_pos,
                      mode='markers + text',
                      name='Pfad',
                      marker=dict(symbol='square',
                                    size=70,
                                    color= '#c9f283', #'#6175c1',    #'#DB4551',
                                    line=dict(color='rgb(50,50,50)', width=1)
                                    ),
                      text=labels,
                      customdata = get_custom_texts(node_index, feature, x_test, threshold, datapoint_index, position),
                      hovertemplate="<br>".join([
                          "%{customdata[1]}",
                          "%{customdata[2]}",
                          "Threshold: %{customdata[3]}",
                          "Pos: %{customdata[0]}"
                      ]),
                      opacity=0.8
                      )
    fig.add_trace(dt_nodes)
    fig.update_layout(title= "Tree test",
              annotations=make_annotations(edge_labels, edge_position),
              font_size=12,
              showlegend=False,
              xaxis=axis,
              yaxis=axis,
              margin=dict(l=40, r=40, b=85, t=100),
              hovermode='closest',
              plot_bgcolor='rgb(248,248,248)'
              )
    return fig, dt_edges, dt_nodes

def get_labels(labels, node_index, feature, x_test, threshold, datapoint_index, predictions):
    for node_id in node_index:
        # check if value of the split feature for sample is below threshold
        # feature_name = dt.get_feature_name(x_test, feature, node_id)
        feature_name = ds.cols_onehot[feature[node_id]]
        feature_value = x_test[datapoint_index, feature[node_id]]
        threshold_value = threshold[node_id]
        feature_name_split = feature_name.split("_")
        pred = predictions[node_id]
        labels.append('<br>'.join(feature_name_split) + '<br>')
    las_node_id = len(node_index)
    #labels[las_node_id] = labels[las_node_id] + 'Klasse: ' + str(predidtions[las_node_id])
    return labels

def get_edge_labels(labels, node_index, feature, x_test, threshold, datapoint_index):
    for node_id in node_index:
        # check if value of the split feature for sample is below threshold
        feature_value = x_test[datapoint_index, feature[node_id]]
        threshold_value = threshold[node_id]
        threshold_sign = ""
        if (feature_value <= threshold_value):
            threshold_sign = " <= "
        else:
            threshold_sign = " > "
        labels.append("{value} {inequality} {threshold}".format(
                  value=feature_value,
                  inequality=threshold_sign,
                  threshold=threshold[node_id]))
    return labels

def get_custom_texts(node_index, feature, x_test, threshold, datapoint_index, position):
    data = []
    i = 0
    for node_id in node_index:
        # check if value of the split feature for sample is below threshold
        # feature_name = dt.get_feature_name(x_test, feature, node_id)
        x_pos = position[i][0]
        y_pos = position[i][1]
        feature_name = ds.cols_onehot[feature[node_id]]
        feature_value = x_test[datapoint_index, feature[node_id]]
        threshold_value = threshold[node_id]
        threshold_sign = ""
        if (feature_value <= threshold_value):
            threshold_sign = " <= "
        else:
            threshold_sign = " > "
        current_pos = ("x = {x_p} , y = {y_p}".format(x_p = x_pos, y_p = y_pos))
        nodes = ("Knoten #{node}".format(node = node_id))
        features = ("Feature {feature_n} = {value}". format(feature_n = feature_name,
                                                                    value = feature_value.item()))
        signs_and_thresholds = ("{sign} {threshold_val}".format(sign = threshold_sign,
                                                                        threshold_val = threshold_value))
        data.append([current_pos, nodes, features, signs_and_thresholds])
        i = i+1
    return data

def make_annotations(labels, position, font_size=10, font_color='rgb(10,40,87)'):
    L=len(position)
    if len(labels)!=L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = []
    for k in range(L):
        if k > 0:
            annotations.append(
                dict(
                    text=labels[k], # or replace labels with a different list for the text within the circle
                    x=position[k][0], y=position[k][1],
                    xref='x1', yref='y1',
                    font=dict(color=font_color, size=font_size),
                    showarrow=False)
            )
    return annotations
#endregion

#region Linear Model
def linear_model_whole_get_xy():
    lin_cols, lin_coeffs, predictions = lin.get_columns_and_coeff()

    return lin_cols, lin_coeffs[0], predictions

def linear_model_single_datapoint_get_xy(datapoint_index = 0, penalty='l2', dual=False, tol=0.0001,
                                        C=1.0, fit_intercept=True, intercept_scaling=1,
                                        random_state=None, solver='sag',
                                        max_iter=100, multi_class='auto', verbose=0,
                                        n_jobs=None, l1_ratio=None):
    lin_cols, lin_coeffs, predictions = lin.get_columns_and_coeff(penalty=penalty, dual=dual, tol=tol,
                                                    C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                                    random_state=random_state, solver=solver,
                                                    max_iter=max_iter, multi_class=multi_class, verbose=verbose,
                                                    n_jobs=n_jobs, l1_ratio=l1_ratio)
    global x_test_set
    new_x_test_set = helper.reshape(x_test_set)

    dp_coeffs = lin_coeffs * new_x_test_set[datapoint_index]
    dp_coeffs = dp_coeffs[0]
    return lin_cols, dp_coeffs, predictions

def show_whole_and_specific_linear_model_plot():
    fig = make_subplots(rows=1, cols=3,
                        specs = [[{'type': 'bar'}, {'type' : 'bar'}, {'type' : 'bar'}]],
                        subplot_titles=("Einzelner Datenpunkt", "Ganzes Modell", "beides"))

    singleX, singleY, predictions = linear_model_single_datapoint_get_xy()
    single_plot = go.Bar(x = singleX, y = singleY, name = "Einzelner Datenpunkt")

    wholeX, wholeY, predictions = linear_model_whole_get_xy()
    whole_plot = go.Bar(x = wholeX, y = wholeY, name = "Ganzes Modell")

    fig.add_trace(single_plot, row=1, col=1)
    fig.add_trace(whole_plot, row=1, col=2)
    fig.add_trace(go.Bar(x=singleX, y=singleY, name = "Einzelner Datenpunkt"), row=1, col=3)
    fig.add_trace(go.Bar(x=wholeX, y=wholeY, name = "Ganzes Modell"), row=1, col=3)

    fig.update_layout(legend_title_text = "")
    fig.update_xaxes(title_text="Feature")
    fig.update_yaxes(title_text="Relevanz")
    # plot(fig)
    return fig

def show_whole_and_specific_linear_model_plot_no_zeros():
    fig = make_subplots(rows=1, cols=3,
                        specs = [[{'type': 'bar'}, {'type' : 'bar'}, {'type' : 'bar'}]],
                        subplot_titles=("Einzelner Datenpunkt", "Ganzes Modell", "beides"))
    datapoint_index = 0
    # x axis has columns, y axis has coeff values
    singleX, singleY, predictions = linear_model_single_datapoint_get_xy(datapoint_index)
    wholeX, wholeY, predictions = linear_model_whole_get_xy()
    newX = []
    newSingleY = []
    newWholeY = []
    count = range(len(singleX))
    for i in count:
        if singleY[i] > 0:
            newX.append(singleX[i])
            newSingleY.append(singleY[i])
            newWholeY.append(wholeY[i])

    single_plot = go.Bar(x = newX, y = newSingleY, name = "Einzelner Datenpunkt")
    whole_plot = go.Bar(x = newX, y = newWholeY, name = "Ganzes Modell")

    fig.add_trace(single_plot, row=1, col=1)
    fig.add_trace(whole_plot, row=1, col=2)
    single_fig = go.Bar(x=newX, y=newSingleY, name = "Einzelner Datenpunkt")
    fig.add_trace(single_fig, row=1, col=3)
    whole_fig = go.Bar(x=newX, y=newWholeY, name = "Ganzes Modell")
    fig.add_trace(whole_fig, row=1, col=3)

    fig.update_layout(legend_title_text = "")
    fig.update_xaxes(title_text="Feature")
    fig.update_yaxes(title_text="Relevanz")
    # plot(fig)
    return single_fig, whole_fig

def show_linear_model_both_in_one(sample_id = 0, penalty='l2', dual=False, tol=0.0001,
                                C=1.0, fit_intercept=True, intercept_scaling=1,
                                random_state=None, solver='sag',
                                max_iter=100, multi_class='auto', verbose=0,
                                n_jobs=None, l1_ratio=None):
    fig = make_subplots(rows=1, cols=1,
                        specs = [[{'type': 'bar'}]],
                        subplot_titles=("Einzelner Datenpunkt und Ganzes Modell"))

    # x axis has columns, y axis has coeff values
    singleX, singleY, predictions = linear_model_single_datapoint_get_xy(sample_id, penalty, dual, tol,
                                                            C, fit_intercept, intercept_scaling,
                                                            random_state, solver,
                                                            max_iter, multi_class, verbose,
                                                            n_jobs, l1_ratio)
    wholeX, wholeY, predictions = linear_model_whole_get_xy()
    newX = []
    newSingleY = []
    newWholeY = []
    count = range(len(singleX))
    for i in count:
        if singleY[i] != 0:
            newX.append(singleX[i])
            newSingleY.append(singleY[i])
            newWholeY.append(wholeY[i])
    single_dp_go = go.Bar(x=newX, y=newSingleY, name = "Einzelner Datenpunkt")
    whole_go = go.Bar(x=newX, y=newWholeY, name = "Ganzes Modell")
    fig.add_trace(single_dp_go, row=1, col=1)
    fig.add_trace(whole_go, row=1, col=1)

    fig.update_layout(legend_title_text = "")
    fig.update_xaxes(title_text="Feature")
    fig.update_yaxes(title_text="Relevanz")
    # plot(fig)
    # return fig
    # return single_dp_go, whole_go

    global y_net_test
    accuracy = lin.lin_mod_accuracy(predictions, y_net_test)

    return fig, accuracy
#endregion

#region In Sample CF
def show_counterfactual_explanation(n_neighbors=5, weights='uniform',
                                   algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None,
                                   sample_id=0, distance_metric = "knn"):
    neigh_dist, neigh_ind = inscf.get_classifier_and_predictions(n_neighbors=n_neighbors, weights=weights,
                                                                algorithm=algorithm, leaf_size=leaf_size, p=p,
                                                                metric=metric, metric_params=metric_params,
                                                                sample_id=sample_id, distance_metric = distance_metric)
    actual_cf, count = inscf.get_cf_min_dist(neigh_dist, neigh_ind, n_neighbors, x_test, y_test, x_train, y_train)
    counterfactuals = inscf.get_cfs_df(actual_cf,x_test,y_test)

    in_sample_cfs = counterfactuals
    
    return in_sample_cfs
#endregion

#region dice
def show_DiCE_visualization(sample_id=0, no_CFs = 4, desired_class="opposite",
                            proximity_weight = 0.5, diversity_weight = 1.0, 
                            yloss_type='hinge_loss', diversity_loss_type='dpp_style:inverse_dist'):
    global global_dice
    global dice_preditions
    # if global_dice == None:
    global_dice = dice.get_counterfactual_explainer()
    dice_preditions = dice.get_counterfactual_explanation(x_test=x_test, explainer=global_dice, sample_id=sample_id, no_CFs=no_CFs, desired_class=desired_class,
                                                            proximity_weight = proximity_weight, diversity_weight = diversity_weight, 
                                                            yloss_type='hinge_loss', diversity_loss_type='dpp_style:inverse_dist')

    cfs = dice.get_cfs_df(dice_preditions, x_test, y_test, sample_id)
    
    return cfs
#endregion

#region shap
def show_DeepSHAP_visualization(sample_id=0, ranked_outputs=None, output_rank_order="max"):
    # get predictions
    classifier = deeps.get_shap_deep_explainer(x_test)
    shap_explanation = deeps.get_all_shap_results(x_test, classifier)

    shap_barplot_vals_0, shap_barplot_vals_1 = deeps.get_barplot_values(shap_explanation, sample_id)

    x_0 = []
    y_0 = []
    x_1 = []
    y_1 = []
    for x in shap_barplot_vals_0:
        x_0.append(x)
        y_0.append(shap_barplot_vals_0[x])

    for x in shap_barplot_vals_1:
        x_1.append(x)
        y_1.append(shap_barplot_vals_1[x])

    plot_class_0 = go.Bar(x = x_0, y = y_0, name = "SHAP 0")
    plot_class_1 = go.Bar(x = x_1, y = y_1, name = "SHAP 1")
    return plot_class_0, plot_class_1
#endregion

#region lrp
def show_lrp_visualization(layer=0, sample_id=0, lrp_type="gamma"):
    L, layers, A = lrp.forward(x_test, y_test, sample_id)
    A, R = lrp.lrp_backward(L, layers, A, y_net_test, sample_id, lrp_type)
    R = lrp.last_step(A, layers, R)
    relevances = None
    newX = []
    newY = []

    if layer == 0:
        lrp_barplot_values = lrp.get_barplot_values(R[layer])
        # filter out values that are 0
        for i,key in enumerate(lrp_barplot_values):
            if lrp_barplot_values[key] != 0:
                newX.append(key)
                newY.append(lrp_barplot_values[key])
    else:
        # filter out values that are 0
        for i in range(len(R[layer])):
            if R[layer][i] != 0:
                newX.append(i)
                newY.append(R[layer][i])
        
    relevances = go.Figure().add_trace(go.Bar(x=newX, y=newY, name = "Layer {no}".format(no=layer)))

    return relevances
#endregion

#region old
def show_all_plots_interactive():
    # show_decision_tree_path()
    # show_whole_and_specific_linear_model_plot()
    # show_whole_and_specific_linear_model_plot_no_zeros()
    # show_linear_model_both_in_one()
    # show_counterfactual_explanation()
    # show_DiCE_visualization()
    # show_DeepSHAP_visualization()
    # show_lrp_visualization()

    dt_edges, dt_nodes = show_decision_tree_path()
    lm_both_fig_single, lm_both_fig_whole = show_whole_and_specific_linear_model_plot_no_zeros()
    # fat_f_cf_fig = show_counterfactual_explanation()
    # dice_fig = show_DiCE_visualization()
    shap_fig_0, shap_fig_1 = show_DeepSHAP_visualization()
    # lrp_fig = show_lrp_visualization()

    fig = make_subplots(rows=2, cols=3,
                        specs = [[{'type': 'scatter'}, {'type' : 'bar'}, {'type' : 'scatter'}],
                                [{'type': 'scatter'}, {'type' : 'bar'}, {'type' : 'bar'}]],
                        subplot_titles=("Entscheidungsbaum", "Lineares Modell", "Conterfactuals (Fat Forensics)",
                                        "Diverse Counterfactual Explanation (DiCE)","DeepSHAP", "Layerwise Relevance Propagation (LRP)"))
    datapoint_index = 0
    fig.add_trace(dt_edges, row=1, col=1)
    fig.add_trace(dt_nodes, row=1, col=1)
    fig.add_trace(lm_both_fig_single, row=1, col=2)
    fig.add_trace(lm_both_fig_whole, row=1, col=2)
    fig.add_trace(dt_edges, row=1, col=3)
    fig.add_trace(dt_nodes, row=2, col=1)
    # fig.add_trace(lm_both_fig_single, row=2, col=2)
    fig.add_trace(lm_both_fig_whole, row=2, col=3)

    fig.add_trace(shap_fig_0, row=2, col=2)
    fig.add_trace(shap_fig_1, row=2, col=2)

    fig.update_layout(legend_title_text = "")
    fig.update_xaxes(title_text="Feature")
    fig.update_yaxes(title_text="Relevanz")

    plot(fig)

def show_dt_lm():

    dt_edges, dt_nodes = show_decision_tree_path()
    lm_both_fig_single, lm_both_fig_whole = show_whole_and_specific_linear_model_plot_no_zeros()
    fig = make_subplots(rows=1, cols=2,
                        specs = [[{'type': 'scatter'}, {'type' : 'bar'}]],
                        subplot_titles=("Entscheidungsbaum", "Lineares Modell"))
    datapoint_index = 0
    fig.add_trace(dt_edges, row=1, col=1)
    fig.add_trace(dt_nodes, row=1, col=1)
    fig.add_trace(lm_both_fig_single, row=1, col=2)
    fig.add_trace(lm_both_fig_whole, row=1, col=2)


    plot(fig)
#endregion

#region Datapoints
def create_datapoint_overview():
    global global_dp_df

    if global_dp_df.empty:
        cat_names_val = ds.categorical_variables
        cats = ds.categories
        num_names_val = ds.numerical_variables

        max_name_val = dict(zip(num_names_val, [0]*len(num_names_val))) 
        min_name_val = dict(zip(num_names_val, [9999999999]*len(num_names_val)))
        sum_name_val = dict(zip(num_names_val, [0]*len(num_names_val)))

        test_list = []
        count = 0
        cat_name_val_dict = dict()
        for category_list in cats:
            temp_dict = dict(zip(category_list, [0]*len(cats)))
            temp = cat_names_val[count]
            cat_name_val_dict[temp] = temp_dict
            count += 1

        for dp in x_test:
            inversed_num, inversed_cat = helper.inverse_preprocessing_single(ds, dp)
            num_name_val_pair = zip(num_names_val, inversed_num.tolist())
            cat_name_val_pair = zip(cat_names_val, inversed_cat.tolist())

            for name,val in num_name_val_pair:
                cur_max = max_name_val[name]
                if val > cur_max:
                    max_name_val[name] = val 
                cur_min = min_name_val[name]
                if val < cur_min:
                    min_name_val[name] = val 
                sum_name_val[name] = sum_name_val[name] + val

            for i, (name, val) in enumerate(cat_name_val_pair):
                for actual in cats[i]:
                    if actual == val:
                        cur_cat = cat_name_val_dict[name]
                        cat_name_val_dict[name][actual] = cur_cat[actual] +1
            

        max_for_features = [x for x in max_name_val.values()] + ['']*len(cat_names_val)
        min_for_features = [x for x in min_name_val.values()] + ['']*len(cat_names_val)
        avg_for_features = [x/len(x_test) for x in sum_name_val.values()] + ['']*len(cat_names_val)

        max_cat_list = []
        for val in cat_name_val_dict.values():
            max_cat_list.append(max(val, key=val.get))
        cat_feature_value = ['']*len(num_names_val) + max_cat_list

        selected_dp = pd.DataFrame({"Numerisch: Maximaler Wert":max_for_features})
        selected_dp.insert(1,"Numerisch: Minimaler Wert", min_for_features, True)
        selected_dp.insert(2,"Numerisch: Durchschnitt Wert", avg_for_features, True)
        selected_dp.insert(3,"Kategorisch: Häufigste Ausprägung", cat_feature_value, True)
        
        feature_names = ds.data.columns
        selected_dp.insert(0,'Eigenschaften',feature_names,True)
        global_dp_df = selected_dp
    return global_dp_df
#endregion


def dash_set_layout():
    #region variable preparations
    feature_names = ds.data.columns

    selected_dp = create_datapoint_overview()
    inversed_num, inversed_cat = helper.inverse_preprocessing_single(ds, x_test[0])
    inversed = inversed_num.tolist() + inversed_cat.tolist()
    selected_dp.insert(1,"Datenpunkt", inversed, True)
    

    in_sample_cfs = show_counterfactual_explanation()
    df_datapoint = pd.DataFrame([inversed], columns=ds.numerical_variables + ds.categorical_variables)
    in_sample_cfs = pd.concat([df_datapoint, in_sample_cfs])
    in_sample_cfs.insert(0,"Datenpunk-Typ", ['originaler Datenpunkt'] + ['Counterfactual {}'.format(i) for i in range(len(in_sample_cfs[:-1]))])

    shap_fig = go.Figure()
    class0, class1 = show_DeepSHAP_visualization()
    shap_fig.add_trace(class0)
    shap_fig.add_trace(class1)

    marks_to_1 = { 0.1*i : "{val}".format(val = helper.round_to_1(0.1*i)) for i in range(10)}
    marks_to_5 = { i : "{val}".format(val = i) for i in range(5)}
    marks_to_10 = { i : "{val}".format(val = i) for i in range(10)}
    marks_to_20 = { i : "{val}".format(val = i) for i in range(20)}
    marks_to_100 = { 10*i : "{val}".format(val = 10*i) for i in range(10)}
    marks_to_200 = { 10*i : "{val}".format(val = 10*i) for i in range(20)}
    marks_shap = {10*i: "{val}".format(val=10*i) for i in range(len(x_test[0]))}
    dice_cfs = show_DiCE_visualization()
    df_datapoint = pd.DataFrame([inversed], columns=ds.numerical_variables + ds.categorical_variables)
    dice_cfs = pd.concat([df_datapoint, dice_cfs])
    dice_cfs.insert(0,"Datenpunk-Typ", ['originaler Datenpunkt'] + ['Counterfactual {}'.format(i) for i in range(len(dice_cfs[:-1]))])
    #endregion

    app.layout = html.Div([
        html.Div(html.H2("Visualisierungen")),
        #region datapoint selection dropdown
        html.Br(),
        html.Div(dcc.Dropdown(
            id='datapoint_selection_dropdown',
            multi=False,
            clearable=True,
            options=[{"label":"Datenpunkt {dp}".format(dp=helper.get_id_for_dp(x_test, c.tolist())), "value":c}
                    for c in x_test],
            value = x_test[0]
        )),
        html.Div(id='dd-output-container'),
        #endregion
        #region datapoint visualization
        dash_table.DataTable(
            id='selected-dp',
            columns=[
                {"name": i, "id": i, "deletable":False, "selectable":True, "hideable":True}
                for i in selected_dp.columns
            ],
            data=selected_dp.to_dict('records')),
        html.Br(),
        #endregion
        #region DT- Bereich
        html.Div([
            html.Div("Tiefe"),
            dcc.Slider(id='input_dt_depth', value=8,
                        max=100, min=2, step=1, marks = marks_to_100),
            html.Div("Anzahl Beispiele für Split"),
            dcc.Slider(id='min_samples_split_dt', value=2,
                        max=100, min=2, step=1, marks = marks_to_100),
            html.Div("Anzahl Beispiele in Blatt"),
            dcc.Slider(id='min_smp_lf_dt', value=1,
                        max=100, min=1, step=1, marks = marks_to_100),
            html.Div("Anzahl Blätter"),
            dcc.Slider(id='max_leaf_nodes_dt', value=None,
                        max=100, min=1, step=1, marks = marks_to_100),
            html.Div("Minimale Unreinheitsänderung"),
            dcc.Slider(id='min_impurity_decrease_dt', value=0.0,
                        max=1, min=0.0, step=1e-3, marks = marks_to_1),
            # html.Div("Minimale Unreinheit für Split"),
            # dcc.Slider(id='min_impurity_split_dt', value=0.0,
            #             max=100, min=0.0, step=1, marks = marks_to_100),
            html.Div("Pruning"),
            dcc.Slider(id='ccp_alpha_dt', value=0.0,
                        max=100, min=0.0, step=1, marks = marks_to_100),
            html.Button(id='submit_button_depth', children='submit'),
            html.Div("Unreinheitsmaß"),
            dcc.RadioItems(id='impurity_criterion', options=[{'label':'Gini', 'value':'gini'}, {'label':'Entropie', 'value':'entropy'}], value='gini'),
            html.Div("Splitter"),
            dcc.RadioItems(id='splitter_dt', options=[{'label':'Best', 'value':'best'}, {'label':'Random', 'value':'random'}], value='best'),
            html.Div("Maximale Anzahl Feature im Blatt"),
            dcc.RadioItems(id='max_features_dt', options=[{'label':'Auto', 'value':'auto'}, {'label':'Wurzel', 'value':'sqrt'}, {'label':'Logarithmus', 'value':'log2'}], value='auto'),
            html.Div(id='dt-accuracy'),
            dcc.Graph(id = 'DT-Graph', figure = show_decision_tree_path(0), animate=False),
            html.Div(id='dt-text')
        ]),
        html.Br(),
        #endregion
        #region Linear Modell- Bereich
        html.Div([
            # html.Div("Dual? (sonst Primal)"), html.Div(id='dual_output'),
            # daq.BooleanSwitch(id='dual', on=False),
            html.Div("Toleranz"),
            dcc.Slider(id='tol', value=1e-4,
                        max=1, min=0, step=1e-5, marks = marks_to_1),
            html.Div("Inverse Regulierung"),
            dcc.Slider(id='C', value=1.0,
                        max=10, min=0, step=1, marks = marks_to_10),
            html.Div("Zusätzliche Konstante addieren?"),
            daq.BooleanSwitch(id='fit_intercept', on=False),
            # html.Div("Intercept Skalierung (nur für liblinear *und* Zusätzliche Konstante addieren = True)"),
            # dcc.Slider(id='intercept_scaling', value=1.0,
            #             max=100, min=1.0, step=1, marks = marks_to_100),
            html.Div("Anzahl Maximale Iterationen"),
            dcc.Slider(id='max_iter', value=100,
                        max=200, min=0, step=1, marks = marks_to_200),
            # html.Div("l1-Ratio"),
            # dcc.Slider(id='l1-ratio', value=None,
            #             max=1, min=0, step=1e-2, marks = marks_to_1),
            html.Button(id='submit_button_lin_mod', children='submit'),
            html.Div("Penalty l2 verwenden?"), html.Div(id='penalty_output'),
            daq.BooleanSwitch(id='penalty', on=False),
            # dcc.RadioItems(id='penalty', options=[{'label':'L1', 'value':'l1'}, {'label':'L2', 'value':'l2'}, {'label':'Elastic Net', 'value':'elasticnet'}, {'label':'None', 'value':'none'}], value='l2'),
            # html.Div("Solver"),
            # dcc.RadioItems(id='solver', options=[{'label':'Newton cg', 'value':'newton-cg'}, {'label':'lbfgs', 'value':'lbfgs'}, {'label':'liblinear', 'value':'liblinear'}, {'label':'sag', 'value':'sag'}, {'label':'saga', 'value':'saga'}], value='lbfgs'),
            # html.Div("Multiclass"),
            # dcc.RadioItems(id='multiclass', options=[{'label':'Auto', 'value':'auto'}, {'label':'Ovr', 'value':'ovr'}, {'label':'Multinomial', 'value':'multinomial'}], value='auto'),
            html.Div(id='lin-mod-accuracy'),
            dcc.Graph(id = 'Linear', figure = show_linear_model_both_in_one(0))
        ]),
        html.Br(),
        #endregion
        #region in sample cf
        html.Div([
            html.Div("Anzahl Nachbarn"),
            dcc.Slider(id='n_neighbors', value=5.0,
                        max=100, min=0.0, step=1, marks = marks_to_100),
            html.Button(id='submit_button_cf', children='submit'),
            html.Div("Distanzmetrik"),
            dcc.RadioItems(id='distance_metric', options=[{'label':'Euklidisch', 'value':'euclidean'}, {'label':'Gower-Distanz', 'value':'gower'}], value='euclidean'),
            # html.Div("Wenn knn: Metrik zur Distanzberechnung für knn"),
            # dcc.RadioItems(id='metric', options=[{'label':'k-NearestNeighbor', 'value':'knn'}, {'label':'Gower-Distanz', 'value':'gower'}], value='knn'),
            # html.Div("Zusätzliche Parameter für gewählte Metrik"),
            # dcc.RadioItems(id='metric_params', options=[{'label':'k-NearestNeighbor', 'value':'knn'}, {'label':'Gower-Distanz', 'value':'gower'}], value='knn'),
            dash_table.DataTable(
            id='in-sample-cf',
            columns=[
                {"name": i, "id": i, "deletable":False, "selectable":True, "hideable":True}
                for i in in_sample_cfs.columns
            ],
            data=in_sample_cfs.to_dict('records'))
        ]),
        html.Br(),
        #endregion
        #region Dice
        html.Div([
            html.Div("Anzahl zu berechnender Counterfactuals"),
            dcc.Slider(id='no_CFs', value=4,
                        max=20, min=1, step=1, marks = marks_to_20),
            html.Div("Nähe zum Datenpunkt (je höher desto näher)"),
            dcc.Slider(id='proximity_weight', value=0.5,
                        max=10, min=0, step=0.1, marks = marks_to_10),
            html.Div("Diversität der CFs (je höher desto diverser)"),
            dcc.Slider(id='diversity_weight', value=0.1,
                        max=10, min=0, step=0.1, marks = marks_to_10),
            html.Button(id='submit_button_dice', children='submit'),
            html.Div("Gewünschte Klasse der Counterfactuals"),
            dcc.RadioItems(id='desired_class', options=[{'label':'Gegenteil der aktuellen', 'value':'opposite'}, {'label':'Aktuelle Klasse', 'value':'same'}], value='opposite'),
            # html.Div("Posthoc Algorithmus"),
            # dcc.RadioItems(id='posthoc_sparsity_algorithm', options=[{'label':'Linear', 'value':'linear'}, {'label':'Binär', 'value':'binary'}], value='linear'),
            dash_table.DataTable(
            id='dice',
            columns=[
                {"name": i, "id": i, "deletable":False, "selectable":True, "hideable":True}
                for i in dice_cfs.columns
            ],
            data=dice_cfs.to_dict('records'))]),
        html.Br(),
        #endregion
        #region Shap
        html.Div([
            # html.Div("Ranked Outputs"),
            # dcc.Slider(id='ranked_outputs', value=None,
            #             max=len(x_test[0]), min=0, step=1, marks = marks_shap),
            # html.Button(id='submit_button_shap', children='submit'),
            # html.Div("Output Sortierung"),
            # dcc.RadioItems(id='output_rank_order', options=[{'label':'Max', 'value':'max'}, {'label':'Max Abs', 'value':'max_abs'},{'label':'Min', 'value':'min'}], value='max'),
            dcc.Graph(id = 'deepShap', figure = shap_fig)
        ]),
        html.Br(),
        #endregion
        #region Lrp
        html.Div([
            # html.Div("Schicht, die angezeigt werden soll"),
            # dcc.Slider(id='layer', value=0,
            #             max=4, min=0, step=1, marks = marks_to_5),
            # html.Button(id='submit_button_lrp', children='submit'),
            html.Div("LRP-Regel für Berechnung der hidden-layer Relevanzen"),
            dcc.RadioItems(id='lrp_type', options=[{'label':'LRP-Gamma', 'value':'gamma'}, {'label':'LRP-Epsilon', 'value':'epsilon'}, {'label':'LRP-0', 'value':'0'}], value='gamma'),
            dcc.Graph(id = 'LRP', figure = show_lrp_visualization(0,0))
        ]),
        html.Br(),
        html.Br()
        #endregion
    ])

@app.callback(
    Output(component_id='selected-dp', component_property='data'),
    [Input(component_id='datapoint_selection_dropdown', component_property='value')])
def update_dp(selected_datapoint):
    print("DP callback")
    idx = helper.get_id_for_dp(x_test, selected_datapoint)
    dp_upd = create_datapoint_overview()
    inversed_num, inversed_cat = helper.inverse_preprocessing_single(ds, x_test[idx])
    inversed = inversed_num.tolist() + inversed_cat.tolist()
    dp_upd.drop(axis=1, labels="Datenpunkt", inplace=True)
    dp_upd.insert(1,"Datenpunkt", inversed, True)

    return dp_upd.to_dict('records')


@app.callback(
    [Output(component_id='DT-Graph', component_property='figure'),
    Output('dt-text', 'children'),
    Output('dt-accuracy', 'children')],
    [Input(component_id='submit_button_depth', component_property='n_clicks')],
    [Input(component_id='datapoint_selection_dropdown', component_property='value')],
    [Input(component_id='impurity_criterion', component_property='value')],
    [Input(component_id='splitter_dt', component_property='value')],
    [Input(component_id='max_features_dt', component_property='value')],
    [Input(component_id='input_dt_depth', component_property='value')],
    [Input(component_id='min_samples_split_dt', component_property='value')],
    [Input(component_id='min_smp_lf_dt', component_property='value')],
    [Input(component_id='max_leaf_nodes_dt', component_property='value')],
    [Input(component_id='ccp_alpha_dt', component_property='value')])
def update_dt_depth(n_clicks, selected_datapoint, criterion, splitter,
                    max_features, dt_depth, min_samples_split, min_smp_lf,
                    max_leaf_nodes, ccp_alpha):
    global global_dp_selection_index
    idx = helper.get_id_for_dp(x_test, selected_datapoint)


    dt_upd, text, accuracy  = show_decision_tree_path(idx, criterion, splitter=splitter, max_depth=dt_depth,
                                            min_samples_split=min_samples_split, min_smp_lf=min_smp_lf,
                                            max_features=max_features,
                                            max_leaf_nodes=max_leaf_nodes,
                                            ccp_alpha=0)
    global_dp_selection_index = idx

    accuracy = "Genauigkeit: {acc}".format(acc=accuracy)
    print("DT Callback ")
    return dt_upd, text, accuracy


@app.callback(
    [Output(component_id='Linear', component_property='figure'),
    Output(component_id='lin-mod-accuracy', component_property='children')
    ],
    [Input(component_id='submit_button_lin_mod', component_property='n_clicks')],
    [Input(component_id='datapoint_selection_dropdown', component_property='value')],
    [Input(component_id='penalty', component_property='on')],
    [Input(component_id='tol', component_property='value')],
    [Input(component_id='fit_intercept', component_property='on')],
    [Input(component_id='C', component_property='value')],
    [Input(component_id='max_iter', component_property='value')])
def update_lin(n_clicks, selected_datapoint, penalty, tol,
                fit_intercept, C, max_iter):
    idx = helper.get_id_for_dp(x_test, selected_datapoint)
    
    if penalty:
        penalty = "l2"
    else:
        penalty = "none"
    
    lin_upd, accuracy= show_linear_model_both_in_one(sample_id = idx, penalty=penalty, tol=tol,
                                                        C=C, fit_intercept=fit_intercept,
                                                        max_iter=max_iter)
    print("Linear Model Callback")

    accuracy = "Genauigkeit: {acc}".format(acc=accuracy)

    return lin_upd, accuracy


@app.callback(
    Output(component_id='in-sample-cf', component_property='data'),
    [Input(component_id='submit_button_cf', component_property='n_clicks')],
    [Input(component_id='datapoint_selection_dropdown', component_property='value')],
    [Input(component_id='distance_metric', component_property='value')],
    [Input(component_id='n_neighbors', component_property='value')])
def update_cf(n_clicks, selected_datapoint, distance_metric, n_neighbors):
    idx = helper.get_id_for_dp(x_test, selected_datapoint)

    cf_upd = show_counterfactual_explanation(n_neighbors=n_neighbors,
                                            sample_id=idx, 
                                            distance_metric=distance_metric)

    
    inversed_num, inversed_cat = helper.inverse_preprocessing_single(ds, x_test[idx])
    inversed = inversed_num.tolist() + inversed_cat.tolist()

    df_datapoint = pd.DataFrame([inversed], columns=ds.numerical_variables + ds.categorical_variables)
    cf_upd = pd.concat([df_datapoint, cf_upd])
    cf_upd.insert(0,"Datenpunk-Typ", ['originaler Datenpunkt'] + ['Counterfactual {}'.format(i) for i in range(len(cf_upd[:-1]))])

    print("CF Callback")

    return cf_upd.to_dict('records')


@app.callback(
    Output(component_id='dice', component_property='data'),
    [Input(component_id='submit_button_dice', component_property='n_clicks')],
    [Input(component_id='datapoint_selection_dropdown', component_property='value')],
    [Input(component_id='desired_class', component_property='value')],
    [Input(component_id='no_CFs', component_property='value')],
    [Input(component_id='proximity_weight', component_property='value')],
    [Input(component_id='diversity_weight', component_property='value')])
def update_dice(n_clicks, selected_datapoint, desired_class,
                no_CFs, proximity_weight, diversity_weight):
    idx = helper.get_id_for_dp(x_test, selected_datapoint)

    if desired_class == "same":
        desired_class = y_net_test[idx]
    
    inversed_num, inversed_cat = helper.inverse_preprocessing_single(ds, x_test[idx])
    inversed = inversed_num.tolist() + inversed_cat.tolist()

    dice_upd = show_DiCE_visualization(idx, no_CFs=no_CFs, desired_class=desired_class,
                                        proximity_weight = proximity_weight, diversity_weight = diversity_weight, 
                                        yloss_type='hinge_loss', diversity_loss_type='dpp_style:inverse_dist')

    
    df_datapoint = pd.DataFrame([inversed], columns=ds.numerical_variables + ds.categorical_variables)
    dice_upd = pd.concat([df_datapoint, dice_upd])
    dice_upd.insert(0,"Datenpunk-Typ", ['originaler Datenpunkt'] + ['Counterfactual {}'.format(i) for i in range(len(dice_upd[:-1]))])

    print("Dice Callback")
    return dice_upd.to_dict('records')


@app.callback(
    Output(component_id='deepShap', component_property='figure'),
    [Input(component_id='datapoint_selection_dropdown', component_property='value')])
def update_ds(selected_datapoint):
    idx = helper.get_id_for_dp(x_test, selected_datapoint)

    ds_upd = go.Figure()
    ds_upd_0, ds_upd_1= show_DeepSHAP_visualization(idx)
    ds_upd.add_trace(ds_upd_0)
    ds_upd.add_trace(ds_upd_1)
    print("Shap Callback")
    return ds_upd


@app.callback(
    Output(component_id='LRP', component_property='figure'),
    [Input(component_id='datapoint_selection_dropdown', component_property='value')],
    [Input(component_id='lrp_type', component_property='value')])
def update_lrp(selected_datapoint, lrp_type):
    idx = helper.get_id_for_dp(x_test, selected_datapoint)

    lrp_upd = show_lrp_visualization(0,idx, lrp_type)
    print("lrp Callback")
    return lrp_upd

def main():
    dash_set_layout()
    print("run server")
    app.run_server()


# Calling main function
if __name__=="__main__":
    main()