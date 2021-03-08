import pandas as pd
import numpy as np
import igraph
from igraph import Graph, EdgeSeq
# import cufflinks as cfl
# import plotly.express as px
# %matplotlib inline
import dash
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html

import decision_tree as dt
import linear_model as lin
import counterfactuals as cf
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

# Get pretrained PyTorch model and dataset
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")
app = dash.Dash()


def show_decision_tree_path(datapoint_index = 0):
    print("DT Path: nothing here yet")
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    classifier = dt.get_classifier()
    predictions = classifier.predict(x_test)
    
    feature = classifier.tree_.feature
    threshold = classifier.tree_.threshold
    node_indicator = classifier.decision_path(x_test)
    
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[node_indicator.indptr[datapoint_index]:
                                    node_indicator.indptr[datapoint_index + 1]]

    fig, dt_edges, dt_nodes = create_and_get_tree(node_index, feature, x_test, threshold, datapoint_index, predictions)

    visual_style = {}
    # visual_style["vertex_size"] = 200
    # plot(fig, visual_style)

    return dt_edges, dt_nodes

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
            print(x_old, x_pos, 0.5* (x_old+x_pos))
            # add edges' middle positions
            edge_x = 0.5 * (x_old + x_pos) - 2
        else : 
            x_old = x_pos
            x_pos = x_pos + 4
            print(x_old, x_pos, 0.5*(x_old+x_pos))
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
    # Y = [layout_def[k][1] for k in range(number_nodes)]
    # M = max(Y)
    print(edge_position)

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
            range = (-100*level -2, 2))

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
        # print("decision node {node} : (datapoint[{sample}, (original: still open) scaled: {feature}] = {value}) "
        #       "{inequality} {threshold})".format(
        #           node=node_id,
        #           sample=datapoint_index,
        #           feature=feature_name,
        #           value=feature_value,
        #           inequality=threshold_sign,
        #           threshold=threshold[node_id]))
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

def linear_model_whole_get_xy():
    lin_cols, lin_coeffs = lin.get_columns_and_coeff()

    x_train_set, y_train_set, x_test_set, y_test_set = ds.numpy()
    x_test_set = helper.reshape(x_test_set)
    # since lin_coeffs is array of shape (1,61): use only dimension 1
    return lin_cols, lin_coeffs[0]
    # plot(fig)
    # return fig

def linear_model_single_datapoint_get_xy(datapoint_index = 0):
    lin_cols, lin_coeffs = lin.get_columns_and_coeff()

    x_train_set, y_train_set, x_test_set, y_test_set = ds.numpy()
    x_test_set = helper.reshape(x_test_set)
    
    print(ds.data.iloc[datapoint_index])
    # x_test_set[0,:], 
    dp_coeffs = lin_coeffs[datapoint_index] * x_test_set[0]
    return lin_cols, dp_coeffs
    # plot(fig)
    # return fig

def show_whole_and_specific_linear_model_plot():
    fig = make_subplots(rows=1, cols=3, 
                        specs = [[{'type': 'bar'}, {'type' : 'bar'}, {'type' : 'bar'}]],
                        subplot_titles=("Einzelner Datenpunkt", "Ganzes Modell", "beides"))

    singleX, singleY = linear_model_single_datapoint_get_xy()
    single_plot = go.Bar(x = singleX, y = singleY, name = "Einzelner Datenpunkt")

    wholeX, wholeY = linear_model_whole_get_xy()
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
    singleX, singleY = linear_model_single_datapoint_get_xy(datapoint_index)
    wholeX, wholeY = linear_model_whole_get_xy()
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

def show_linear_model_both_in_one():
    fig = make_subplots(rows=1, cols=1, 
                        specs = [[{'type': 'bar'}]],
                        subplot_titles=("Einzelner Datenpunkt und Ganzes Modell"))

    # x axis has columns, y axis has coeff values
    singleX, singleY = linear_model_single_datapoint_get_xy()
    wholeX, wholeY = linear_model_whole_get_xy()
    newX = []
    newSingleY = [] 
    newWholeY = []
    count = range(len(singleX))
    for i in count:
        if singleY[i] > 0:
            newX.append(singleX[i])
            newSingleY.append(singleY[i])
            newWholeY.append(wholeY[i])

    fig.add_trace(go.Bar(x=newX, y=newSingleY, name = "Einzelner Datenpunkt"), row=1, col=3)
    fig.add_trace(go.Bar(x=newX, y=newWholeY, name = "Ganzes Modell"), row=1, col=3)
    
    fig.update_layout(legend_title_text = "")
    fig.update_xaxes(title_text="Feature")
    fig.update_yaxes(title_text="Relevanz")
    # plot(fig)
    return fig


def show_counterfactual_explanation():
    print("CF Text: nothing here yet")

def show_DiCE_visualization(sample_id=0):
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    classifier = dice.get_counterfactual_explainer()
    predictions = dice.get_counterfactual_explanation(x_test, classifier)
    cfs = dice.get_cfs_df(predictions, x_test, y_test, sample_id)
    print(cfs)
    print(x_test[sample_id])
    app.layout = html.Div([
        dash_table.DataTable(
            id='dt', 
            columns=[
                {"name": i, "id": i, "deletable":True, "selectable":True, "hideable":True}
                for i in cfs.columns
            ],
            data=cfs.to_dict('records')),
        html.Br(),
        html.Br(),
        html.Div(html.H2("Überschrift")),
        html.Div(id='')
    ])
    # fig = go.Table(header=dict(values=))

def show_DeepSHAP_visualization(sample_id=0):
    # get trianing and test tensors and net trained labels
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    # get predictions 
    classifier = deeps.get_shap_deep_explainer(x_train)
    shap_explanation = deeps.get_shap_explanation(x_test, classifier)
    print(shap_explanation[0][sample_id].shape)
    plot_class_0 = go.Bar(x = ds.cols_onehot, y = shap_explanation[0][sample_id], name = "SHAP 0")
    plot_class_1 = go.Bar(x = ds.cols_onehot, y = shap_explanation[1][sample_id], name = "SHAP 1")
    return plot_class_0, plot_class_1

def show_lrp_visualization():
    print("LRP: nothing here yet")

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
    print("oh-oh, no interactive visuals yet")

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

def main():
    # dt.main()
    # lin.main()
    # cf.main()
    # dice.main()
    # ds.main()
    # lrp.main()
    # show_all_plots_interactive()
    # show_dt_lm()
    show_DiCE_visualization()
    print("run server")
    app.run_server()
    

# Calling main function 
if __name__=="__main__": 
    main() 