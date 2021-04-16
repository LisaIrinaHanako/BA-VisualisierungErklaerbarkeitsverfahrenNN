import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import torchvision
import helper_methods as helper
import copy


from interactive_ba_preparation_master.model import load_model
from interactive_ba_preparation_master.dataset import German_Credit

# Get pretrained PyTorch model and dataset
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")


def forward(x_test, y_test, sample_id=0):
    layers = [module for module in clf.modules() if type(module) != nn.Sequential]
    layers = layers[1:]
    L = len(layers)
    X=x_test[sample_id]
    A = [X]+[X]*L
    for x in range(len(A)):
        A[x] = torch.FloatTensor(A[x])

    for l in range(L): 
        A[l+1] = layers[l].forward(torch.FloatTensor(A[l]))

    # scores = np.array(A[-1].data.view(-1))
    # ind = np.argsort(-scores)
    # # for i in ind[:10]:
    # #     print('%20s (%3d): %6.3f'%(y_test[i],i,scores[i]))

    return L, layers, A

def newlayer(layer, g):
    # Clone a layer and pass its parameters through the function g.
    layer = copy.deepcopy(layer)
    if isinstance(layer,torch.nn.Linear):
        layer.weight = torch.nn.Parameter(g(layer.weight))
        layer.bias = torch.nn.Parameter(g(layer.bias))
    return layer

def lrp_backward(L, layers, A, y_test_net, sample_id=0, type = "gamma"):
    # was ist das?
    T = torch.FloatTensor((1.0*(np.arange(2)).reshape([1,2,1])))
    # T = torch.FloatTensor([T])
    R = [None]*L + [(A[-1]*T).data]
    for l in range(1,L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)
        # brauche ich hier noch mehr Layer-Abfragen? 
        if isinstance(layers[l],torch.nn.Linear) or isinstance(layers[l],torch.nn.ReLU):
            # ausprobieren?
            # LRP-Epsilon
            if type == "epsilon":
                rho = lambda p: p
                incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
            # LRP-0
            elif type == "0":
                rho = lambda p: p
                incr = lambda z: z+1e-9
            else: 
            # LRP-Gamma (default)
                rho = lambda p: p + 0.25*p.clamp(min=0)
                incr = lambda z: z+1e-9

            z = incr(newlayer(layers[l],rho).forward(A[l]))  
            s = (R[l+1]/z).data                                    # step 2
            (z*s).sum().backward()                                 # step 3 (a)
            c = A[l].grad                                          # step 3 (b)
            R[l] = (A[l]*c).data  
        elif isinstance(layers[l],torch.nn.ReLU):
            #LRP-0
            rho = lambda p: p
            incr = lambda z: z+1e-9
                
            z = incr(newlayer(layers[l],rho).forward(A[l])) 
            s = (R[l+1]/z).data                                    # step 2
            (z*s).sum().backward()                                 # step 3 (a)
            c = A[l].grad                                          # step 3 (b)
            R[l] = (A[l]*c).data  
        else:
            R[l] = R[l+1]
            

    return A, R

def last_step(A, layers, R):

    # lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
    # hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)

    # z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
    # # z -= newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)        # step 1 (b)
    # # z -= newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)        # step 1 (c)
    # s = (R[1]/z).data                                                      # step 2
    # (z*s).sum().backward()                                                 # step 3 (a)
    # c = A[0].grad                                                          # step 3 (b)
    # R[0] = (A[0]*c).data                                                   # step 4
    
    
    # Schritt für den letzten layer 
    # w^2 rule
    A[0] = (A[0].data).requires_grad_(True)
    w = newlayer(layers[0], lambda x: x).weight
    w_transpose = torch.transpose(w,0,1)
    # print("weights: ", w.shape, len(w), len(w[0]))
    # nenner = (w**2).sum()

    nenner = []
    zaehler_w_sq = []
    # calculate nenner : sum_i w_{ij}^2
    # calculate zaehler w: sum_j w_{ij}^2
    for i in range(len(w)):
        w_sum = 0
        for j in range(len(w[0])):
            w_sum += (w[i][j]**2).sum()
        nenner.append(w_sum)
    nenner = torch.Tensor(nenner)
    # print("nenner: ", nenner.shape)
    
    
    for j in range(len(w[0])):
        w_helper = (w_transpose[j].detach().numpy())**2
        zaehler_w_sq.append(w_helper)

    zaehler_w_sq = torch.Tensor(zaehler_w_sq)
    # print("zaehler_w_sq: ", zaehler_w_sq.shape)

    zaehler_r = torch.Tensor(R[1].detach().numpy())
    # print("zaehler_r: ", zaehler_r.shape)
    
    # calculate relevances: R[1] * zaehler w / nenner = (R*w^2)/(sum w^2)
    R[0]=torch.Tensor((np.tensordot(zaehler_w_sq,(zaehler_r/nenner), axes=([1],[0]))))

    return R

# Function to sum up all one-hot encoded column values
def get_barplot_values(vals):
    col_vals_summed = dict(zip(ds.column_names, [0]*(len(ds.column_names))))
    for count, i in enumerate(vals):
        col_onehot = ds.cols_onehot[count]
        col_name = col_onehot.split(':')[0]
        col_vals_summed[col_name] += i.item()
    return col_vals_summed


def main(): 
    # get trianing and test tensors and net trained labels
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    sample_id = 0

    L, layers, A = forward(x_test, y_net_test, sample_id)
    A, R = lrp_backward(L, layers, A, y_net_test, sample_id)
    R = last_step(A, layers, R)

    print(R[0].shape)
    for t in R:
        print(sum(list(t)))

    # get_inversed_lrp_first_layer(R)

    
# Calling main function 
if __name__=="__main__": 
    main() 