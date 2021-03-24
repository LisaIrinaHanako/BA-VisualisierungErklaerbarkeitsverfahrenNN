import torch
import torch.nn as nn
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

def lrp_backward(L, layers, A, y_test_net, sample_id=0):
    # was ist das?
    T = torch.FloatTensor((1.0*(np.arange(2)).reshape([1,2,1])))
    # T = torch.FloatTensor([T])
    R = [None]*L + [(A[-1]*T).data]
    for l in range(1,L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)
        # brauche ich hier noch mehr Layer-Abfragen? 
        if isinstance(layers[l],torch.nn.Linear):
            # ausprobieren?
            # if l <= 1:
            # LRP-Gamma
            rho = lambda p: p + 0.25*p.clamp(min=0)
            incr = lambda z: z+1e-9
            # if 2 <= l <= 3:
            #     rho = lambda p: p
            #     incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data

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
    print("weights: ", w.shape)
    # nenner = (w**2).sum()

    # calculate nenner : sum_i w_{ij}^2
    nenner = []
    for i in range(len(w)):
        nenner.append((w[i]**2).sum())
    
    # calculate zaehler w: sum_j w_{ij}^2
    zaehler_w_sq = []
    for i in range(len(R[1])):
        zaehler_w_sq.append((w[:][i]**2).sum())
    zaehler_r = R[1]
    
    # calculate relevances: R[1] * zaehler w / nenner = (R*w^2)/(sum w^2)
    R[0]=(torch.Tensor(zaehler_r)*torch.Tensor(zaehler_w_sq))/torch.Tensor(nenner)

    return R


    
def main(): 
    # get trianing and test tensors and net trained labels
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    sample_id = 0

    L, layers, A = forward(x_test, y_net_test, sample_id)
    A, R = lrp_backward(L, layers, A, y_net_test, sample_id)
    R = last_step(A, layers, R)

    for t in R:
        print(sum(list(t)))

    
# Calling main function 
if __name__=="__main__": 
    main() 