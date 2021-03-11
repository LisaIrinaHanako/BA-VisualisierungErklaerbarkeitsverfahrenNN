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
        print(type(layers[l]))
        # irgendwas stimmt hier nicht mit der ausgewählten layer oder irgendwas anderem
        A[l+1] = layers[l].forward(torch.FloatTensor(A[l]))

    scores = np.array(A[-1].data.view(-1))
    ind = np.argsort(-scores)
    for i in ind[:10]:
        print('%20s (%3d): %6.3f'%(y_test[i],i,scores[i]))

    return L, layers, A

def newlayer(layer, g):
    # Clone a layer and pass its parameters through the function g.
    layer = copy.deepcopy(layer)
    layer.weight = torch.nn.Parameter(g(layer.weight))
    layer.bias = torch.nn.Parameter(g(layer.bias))
    return layer

def lrp_backward(L, layers, A):
    # was ist das?
    T = A[-1].cpu().detach().numpy().tolist()[0]
    T = torch.FloatTensor([T])
    R = [None]*L + [(A[-1]*T).data]
    print(R)
    for l in range(0,L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)
        # brauche ich hier noch mehr Layer-Abfragen? 
        if isinstance(layers[l],torch.nn.Linear): #or isinstance(layers[l],torch.nn.ReLU):
            # ausprobieren?
            if l <= 1:
                # LRP-Gamma
                rho = lambda p: p + 0.25*p.clamp(min=0)
                incr = lambda z: z+1e-9
            if 2 <= l <= 3:
                rho = lambda p: p
                incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
            if l >= 4:
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

    # A[0] = (A[0].data).requires_grad_(True)

    # lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
    # hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)

    # z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
    # z -= newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)    # step 1 (b)
    # z -= newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)    # step 1 (c)
    # s = (R[1]/z).data                                                      # step 2
    # (z*s).sum().backward()                                                 # step 3 (a)
    # c,cp,cm = A[0].grad,lb.grad,hb.grad                                    # step 3 (b)
    # R[0] = (A[0]*c+lb*cp+hb*cm).data                                       # step 4

    # return R


    
def main(): 
    # get trianing and test tensors and net trained labels
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    L, layers, A = forward(x_test, y_test)
    A, R = lrp_backward(L, layers, A)

    print(len(R))
    for t in R:
        print(sum(list(t)))
    print(R)

    
# Calling main function 
if __name__=="__main__": 
    main() 