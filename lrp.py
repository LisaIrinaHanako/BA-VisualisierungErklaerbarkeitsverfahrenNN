import torch
import numpy
import torchvision
import helper_methods as helper


from interactive_ba_preparation_master.model import load_model
from interactive_ba_preparation_master.dataset import German_Credit

# Get pretrained PyTorch model and dataset
clf = load_model(path="./interactive_ba_preparation_master/net.pth")
ds = German_Credit(path="./interactive_ba_preparation_master/german.data")


def forward(x_test, sample_id=0):
    layers = list(clf.modules())
    L = len(layers)
    A = x_test.tolist()+[None]*L
    # print(A)
    i = 0
    for l in range(L): 
        print(i)
        i += 1
        print(type(layers[l]))
        A[l+1] = layers[l].forward(A[l])

    scores = numpy.array(A[-1].data.view(-1))
    ind = numpy.argsort(-scores)
    # for i in ind[:10]:
    #     print('%20s (%3d): %6.3f'%(utils.imgclasses[i][:20],i,scores[i]))

    return L, layers, A

def backward(L, layers, A):
    T = torch.FloatTensor((1.0*(numpy.arange(1000)==483).reshape([1,1000,1,1])))
    R = [None]*L + [(A[-1]*T).data]

    for l in range(1,L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)
        if isinstance(layers[l],torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)
        if isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):
            if l <= 16:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
            if 17 <= l <= 30: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
            if l >= 31:       rho = lambda p: p;                       incr = lambda z: z+1e-9

            z = incr(utils.newlayer(layers[l],rho).forward(A[l]))  # step 1
            s = (R[l+1]/z).data                                    # step 2
            (z*s).sum().backward(); c = A[l].grad                  # step 3
            R[l] = (A[l]*c).data                                   # step 4
        else:
            R[l] = R[l+1]

    A[0] = (A[0].data).requires_grad_(True)

    lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
    hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)

    z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
    z -= utils.newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)    # step 1 (b)
    z -= utils.newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)    # step 1 (c)
    s = (R[1]/z).data                                                      # step 2
    (z*s).sum().backward(); c,cp,cm = A[0].grad,lb.grad,hb.grad            # step 3
    R[0] = (A[0]*c+lb*cp+hb*cm).data                                       # step 4

    return R


    
def main(): 
    # get trianing and test tensors and net trained labels
    x_test, y_test, x_train, y_train, y_net_test, y_net_train = helper.get_samples_and_labels(ds, clf)
    
    L, layers, A = forward(x_test)
    R = backward(L, layers, A)

    print(R)

    
# Calling main function 
if __name__=="__main__": 
    main() 