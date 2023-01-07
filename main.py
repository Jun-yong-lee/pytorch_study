import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from function.convolution import Conv
from function.pool import Pool
from function.activation import *

def convolution():
    print("convolution")
    
    # define the shape of input & weight
    
    in_w = 6
    in_h = 6
    in_c = 3
    out_c = 16
    batch = 1
    k_w = 3
    k_h = 3
    
    X = np.arange(in_w * in_h * in_c * batch, dtype=np.float32).reshape([batch, in_c, in_h, in_w])
    W = np.array(np.random.standard_normal([out_c, in_c, k_h, k_w]), dtype=np.float32)
    
    Convolution = Conv(batch=batch,
                in_c=in_c,
                out_c=out_c,
                in_h=in_h,
                in_w=in_w,
                k_h=k_h,
                k_w=k_w,
                dilation=1,
                stride=1,
                pad=0)
    
    # print(f"X = {X}")
    # print(f"W = {W}, W.shape = {W.shape}")
    
    L1_time = time.time()
    
    for i in range(5):
        L1 = Convolution.conv(X, W)            
    print(f"L1 time : {time.time() - L1_time}")
    print(f"L1 : {L1}")

    L2_time = time.time()
    for i in range(5):
        L2 = Convolution.gemm(X, W)
    print(f"L2 time : {time.time() - L2_time}")
    print(f"L2 : {L2}")
    
    torch_conv = nn.Conv2d(in_c,
                           out_c,
                           kernel_size=k_h,
                           stride=1,
                           padding=0,
                           bias=False,
                           dtype=torch.float32)
    torch_conv.weight = torch.nn.Parameter(torch.tensor(W))
    
    L3_time = time.time()
    for i in range(5):
        L3 = torch_conv(torch.tensor(X, requires_grad=False, dtype=torch.float32))
    print(f"L3 time : {time.time() - L3_time}")
    print(f"L3 : {L3}")

def forward_net():
    """_summary_
    'Conv - Pooling - FC' model inference code 
    """
    #define
    batch = 1
    in_c = 3
    in_w = 6
    in_h = 6
    k_h = 3
    k_w = 3
    out_c = 1
    
    X = np.arange(batch*in_c*in_w*in_h, dtype=np.float32).reshape([batch,in_c,in_w,in_h])
    W1 = np.array(np.random.standard_normal([out_c,in_c,k_h,k_w]), dtype=np.float32)
    
    Convolution = Conv(batch = batch,
                        in_c = in_c,
                        out_c = out_c,
                        in_h = in_h,
                        in_w = in_w,
                        k_h = k_h,
                        k_w = k_w,
                        dilation = 1,
                        stride = 1,
                        pad = 0)
    
    L1 = Convolution.gemm(X,W1)
    
    print("L1 shape : ", L1.shape)
    print(L1)
    
    Pooling = Pool(batch=batch,
                   in_c = 1,
                   out_c = 1,
                   in_h = 4,
                   in_w = 4,
                   kernel=2,
                   dilation=1,
                   stride=2,
                   pad = 0)
    
    L1_MAX = Pooling.pool(L1)
    print("L1_MAX shape : ", L1_MAX.shape)
    print(L1_MAX)
    
    #fully connected layer
    W2 = np.array(np.random.standard_normal([1, L1_MAX.shape[1] * L1_MAX.shape[2] * L1_MAX.shape[3]]), dtype=np.float32)
    Fc = FC(batch = L1_MAX.shape[0],
            in_c = L1_MAX.shape[1],
            out_c = 1,
            in_h = L1_MAX.shape[2],
            in_w = L1_MAX.shape[3])

    L2 = Fc.fc(L1_MAX, W2)
    
    print("L2 shape : ", L2.shape)
    print(L2)
    
def plot_activation():
    """_summary_
    Plot the activation output of [-10,10] inputs
    activations : relu, leaky_relu, sigmoid, tanh
    """    
    x = np.arange(-10,10,1)
    
    out_relu = relu(x)
    out_leaky = leaky_relu(x)
    out_sigmoid = sigmoid(x)
    out_tanh = tanh(x)

    #print(out_relu, out_leaky, out_sigmoid, out_tanh)
    
    plt.plot(x, out_relu, 'r', label='relu')
    plt.plot(x, out_leaky, 'b', label='leaky')
    plt.plot(x, out_sigmoid, 'g', label='sigmoid')
    plt.plot(x, out_tanh, 'bs', label='tanh')
    plt.ylim([-2,2])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # convolution()
    # forward_net()
    plot_activation()