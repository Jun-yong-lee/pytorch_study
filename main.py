import numpy as np
import time
import torch
import torch.nn as nn

from function.convolution import Conv

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

if __name__ == "__main__":
    convolution()