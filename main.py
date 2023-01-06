import numpy as np

from function.convolution import Conv

def convolution():
    print("convolution")
    
    # define the shape of input & weight
    
    in_w = 3
    in_h = 3
    in_c = 1
    out_c = 16
    batch = 1
    k_w = 3
    k_h = 3
    
    X = np.arange(9, dtype=np.float32).reshape([batch, in_c, in_h, in_w])
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
    
    print(f"X = {X}")
    print(f"W = {W}, W.shape = {W.shape}")
    L1 = Convolution.conv(X, W)
    print(f"L1 shape : {L1.shape}")
    print(f"L1 : {L1}")
    


if __name__ == "__main__":
    convolution()