import numpy as np

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
    
    print(f"X = {X}")
    print(f"W = {W}, W.shape = {W.shape}")

if __name__ == "__main__":
    convolution()