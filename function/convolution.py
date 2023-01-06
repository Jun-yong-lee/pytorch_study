import numpy as np

class Conv:
    def __init__(self, batch, in_c, out_c, in_h, in_w, k_h, k_w, dilation, stride, pad):
        self.batch = batch
        self.in_c = in_c
        self.out_c = out_c
        self.in_h = in_h
        self.in_w = in_w
        self.k_h = k_h
        self.k_w = k_w
        self.dilation = dilation
        self.stride = stride
        self.pad = pad
        
    # naive convolution Sliding window metric
        