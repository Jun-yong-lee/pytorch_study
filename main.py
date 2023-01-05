import torch
import numpy as np

# print(torch.__version__)

def make_tensor():
    # int16
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.int16)
    # float
    b = torch.tensor([2], dtype=torch.float32)
    # double
    c = torch.tensor([3], dtype=torch.float64)
    
    print(a, b, c)
    
    tensor_list = [a, b, c]
    
    for t in tensor_list:
        print(f"shape of tensor {t.shape}")
        print(f"datatype of tensor {t.dtype}")
        print(f"device tensor is stored on {t.device}")
    
def sumsub_tensor():
    a = torch.tensor([3, 2])
    b = torch.tensor([5, 3])
    
    print(f"input {a}, {b}")
    
    # sum
    sum = a + b
    print(f"sum : {sum}")
    # sub
    sub = a - b
    print(f"sub : {sub}")
    
    sum_element_a = a.sum()
    print(f"sum_element_a : {sum_element_a}")
    
def muldiv_tensor():
    a = torch.arange(0, 9).view(3, 3)
    b = torch.arange(0, 9).view(3, 3)
    print(f"input tensor :\n {a} \n {b}")
    
    # mat_mul
    c = torch.matmul(a, b) # matrix multiplication
    print(f"mat_mul : {c}")
    
    # elementwise multiplication
    d = torch.mul(a, b)
    print(f"elementwise mul : {d}")
    
def reshape_tensor():
    a = torch.tensor([2, 4, 5, 6, 7, 8])
    print(f"input tensor : \n {a}")
    
    # view
    b = a.view(2, 3)
    print(f"view \n {b}")
    
    # transpose
    bt = b.t()
    print(f"transpose \n {bt}")
    
def access_tensor():
    a = torch.arange(1, 13).view(4, 3)
    print(f"input : \n {a}")
    
    # first col
    print(a[:, 0])
    # first row
    print(a[0, :])
    # [1, 1]
    print(a[1, 1])
    
def transform_numpy():
    a = torch.arange(1, 13).view(4, 3)
    print(f"input : \n {a}")
    
    a_np = a.numpy()
    print(f"numpy : {a_np}")
    
    b = np.array([1, 2, 3])
    bt = torch.from_numpy(b)
    print(bt)
    
def concat_tensor():
    a = torch.arange(1, 10).view(3, 3)
    b = torch.arange(10, 19).view(3, 3)
    c = torch.arange(19, 28).view(3, 3)
    
    abc = torch.cat([a, b, c], dim=0)
    
    print(f"input tensor : \n {a} \n {b} \n {c}")
    print(f"concat : \n {abc}")
    print(abc.shape)
    
def stack_tensor():
    a = torch.arange(1, 10).view(3, 3)
    b = torch.arange(10, 19).view(3, 3)
    c = torch.arange(19, 28).view(3, 3)
    
    abc = torch.stack([a, b, c], dim=0)
    
    print(f"input tensor : \n {a} \n {b} \n {c}")
    print(f"stack : \n {abc}")
    print(abc.shape)
    
def transpose_tensor():
    a = torch.arange(1, 10).view(3, 3)
    print(f"input tensor : \n {a}")
    
    # transpose
    at = torch.transpose(a, 0, 1)
    print(f"transpose : \n {at}")
    
    b = torch.arange(1, 25).view(4, 3, 2)
    print(f"input b tensor : \n {b}")
    
    bt = torch.transpose(b, 0, 2)
    print(f"transpose : \n {bt}")
    print(bt.shape)
    
    bp = b.permute(2, 0, 1) # 0, 1, 2
    print(f"permute : \n {bp}")
    print(bp.shape)
    
if __name__ == "__main__":
    # make_tensor()
    # sumsub_tensor()
    # muldiv_tensor()
    # reshape_tensor()
    # access_tensor()
    # transform_numpy()
    # concat_tensor()
    # stack_tensor()
    transpose_tensor()