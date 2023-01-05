import torch

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
    
if __name__ == "__main__":
    # make_tensor()
    # sumsub_tensor()
    # muldiv_tensor()
    reshape_tensor()