from turtle import down
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="MNIST")
    parser.add_argument('--mode', dest='mode', help="train / eval / test",
                        default=False, type=bool)
    parser.add_argument('--download', dest='download', help="download MNIST dataset",
                        default=False, type=bool)
    parser.add_argument('--output_dir', dest='output_dir', help="output directory",
                        default='./output', type=str)
    parser.add_argument('--checkpoint', dest='checkpoint', help="checkpoint trained model",
                        default=None, type=str)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    return args

def get_data():
    my_transform = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))
    ])
    download_root = "./mnist_dataset"
    train_dataset = MNIST(root=download_root,
                          transform=my_transform,
                          train=True,
                          download=args.download)
    eval_dataset = MNIST(root=download_root,
                         transform=my_transform,
                         train=False,
                         download=args.download)
    test_dataset = MNIST(root=download_root,
                         transform=my_transform,
                         train=False,
                         download=args.download)
    
    return train_dataset, eval_dataset, test_dataset

def main():
    print(torch.__version__)
    
    if torch.cuda.is_available():
        print("gpu")
        device = torch.device("cuda")
    else:
        print("cpu")
        device = torch.device("cpu")
    
    # Get MNIST Dataset
    train_dataset, eval_dataset, test_dataset = get_data()
        
if __name__ == "__main__":
    args = parse_args()
    main()