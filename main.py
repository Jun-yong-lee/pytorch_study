from turtle import down
import argparse
import sys
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

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

    # Make DataLoader
    train_loader = DataLoader(train_dataset,
                              batch_size=8,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True)
    eval_loader = DataLoader(eval_dataset,
                            batch_size=1,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                        batch_size=1,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False,
                        shuffle=False)
    
    # LeNet5
    
if __name__ == "__main__":
    args = parse_args()
    main()