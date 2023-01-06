from turtle import down
import argparse
import sys
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

from model.models import *
from loss.loss import *

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
    
    _model = get_model('lenet5')
    
    # LeNet5
    
    if args.mode == "train":
        model = _model(batch=8, n_classes=10, in_channel=1, in_width=32, in_height=32, is_train=True)
        model.to(device)
        model.train() # trian
        
        # optimizer & scheduler
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        criterion = get_criterion(crit='mnist', device=device)
        
        epoch = 15
        iter = 0
        for e in range(epoch):
            total_loss = 0
            for i, batch in enumerate(train_loader):
                img = batch[0]
                gt = batch[1]
                
                img = img.to(device)
                gt = gt.to(device)
                
                out = model(img)
                
                loss_val = criterion(out, gt)
                
                # print(f"loss : {loss_val}")
                print("loss : {}".format(loss_val))
    
if __name__ == "__main__":
    args = parse_args()
    main()
    
# python main.py --mode train --download 1 --output_dir "./output"