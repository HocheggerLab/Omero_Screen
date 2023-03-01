import matplotlib.pyplot as plt

from CNN_model import CNN
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

import torch.optim as optim
from torch.utils.data import DataLoader

from CCdataset import CCdata
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
load_model=False
from torch.utils.tensorboard import SummaryWriter

# save weights of model
def save_checkpoint(state,filename='./CNN_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state,filename)
# check the accuracy of model predict with ground truth
def val_accuracy(loader,model,device,best_loss,epoch,writer,optimizer):
    correct=0
    total=0
    model.eval()
    val_loss_batch=[]
    criterion=nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device=device)
            y=y.to(device=device)
            val_scores = model(x)
            val_loss_batch.append(criterion(val_scores, y).item())
            _,predictions= val_scores.max(1)
            correct += (predictions==y).sum().item()
            total+=y.size(0)
        val_loss_mean = sum(val_loss_batch) / len(loader.dataset)
        correct /= total
        print('Loss/val',val_loss_mean,epoch)
        writer.add_scalar('Loss/val',val_loss_mean,epoch)
        writer.add_scalar('Accuracy/val', correct, epoch)
    model.train()
    return val_loss_mean



def main():
    # Create a summary writer to log the loss and validation loss
    writer = SummaryWriter('./runs/logs')

    in_channel = 3
    num_classes = 2
    learning_rate = 0.001
    num_epochs =50
    D_PATH = '../CNN_pytorch/data/'
    # device using mps
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    # increase the samples by augmentation
    transform = A.Compose(
        [
            A.Resize(32, 32, ),
            # A.CenterCrop(60, 60,),
            # A.Resize(32, 32),
            A.Rotate(limit=40, p=0.6),
            A.HorizontalFlip(p=0.6),
            A.RandomBrightnessContrast(p=0.4),
            A.VerticalFlip(p=0.6),
            A.VerticalFlip(p=0.6),
            A.OneOf(
                [
                    # A.Blur(blur_limit=3, p=0.8),
                    # A.Blur(blur_limit=3, p=0.5),
                    # A.ColorJitter(p=0.6),
                ], p=1.0
            ),
            ToTensorV2(),
        ]
    )
    #  load dataset
    datasets = CCdata(D_PATH=D_PATH, transform=transform)
    # Set the RNG seed for reproducibility
    torch.manual_seed(0)

    # define the sizes of training, validation and test dataset
    train_size = int(0.90* len(datasets))
    print(train_size)
    val_size = int(0.10 * len(datasets))
    print(val_size)
    test_size = len(datasets) - train_size - val_size

    # Use random_split to split the dataset into the three subsets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(datasets, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(0))

    # Create data loaders for each dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # load model
    model = CNN(in_channels=in_channel, num_classes=num_classes).to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = 0.2
    # training the model
    for epoch in range(num_epochs):

        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)
        # forward
            scores= model(data)
            loss = criterion(scores, targets)
         # backward
            optimizer.zero_grad()
            loss.backward()
         # gradient descent or adam step
            optimizer.step()
        val_loss_mean=val_accuracy(val_loader, model, device, best_loss, epoch, writer, optimizer)
        # Save best model based on validation loss
        if val_loss_mean <= best_loss:
            best_loss = val_loss_mean
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)

    # Close the summary writer
    writer.close()

if __name__=='__main__':
    main()
