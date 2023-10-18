import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import models


# Showing images
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Plotting data
def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()
class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # Look untill the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                # Look untill a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert("RGB")
        img1 = img1.convert("RGB")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidean distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive

class SiameseNetwork(nn.Module):
    def __init__(self, backbone="resnet18"):
        '''
        Creates a siamese network with a network from torchvision.models as backbone.

            Parameters:
                    backbone (str): Options of the backbone networks can be found at https://pytorch.org/vision/stable/models.html
        '''

        super().__init__()

        if backbone not in models.__dict__:
            raise Exception("No model named {} exists in torchvision.models.".format(backbone))

        # Create a backbone network from the pretrained models provided in torchvision.models
        self.backbone = models.__dict__[backbone](pretrained=True, progress=True)

        self.criterion = ContrastiveLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.0005)
        self.model_path = "../networks/siamese_{}".format(backbone)

        # Get the number of features that are outputted by the last layer of backbone network.
        out_features = list(self.backbone.modules())[-1].out_features

        # Create an MLP (multi-layer perceptron) as the classification head.
        # Classifies if provided combined feature vector of the 2 images represent same player or different.
        self.cls_head = nn.Sequential(
            nn.Linear(out_features, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 2)
        )
    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        output = self.backbone(x)
        output = output.view(output.size()[0], -1)
        output = self.cls_head(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

    def train_model(self, num_epoch, dataloader):
        counter = []
        loss_history = []
        iteration_number = 0
        best_loss = 1000000
        for epoch in range(num_epoch):
            self.train()
            # Iterate over batches
            for i, (img0, img1, label) in enumerate(dataloader, 0):

                # Send the images and labels to CUDA
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                # Zero the gradients
                self.optimizer.zero_grad()

                # Pass in the two images into the network and obtain two outputs
                output1, output2 = self.forward(img0, img1)

                # Pass the outputs of the networks and label into the loss function
                loss_contrastive = self.criterion(output1, output2, label)

                # Calculate the backpropagation
                loss_contrastive.backward()

                # Optimize
                self.optimizer.step()

                # Every 10 batches print out the loss
                if i % 10 == 0:
                    print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
                    iteration_number += 10

                    counter.append(iteration_number)
                    loss_history.append(loss_contrastive.item())
                if loss_contrastive.item() <= best_loss:
                    best_loss = loss_contrastive.item()
        answer = input(f"Best loss is: {best_loss}. Would you like to save model?(y/n):")
        if answer == 'y':
            self.save_model()
            print("Model saved")
        self.eval()
        show_plot(counter, loss_history)

    def load_model(self):
        self.load_state_dict(torch.load(self.model_path))
        self.eval()
    def save_model(self):
        torch.save(self.state_dict(), self.model_path)

if __name__ == '__main__':
    folder_dataset = datasets.ImageFolder(root="../../../Documents/faces/training/")

    # Resize the images and transform to tensors
    transformation = transforms.Compose([transforms.Resize((100, 100)),
                                         transforms.ToTensor()
                                         ])

    # Initialize the network
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transformation)

    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=8,
                                  batch_size=64)
    net = SiameseNetwork(backbone='resnet18').cuda()

    net.train_model(10, train_dataloader)
