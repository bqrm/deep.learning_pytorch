# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2019/12/14

# https://blog.csdn.net/jeryjeryjery/article/details/79426907

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import dataloader.custom_transforms as tr

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, np.cumprod(x.size()[1:])[-1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def test(model, criterion, data_loader, use_cuda=False):
    test_loss = 0
    correct = 0

    model.eval()

    for sample, target in data_loader:
        if use_cuda:
            sample = sample.cuda()
            target = target.cuda()

        output = model(sample)
        test_loss += criterion(output, target).data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(data_loader)  # loss function already averages over batch size
    acc = 100. * correct / len(data_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n".format(
        test_loss, correct, len(data_loader.dataset), acc))
    return acc, test_loss


def train(model, epoch, criterion, optimizer, data_loader, use_cuda=False):
    model.train()

    for batch_idx, (batch_data, batch_label) in enumerate(data_loader):
        sample, target = batch_data, batch_label

        if use_cuda:
            sample = sample.cuda()
            target = target.cuda()

        output = model(sample)

        optimizer.zero_grad()

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        if 0 == batch_idx % 10:
            print("train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}".format(epoch,
                                                                           (batch_idx + 1) * len(sample),
                                                                           len(data_loader.dataset),
                                                                           100 * (batch_idx + 1) / len(data_loader),
                                                                           loss.data))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device {}".format(device))

    batch_num_size = 64
    epochs = 5

    data_path = "../../../../DataSets"
    train_loader = DataLoader(
        datasets.MNIST(data_path, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_num_size, shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST(data_path, train=False, transform=transforms.ToTensor()),
        batch_size=batch_num_size, shuffle=True)

    model = LeNet5().to(device)

    criterion = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    check_point_name = "mnist_net.t7"
    check_point_path = "../../checkpoint/"
    full_path = os.path.join(check_point_path, check_point_name)
    if os.path.exists(full_path):
        print("Loading model from \"%s\"" % full_path)
        model.load_state_dict(torch.load(full_path, map_location=lambda storage, loc: storage))
        test(model, criterion, test_loader, torch.cuda.is_available())
    else:
        print("Training model")
        for epoch in range(1, epochs + 1):
            train(model, epoch, criterion, optimizer, train_loader, torch.cuda.is_available())
            test(model, criterion, test_loader, torch.cuda.is_available())

        if not os.path.exists(check_point_path):
            os.makedirs(check_point_path)
        torch.save(model.state_dict(), full_path)

    data = model.conv1.weight.cpu().data.numpy()

    kernel_num = data.shape[0]

    fig, axes = plt.subplots(ncols=kernel_num, figsize=(2 * kernel_num, 2))

    for col in range(kernel_num):
        axes[col].imshow(data[col, 0, :, :], cmap=plt.cm.gray)
    plt.show()


if "__main__" == __name__:
    main()
