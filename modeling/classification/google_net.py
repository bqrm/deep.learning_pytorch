# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/2/18 16:22

# https://www.cnblogs.com/ys99/p/10871530.html

from __future__ import absolute_import, print_function

import os
import torch
import torch.nn as nn

from tqdm import tqdm

from dataloader import make_data_loader
from modeling import conv_relu
from modeling.utils.evaluator import Evaluator
from modeling.utils.model_saver import ModelSaver
from utils.dict_object import DictToObject


class BottleNeck(torch.nn.Module):
    """
    GoogleNet 中的 inception
    https://blog.csdn.net/a1103688841/article/details/89290680
    """
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(BottleNeck, self).__init__()

        self.branch_1x1 = conv_relu(in_channel, out1_1, 1)
        self.branch_3x3 = torch.nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)
        )
        self.branch_5x5 = torch.nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )
        self.branch_pool = torch.nn.Sequential(
            torch.nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)
        )

    def forward(self, x):
        f1 = self.branch_1x1(x)
        f2 = self.branch_3x3(x)
        f3 = self.branch_5x5(x)
        f4 = self.branch_pool(x)

        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output


class GoogleNet(nn.Module):
    def __init__(self, in_channel, num_class, is_training, verbose=False):
        super(GoogleNet, self).__init__()

        self.is_training = is_training
        self.verbose = verbose

        self.block1 = nn.Sequential(
            conv_relu(in_channel, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, 2, padding=1),
        )

        self.block2 = nn.Sequential(
            conv_relu(64, 64, 1),
            conv_relu(64, 192, 3, padding=1),
            nn.MaxPool2d(3, 2, padding=1),
        )

        self.block3 = nn.Sequential(
            BottleNeck(192, 64, 96, 128, 16, 32, 32),
            BottleNeck(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2, padding=1),
        )

        self.block4 = nn.Sequential(
            BottleNeck(480, 192, 96, 208, 16, 48, 64),
            BottleNeck(512, 160, 112, 224, 24, 64, 64),
            BottleNeck(512, 128, 128, 256, 24, 64, 64),
            BottleNeck(512, 112, 144, 288, 32, 64, 64),
            BottleNeck(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2, padding=1),
        )

        self.block5 = nn.Sequential(
            BottleNeck(832, 256, 160, 320, 32, 128, 128),
            BottleNeck(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(7, padding=3)
        )

        self.drop_out = nn.Dropout(p=0.4)

        self.classifier = nn.Linear(1024, num_class)

    def forward(self, x):
        if self.verbose:
            print("input: {}".format(x.shape))

        x = self.block1(x)
        if self.verbose:
            print("block 1 output: {}".format(x.shape))

        x = self.block2(x)
        if self.verbose:
            print("block 2 output: {}".format(x.shape))

        x = self.block3(x)
        if self.verbose:
            print("block 3 output: {}".format(x.shape))

        x = self.block4(x)
        if self.verbose:
            print("block 4 output: {}".format(x.shape))

        x = self.block5(x)
        if self.verbose:
            print("block 5 output: {}".format(x.shape))

        if self.is_training:
            x = self.drop_out(x)

        x = x.view(x.shape[0], -1)
        output = self.classifier(x)
        return output


def test(model, criterion, data_loader, evaluator, use_cuda=False):
    test_loss = 0

    evaluator.reset()

    model.eval()

    iter_bar = tqdm(data_loader)
    for batch_index, batch_sample in enumerate(iter_bar, 1):
        sample, target = batch_sample["image"], batch_sample["label"]

        if use_cuda:
            sample = sample.cuda()
            target = target.cuda()

        output = model(sample)
        test_loss += criterion(output, target.long()).data
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        iter_bar.set_description("testing loss: %.3f" % (test_loss / batch_index))
        evaluator.add_batch(target.cpu().numpy(), pred.cpu().numpy())

    acc = evaluator.pixel_accuracy()
    acc_class = evaluator.pixel_accuracy_class()
    miou = evaluator.mean_intersection_over_union()
    fwiou = evaluator.frequency_weighted_intersection_over_union()

    print("\tAcc:{},\n\tAcc_class:{},\n\tmIoU:{},\n\tfwIoU: {}".format(acc, acc_class, miou, fwiou))


def train(model, epoch, criterion, optimizer, data_loader, use_cuda=False):
    total_loss = 0
    model.train()

    iter_bar = tqdm(data_loader)
    for batch_index, batch_sample in enumerate(iter_bar, 1):
        sample, target = batch_sample["image"], batch_sample["label"]

        if use_cuda:
            sample = sample.cuda()
            target = target.cuda()

        output = model(sample)
        optimizer.zero_grad()

        loss = criterion(output, target.long())
        loss.backward()

        optimizer.step()

        total_loss += loss
        iter_bar.set_description("training epoch %d loss: %.3f" % (epoch, total_loss / batch_index))


def main():
    from utils.model_saver import ModelSaver

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device {}".format(device))

    args = DictToObject
    args.batch_size = 32
    args.crop_size = 224
    args.dataset = "image_net"
    args.num_epoch = 300
    args.root_dir = "D:/Projects/DataSets/mini_image_net"
    args.use_sbd = False
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader, test_loader, valid_loader, num_class = make_data_loader(args, **kwargs)
    
    evaluator = Evaluator(num_class)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    check_point_name = "google_net"
    check_point_path = "../../checkpoint/"

    saver = ModelSaver(check_point_name, check_point_path)

    if saver.num_checkpoint():
        print("loading model")

        model = GoogleNet(3, num_class, False).to(device)
        state_dict = saver.get_checkpoint()
        if state_dict is not None:
            model.load_state_dict(state_dict["model"])
            test(model, criterion, test_loader, evaluator, torch.cuda.is_available())
    else:
        print("training model")

        if not os.path.exists(check_point_path):
            os.makedirs(check_point_path)

        model = GoogleNet(3, num_class, True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

        for epoch_idx in range(args.num_epoch):
            train(model, epoch_idx, criterion, optimizer, train_loader, torch.cuda.is_available())

            if 0 == epoch_idx % 1:
                test(model, criterion, test_loader, evaluator, torch.cuda.is_available())
                saver.save_checkpoint({
                    'epoch': epoch_idx + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                })

        saver.save_checkpoint({
            'epoch': args.num_epoch,
            'model_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        })


if "__main__" == __name__:
    main()



























