# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2019/12/26

from __future__ import absolute_import, print_function

import os
import torch

from tqdm import tqdm

from dataloader import make_data_loader
from modeling.backbone import build_backbone
from utils.dict_object import DictToObject
from utils.evaluator import Evaluator


class FCN(torch.nn.Module):
    def __init__(self, layer_config, num_class, is_training):
        super(FCN, self).__init__()

        if layer_config in ["32s", "16s", "8s", "s"]:
            self.layer_config = layer_config
        else:
            raise ValueError("layer_config should be one of [\"32s\", \"16s\", \"8s\", \"s\"]")

        self.num_class = num_class
        self.is_training = is_training

        kwargs = {
            "num_class": num_class,
            "with_bn": False,
            "is_training": is_training,
            "with_fc": False,
            "init_weights": True,
            "verbose": False,
        }
        self.vgg_model = build_backbone("vgg", **kwargs)
        self._make_transpose_conv()

    def _make_transpose_conv(self):
        self.relu = torch.nn.ReLU(inplace=True)
        self.deconv_1 = torch.nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv_2 = torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv_3 = torch.nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv_4 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv_5 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_1 = torch.nn.BatchNorm2d(512)
        self.bn_2 = torch.nn.BatchNorm2d(256)
        self.bn_3 = torch.nn.BatchNorm2d(128)
        self.bn_4 = torch.nn.BatchNorm2d(64)
        self.bn_5 = torch.nn.BatchNorm2d(32)
        self.classifier = torch.nn.Conv2d(32, self.num_class, kernel_size=1)

    def forward(self, x):
        output = self.vgg_model(x)

        x5 = output['x5']
        score = self.bn_1(self.relu(self.deconv_1(x5)))

        if self.layer_config in ["16s", "8s", "s"]:
            score += output['x4']

        score = self.bn_2(self.relu(self.deconv_2(score)))

        if self.layer_config in ["8s", "s"]:
            score += output['x3']

        score = self.bn_3(self.relu(self.deconv_3(score)))

        if "s" == self.layer_config:
            score += output['x2']

        score = self.bn_4(self.relu(self.deconv_4(score)))

        if "s" == self.layer_config:
            score += output['x1']

        score = self.bn_5(self.relu(self.deconv_5(score)))
        score = self.classifier(score)

        return score


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device {}".format(device))

    args = DictToObject
    args.batch_size = 4
    args.crop_size = 512
    args.dataset = "pascal"
    args.num_epoch = 300
    args.root_dir = "D:/Projects/DataSets/VOCdevkit/VOC2012"
    args.use_sbd = False
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader, test_loader, _, num_class = make_data_loader(args, **kwargs)
    
    evaluator = Evaluator(num_class)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)

    check_point_name = "fcn.t7"
    check_point_path = "../../checkpoint/"
    full_path = os.path.join(check_point_path, check_point_name)
    if os.path.exists(full_path):
        print("loading model")

        model = FCN("s", num_class, False).to(device)
        model.load_state_dict(torch.load(full_path, map_location=lambda storage, loc: storage))
        test(model, criterion, test_loader, evaluator, torch.cuda.is_available())
    else:
        print("training model")

        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))

        model = FCN("s", num_class, True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

        for epoch_idx in range(args.num_epoch):
            train(model, epoch_idx, criterion, optimizer, train_loader, torch.cuda.is_available())

            if 0 == epoch_idx % 10:
                test(model, criterion, test_loader, evaluator, torch.cuda.is_available())
                torch.save(model.state_dict(), full_path)

        torch.save(model.state_dict(), full_path)


if "__main__" == __name__:
    main()

































