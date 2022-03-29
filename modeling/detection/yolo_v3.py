# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/3/11 15:59

from __future__ import absolute_import


import os
import sys
sys.path.append(os.getcwd())

import base64
import io
# import json
import numpy as np
import random
import time
import tempfile
import torch
import torchvision
import zipfile

import PIL

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from terminaltables import AsciiTable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from typing import List

# backbone dependency
from dataloader.coco_yolo import YoloDataset
from dataloader.coco_yolo import fetch_from_train_set
from dataloader.coco_yolo import split_dataset

from modeling.backbone.darknet.model import DarkNet
from modeling.backbone.darknet.utils import ap_per_class
from modeling.backbone.darknet.utils import get_batch_statistics
from modeling.backbone.darknet.utils import is_predict_valid
from modeling.backbone.darknet.utils import pad_to_square
from modeling.backbone.darknet.utils import rescale_boxes
from modeling.backbone.darknet.utils import resize
from modeling.backbone.darknet.utils import weights_init_normal

from modeling.detection import ObjDetModel

from utils.logger import logger


class YoloV3(ObjDetModel):
    """
    implements a yolo
    """
    def __init__(self, model_def, pretrained_weights, **kwargs):
        super(YoloV3, self).__init__(**kwargs)

        logger.info("using device {}".format(self.device))

        # default is all 
        self.filter_classes = kwargs["filter_classes"] if "filter_classes" in kwargs else []
        logger.info("filter_classes is {}".format(self.filter_classes))

        # make sure checkpoint folder exists
        self.ckpt_path = kwargs["ckpt_path"] if "ckpt_path" in kwargs else r"./checkpoint/detection/yolo_v3/"
        os.makedirs(self.ckpt_path, exist_ok=True)

        # make sure results folder exists
        self.result_path = kwargs["result_path"] if "result_path" in kwargs else r"./result/detection/yolo_v3/"
        os.makedirs(self.result_path, exist_ok=True)

        self._build_model(model_def=model_def, pretrained_weights=pretrained_weights)

    def __collate_fn__(self, batch):
        return tuple(zip(*batch))

    def _build_model(self, **kwargs):
        """
        build model
        """
        self.model = DarkNet(config_path=kwargs["model_def"]).to(self.device)

        # pretrained weights
        if "pretrained_weights" in kwargs:
            if kwargs["pretrained_weights"].endswith(".pth"):
                if os.path.exists(kwargs["pretrained_weights"]):
                    self.model.load_state_dict(torch.load(kwargs["pretrained_weights"], map_location="cpu"))
                    logger.info("using pretrained_weights {}".format(kwargs["pretrained_weights"]))
                else:
                    self.model.apply(weights_init_normal)
                    logger.warning("pretrained_weights {} not exists, using default weights.".format(kwargs["pretrained_weights"]))
            else:
                if not os.path.exists(kwargs["pretrained_weights"]):
                    import wget
                    os.makedirs(os.path.dirname(kwargs["pretrained_weights"]), exist_ok=True)
                    wget.download(r"https://pjreddie.com/media/files/darknet53.conv.74", out=os.path.dirname(kwargs["pretrained_weights"]))
                self.model.load_darknet_weights(kwargs["pretrained_weights"])
                logger.info("using pretrained_weights {}".format(kwargs["pretrained_weights"]))
        else:
            self.model.apply(weights_init_normal)
            logger.info("using default weights.")

    @torch.no_grad()
    def _evaluate(self, data_loader, **kwargs):
        """
        sub function of evaluate
        """

        self.model.eval()

        conf_thresh = kwargs["conf_thresh"] if "conf_thresh" in kwargs else 0.4
        nms_thresh = kwargs["nms_thresh"] if "nms_thresh" in kwargs else 0.4

        batch_label = []
        sample_metrics = []

        import cv2
        
        from modeling.backbone.darknet.utils import non_max_suppression
        from modeling.backbone.darknet.utils import xywh2xyxy

        for batch_idx, (batch_name, batch_image, batch_target) in enumerate(data_loader):
            batch_label += batch_target[:, 1].tolist()

            batch_target[:, 2:] = xywh2xyxy(batch_target[:, 2:])
            batch_target[:, 2:] *= 416

            batch_output = self.model(batch_image.to(self.device))
            batch_output = non_max_suppression(batch_output, conf_thresh=conf_thresh, nms_thresh=conf_thresh)

            for image_name, image_data, predicted_label in zip(batch_name, batch_image, batch_output):
                tmp = image_data.cpu().detach().permute((1, 2, 0)).mul(255).clamp(0, 255).numpy()
                tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)

                if predicted_label is not None:
                    for rect_info in predicted_label:
                        coord = rect_info.cpu().numpy()
                        if is_predict_valid(coord, self.filter_classes, image_data.size(-1)):
                            cv2.rectangle(tmp, (coord[0], coord[1]), (coord[2], coord[3]), (0, 255, 0), 3)
                
                cv2.imwrite(os.path.join(self.result_path, os.path.basename(image_name)), tmp)

            sample_metrics += get_batch_statistics(batch_output, batch_target, iou_thresh=0.5)

        # return score, evaluate_res_str
        if 0 == len(sample_metrics):
            ap_class = np.array(list(set(batch_label)), dtype=np.int32)
            precision = recall = AP = f1 = np.array([0 for x in ap_class], dtype=np.float64)
        else:
            # Concatenate sample statistics
            true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
            precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, batch_label)

        return precision, recall, AP, f1, ap_class

    def _get_bounding_box(self, img, boxes, pred_cls,  rect_th=3, text_size=1, text_th=3):
        """
        draw the bounding box on img
        """
        
        tmp_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for i in range(len(boxes)):
            cv2.rectangle(tmp_img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 255, 0), rect_th)
            cv2.putText(tmp_img, pred_cls[i], (boxes[i][0], boxes[i][1]), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                        thickness=text_th)

        cv2.imwrite(os.path.join(self.result_path, "{:04d}.png".format(random.randint(0, 9999))), tmp_img)
        return tmp_img

    def _get_prediction(self, img):
        self.model.eval()
        
        img = torchvision.transforms.ToTensor()(img)
        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1], img.shape[2]))
        elif len(img.shape) == 3 and img.shape[0] == 1:
            img = img.expand((3, img.shape[1], img.shape[2]))

        ori_size = img.shape[-2:]

        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        img = torch.unsqueeze(resize(img, 416), 0)
        img = img.to(self.device)
        pred = self.model(img)
        pred = non_max_suppression(pred, conf_thresh=self._knobs.get("conf_thresh"), nms_thresh=self._knobs.get("nms_thresh"))
        pred_class = []
        pred_boxes = []
        if pred[0] is None:
            return None
        
        box_info = rescale_boxes(pred[0], 416, ori_size)
        num_box = box_info.size()[0]

        # get predicted info
        for rect_info in box_info:
            coord = rect_info.cpu().numpy()
            if is_predict_valid(coord, self.filter_classes, img.size(-1)):
                pred_class.append(self.filter_classes[np.int(coord[6])-1])
                pred_boxes.append((np.int(coord[0]), np.int(coord[1]), np.int(coord[2]), np.int(coord[3])))
        
        if len(pred_boxes) == 0:
            return None
        else:
            return pred_boxes, pred_class

    def _train_one_epoch(self, optimizer, data_loader, epoch_idx, **kwargs):
        # lr_scheduler = None
        # if epoch_idx == 0:
        #     warmup_factor = 1. / 1000
        #     warmup_iters = min(1000, len(data_loader) - 1)
        # 
        #     lr_scheduler = self._warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        logger.info("on epoch {}, begin to train".format(epoch_idx))

        self.model.train()

        metrics = [
            "grid_size",
            "loss",
            "x",
            "y",
            "w",
            "h",
            "conf",
            "cls",
            "cls_acc",
            "recall50",
            "recall75",
            "precision",
            "conf_obj",
            "conf_noobj",
        ]

        for batch_idx, (_, batch_image, batch_target) in enumerate(data_loader):
            batches_done = len(data_loader) * epoch_idx + batch_idx
            
            loss, batch_output = self.model(batch_image.to(self.device), batch_target.to(self.device))

            if not np.math.isfinite(loss):
                logger.warn("loss is {}, stop training".format(loss))
                return None
            
            loss.backward()

            if "gradient_accumulation" not in kwargs or batches_done % kwargs["gradient_accumulation"]:
                optimizer.step()
                optimizer.zero_grad()

            log_str = "\n---- [Epoch %d, Batch %d/%d] ----\n" % (epoch_idx, batch_idx, len(data_loader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(self.model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in self.model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # # Tensorboard logging
                # tensorboard_log = []
                # for j, yolo in enumerate(model.yolo_layers):
                #     for name, metric in yolo.metrics.items():
                #         if name != "grid_size":
                #             tensorboard_log += [(f"{name}_{j+1}", metric)]
                # tensorboard_log += [("loss", loss.item())]
                # summary_logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += "\nTotal loss {}".format(loss.item())
            logger.info(log_str)

            # if lr_scheduler is not None:
            #     lr_scheduler.step()

            self.model.seen += batch_image.size(0)
        
        return loss.item()

    def _warmup_lr_scheduler(self, optimizer, warmup_iters, warmup_factor):
        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

    def dump_parameter(self, epoch_idx, **kwargs):
        """
        dump parameters to local file
        """
        
        model_file_path = kwargs["model_file_path"] if "model_file_path" in kwargs else "./checkpoint/detection/yolo_v3"
        model_filename = os.path.join(model_file_path, time.strftime("%Y%m%d_%H%M_{:04d}.pth".format(epoch_idx), time.localtime(time.time())))
        logger.info("store checkpoint to {} after epoch: {}".format(model_filename, epoch_idx))
        torch.save(self.model, model_filename)

    def evaluate(self, dataset_path, **kwargs):
        # load 
        # dataset_zipfile = zipfile.ZipFile(dataset_path, "r")
        # evaluate_folder = tempfile.TemporaryDirectory()
        # dataset_zipfile.extractall(path=evaluate_folder.name)
        # root_path = evaluate_folder.name
        # print("root_path: {}".format(root_path))
        logger.info("root_path: {}".format(dataset_path))

        print("prepare dataset")
        if os.path.isdir(os.path.join(dataset_path, "image")):
            print("split train/val subsets...")
            logger.info("split train/val subsets...")
            split_dataset(dataset_path)
        elif os.path.isdir(os.path.join(dataset_path, "train")):
            if not os.path.exists(os.path.join(dataset_path, "val")):
                fetch_from_train_set(dataset_path)
                logger.info("fetch val from train")
        else:
            print("unsupported dataset format!")
            logger.info("unsupported dataset format!")
            return None

        image_val = os.path.join(dataset_path, "val", "image")
        annotation_val = os.path.join(dataset_path, "val", "annotation")

        dataset_valid = YoloDataset(
            image_val,
            annotation_val,
            is_single_json_file=False,
            filter_classes=self.filter_classes,
            is_train=False,
        )
        data_loader_valid = torch.utils.data.DataLoader(
            dataset_valid,
            batch_size=1,
            shuffle=False,
            collate_fn=dataset_valid.collate_fn
        )

        logger.info("dataset prepared")

        # perform an evaluate
        precision, recall, AP, f1, ap_class = self._evaluate(data_loader_valid, **kwargs)
        
        return np.mean(precision)

    def predict(self):
        """
        predict with trained model
        """
        pass

    def train(self, dataset_path, num_epoch, batch_size, **kwargs):
        # load 
        # dataset_zipfile = zipfile.ZipFile(dataset_path, "r")
        # train_folder = tempfile.TemporaryDirectory()
        # dataset_zipfile.extractall(path=train_folder.name)
        # root_path = train_folder.name
        # logger.info("root_path: {}".format(dataset_path))

        logger.info("prepare dataset")
        if os.path.isdir(os.path.join(dataset_path, "train")) and os.path.isdir(os.path.join(dataset_path, "val")):
            logger.info("train and validate set exist")      
        elif os.path.isdir(os.path.join(dataset_path, "train")):
            if not os.path.exists(os.path.join(dataset_path, "val")):
                logger.info("fetch val from train")
                fetch_from_train_set(dataset_path)
        elif os.path.isdir(os.path.join(dataset_path, "image")):
            logger.info("split train/val subsets...")
            split_dataset(dataset_path)    
        else:
            logger.info("unsupported dataset format!")
            return None

        image_train = os.path.join(dataset_path, "train", "image")
        image_val = os.path.join(dataset_path, "val", "image")
        annotation_train = os.path.join(dataset_path, "train", "annotation")
        annotation_val = os.path.join(dataset_path, "val", "annotation")

        # Get dataloader
        dataset_train = YoloDataset(
            image_train,
            annotation_train,
            is_single_json_file=False,
            filter_classes=self.filter_classes,
            is_train=True,
            augment=True,
            multiscale=True
        )
        # Get dataloader
        dataset_valid = YoloDataset(
            image_val,
            annotation_val,
            is_single_json_file=False,
            filter_classes=self.filter_classes,
            is_train=False,
            augment=False,
            multiscale=False
        )

        logger.info("Training the model YOLO using {}".format(self.device))

        # define training and validation data loaders
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, collate_fn=dataset_train.collate_fn
        )

        data_loader_valid = torch.utils.data.DataLoader(
            dataset_valid, batch_size=batch_size, shuffle=False, collate_fn=dataset_valid.collate_fn
        )

        # construct an optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.01,
        )
        
        torch.manual_seed(1)
        ckpt_thresh = num_epoch // 3 * 2

        for epoch_idx in range(num_epoch):
            # train for one epoch, printing every 10 iterations
            loss_value = self._train_one_epoch(optimizer, data_loader_train, epoch_idx, **kwargs)

            logger.info("loss is {}".format(loss_value))

            if loss_value is None:
                break

            # # update the learning rate
            # lr_scheduler.step()

            if "ckpt_interval" not in kwargs or num_epoch <= kwargs["ckpt_interval"] or epoch_idx > ckpt_thresh and 0 == epoch_idx % kwargs["ckpt_interval"]:
                self.dump_parameter(epoch_idx, model_file_path=self.ckpt_path)

                logger.info("evaluate after epoch: {}".format(epoch_idx))
                precision, recall, AP, f1, ap_class = self._evaluate(data_loader_valid, **kwargs)
            
                logger.info("Average Precisions:")
                for i, c in enumerate(ap_class):
                    logger.info("\t+ Class \"{}\" ({}) - AP: {:.5f}".format(c, dataset_valid.coco.cats[dataset_valid.label_to_cat[c]]["name"], AP[i]))
            
                logger.info("mAP: {:.9f}".format(AP.mean()))
        
        if num_epoch > kwargs["ckpt_interval"] and 1 !=epoch_idx % kwargs["ckpt_interval"]:
            self.dump_parameter(epoch_idx, model_file_path=self.ckpt_path)
        
            logger.info("evaluate after all epoch(s)")
            precision, recall, AP, f1, ap_class = self._evaluate(data_loader_valid, **kwargs)
        
            logger.info("Average Precisions:")
            for i, c in enumerate(ap_class):
                logger.info("\t+ Class \"{}\" ({}) - AP: {:.5f}".format(c, dataset_valid.coco.cats[dataset_valid.label_to_cat[c]]["name"], AP[i]))
        
            logger.info("mAP: {:.9f}".format(AP.mean()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_path",
        type=str,
        default="./dataset/",
    )
    parser.add_argument(
        "batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "num_epoch",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--filter_classes",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="./modeling/backbone/darknet/yolov3_tiny.cfg",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default="./init_weights/darknet/darknet53.conv.74",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoint/detection/yolo_v3/",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="./result/detection/yolo_v3/",
    )

    args = parser.parse_args()
    logger.debug(args)

    model = YoloV3(
        model_def=args.model_def, pretrained_weights=args.pretrained_weights, filter_classes=args.filter_classes,
        ckpt_path=args.ckpt_path, result_path=args.result_path,
    )

    # logger.info("train with nms_thresh:{} conf_thresh:{}".format(kwargs["nms_thresh"], kwargs["conf_thresh"]))
    model.train(
        args.dataset_path, args.num_epoch, args.batch_size,
        # nms_thresh=nms_thresh / 100, conf_thresh=conf_thresh / 100,
        ckpt_interval=args.ckpt_interval,
        gradient_accumulation=args.gradient_accumulation,
    )

    logger.info("{} epoch(s) trained".format(args.num_epoch))

    # logger.info("begin to evaluate")
    # model.evaluate()
    logger.info("all done")






