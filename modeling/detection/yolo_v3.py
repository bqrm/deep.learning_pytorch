# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/3/11 15:59

from __future__ import absolute_import


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
sys.path.append(os.getcwd())

import numpy as np
import torch

from terminaltables import AsciiTable

# backbone dependency
from dataloader.coco_yolo import YoloDataset
from dataloader.coco_yolo import fetch_from_train_set
from dataloader.coco_yolo import split_dataset

from modeling.backbone.darknet.model import DarkNet
from modeling.backbone.darknet.utils import ap_per_class
from modeling.backbone.darknet.utils import get_batch_statistics
from modeling.backbone.darknet.utils import is_predict_valid
from modeling.backbone.darknet.utils import weights_init_normal
from modeling.detection import ObjDetModel

from utils.logger import logger


class YoloV3(ObjDetModel):
    """
    implements a yolo
    """
    def __init__(self, model_def, **kwargs):
        super(YoloV3, self).__init__(**kwargs)

        logger.info("using device {}".format(self.device))

        self.model = DarkNet(config_path=model_def).to(self.device)

        # default is cat, only one class
        self.filter_classes = kwargs["filter_classes"] if "filter_classes" in kwargs else ["cat"]
        logger.info("filter_classes is {}".format(self.filter_classes))

        # make sure results folder exists
        from utils.root_path import root_path
        self.result_path = os.path.join(root_path, "result", "detection")
        os.makedirs(self.result_path, exist_ok=True)

        if "seed" in kwargs:
            self.seed = kwargs["seed"]
            torch.manual_seed(self.seed)
        
        self.gradient_accumulation = kwargs["gradient_accumulations"] if "gradient_accumulations" in kwargs else 2

    @torch.no_grad()
    def _evaluate(self, data_loader, **kwargs):
        """
        sub function of evaluate
        """

        self.model.eval()

        batch_label = []
        sample_metrics = []

        import cv2
        import tqdm
        from modeling.backbone.darknet.utils import non_max_suppression
        from modeling.backbone.darknet.utils import xywh2xyxy

        for batch_idx, (batch_name, batch_image, batch_target) in enumerate(tqdm.tqdm(data_loader)):
            batch_label += batch_target[:, 1].tolist()

            batch_target[:, 2:] = xywh2xyxy(batch_target[:, 2:])
            batch_target[:, 2:] *= 416

            batch_output = self.model(batch_image.to(self.device))
            batch_output = non_max_suppression(batch_output, conf_thresh=kwargs["conf_thresh"], nms_thresh=kwargs["nms_thresh"])

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

    def _train_one_epoch(self, optimizer, data_loader, epoch, **kwargs):
        logger.info("on epoch {}, begin to train".format(epoch))

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
            batches_done = len(data_loader) * epoch + batch_idx

            loss, batch_output = self.model(batch_image.to(self.device), batch_target.to(self.device))

            if not np.math.isfinite(loss):
                logger.warn("loss is {}, stop training".format(loss))
                return None
            
            loss.backward()

            if batches_done % self.gradient_accumulation:
                self.optimizer.step()
                self.optimizer.zero_grad()

            log_str = "\n---- [Epoch %d, Batch %d/%d] ----\n" % (epoch, batch_idx, len(data_loader))

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

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.model.seen += batch_image.size(0)
        
        return loss.item()

    def _warmup_lr_scheduler(self, warmup_iters, warmup_factor):
        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, f)

    def dump_parameters(self, pretrained_weights, **kwargs):
        """
        dump parameters to local file
        """
        pass

    def evaluate(self, dataset_path, batch_size, **kwargs):
        """
        evaluate in 
        """
        
        # perform an evaluate
        precision, recall, AP, f1, ap_class = self._evaluate(data_loader, conf_thresh=0.1, nms_thresh=0.1)

    def load_parameters(self, pretrained_weights):
        """
        load parameters from local file
        """
        if pretrained_weights.endswith(".pth") and os.path.exists(pretrained_weights):
            logger.info("using pretrained_weights {}".format(pretrained_weights))
            self.model.load_state_dict(torch.load(pretrained_weights, map_location="cpu"))
        else:
            if not os.path.exists(pretrained_weights):
                import wget
                os.makedirs(os.path.dirname(pretrained_weights), exist_ok=True)
                wget.download(r"https://pjreddie.com/media/files/darknet53.conv.74", out=os.path.dirname(pretrained_weights))
            logger.info("using pretrained_weights {}".format(pretrained_weights))
            self.model.load_darknet_weights(pretrained_weights)

    def predict(self):
        """
        predict with trained model
        """
        pass

    def train(self, dataset_path, num_epoch, batch_size, **kwargs):        
        root_path = r"/home/taomingyang/dataset/coco_car_cat/"

        # load 
        # dataset_zipfile = zipfile.ZipFile(dataset_path, "r")
        # train_folder = tempfile.TemporaryDirectory()
        # dataset_zipfile.extractall(path=train_folder.name)
        # root_path = train_folder.name
        logger.info("root_path: {}".format(root_path))

        logger.info("prepare dataset")
        if os.path.isdir(os.path.join(root_path, "image")):
            logger.info("split train/val subsets...")
            split_dataset(root_path)          
        elif os.path.isdir(os.path.join(root_path, "train")):
            if not os.path.exists(os.path.join(root_path, "val")):
                logger.info("fetch val from train")
                fetch_from_train_set(root_path)
        else:
            logger.info("unsupported dataset format!")
            return None

        image_train = os.path.join(root_path, "train", "image")
        image_val = os.path.join(root_path, "val", "image")
        annotation_train = os.path.join(root_path, "train", "annotation")
        annotation_val = os.path.join(root_path, "val", "annotation")

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
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=kwargs["lr"] if "lr" in kwargs else 0.01,
        )

        # pretrained weights
        if "pretrained_weights" in kwargs:
            self.load_parameters(kwargs["pretrained_weights"])
        else:
            logger.warning("no pretrained_weights, using default weights.")
            self.model.apply(weights_init_normal)

        # # move model to the right device
        # self.model.to(self.device)

        for epoch_idx in range(num_epoch):
            # train for one epoch
            loss_value = self._train_one_epoch(self.optimizer, data_loader_train, epoch_idx)

            logger.info("loss is {}".format(loss_value))

            if loss_value is None:
                break

            # update the learning rate
            # lr_scheduler.step()

            logger.info("begin to evaluate after epoch: {}".format(epoch_idx))
            precision, recall, AP, f1, ap_class = self._evaluate(data_loader_valid, conf_thresh=0.1, nms_thresh=0.1)
            
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
        default="coco_mini.zip",
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
        "--model_def",
        type=str,
        default="./modeling/backbon/darknet/yolov3.cfg",
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
        "--filter_classes",
        type=str,
        default="cat",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=0.4,
    )

    args = parser.parse_args()
    logger.debug(args)

    model = YoloV3(model_def=args.model_def, filter_classes=[j.strip() for j in args.filter_classes.split(',')])
    logger.info("YoloV3 created")

    logger.info("begin to train")
    model.train(
        args.dataset_path, args.num_epoch, args.batch_size,
        pretrained_weights=args.pretrained_weights, lr=args.lr,
        conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh
    )

    logger.info("{} epoch(s) trained".format(args.num_epoch))

    logger.info("begin to evaluate")
    # model.evaluate()
    logger.info("all done")





























