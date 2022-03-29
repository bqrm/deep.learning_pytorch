# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:      python 3.6
# Description:  
# Author:       bqrmtao@qq.com
# date:         2021/01/15 17:27

from __future__ import absolute_import

import itertools
import json
import numpy as np
import os
import random
import time
import torch
import torchvision

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from collections import defaultdict

from modeling.backbone.darknet.utils import pad_to_square
from modeling.backbone.darknet.utils import resize


def fetch_from_train_set(root_path, split_ratio=0.8):
    image_train_folder = os.path.join(root_path, "train", "image")
    image_val_folder = os.path.join(root_path, "val", "image")
    annotation_train_folder = os.path.join(root_path, "train", "annotation")
    annotation_val_folder = os.path.join(root_path, "val", "annotation")

    os.makedirs(image_val_folder, exist_ok=True)
    os.makedirs(annotation_val_folder, exist_ok=True)

    list_image = list(sorted(os.listdir(image_train_folder)))
    list_annotation = list(sorted(os.listdir(annotation_train_folder)))

    union_list = []
    for image_name in list_image:
        base_name, _ = os.path.splitext(image_name)

        if base_name + ".json" in list_annotation:
            union_list.append(image_name)

    disordered_index = np.random.permutation(range(len(union_list)))
    val_list = disordered_index[np.int(len(union_list) * split_ratio):]
    import shutil

    for image_idx in val_list:
        image_name = union_list[image_idx]
        annotation_name = os.path.splitext(image_name)[0] + ".json"

        shutil.move(os.path.join(image_train_folder, image_name), os.path.join(image_val_folder, image_name))
        shutil.move(os.path.join(annotation_train_folder, annotation_name), os.path.join(annotation_val_folder, annotation_name))


def split_dataset(root_path, split_ratio=0.8):
    image_path = os.path.join(root_path, "image")
    annotation_path = os.path.join(root_path, "annotation")

    image_train_folder = os.path.join(root_path, "train", "image")
    image_val_folder = os.path.join(root_path, "val", "image")
    annotation_train_folder = os.path.join(root_path, "train", "annotation")
    annotation_val_folder = os.path.join(root_path, "val", "annotation")

    os.makedirs(image_train_folder, exist_ok=True)
    os.makedirs(image_val_folder, exist_ok=True)
    os.makedirs(annotation_train_folder, exist_ok=True)
    os.makedirs(annotation_val_folder, exist_ok=True)

    list_image = list(sorted(os.listdir(image_path)))
    list_annotation = list(sorted(os.listdir(annotation_path)))

    union_list = []
    for image_name in list_image:
        base_name, _ = os.path.splitext(image_name)

        if base_name + ".json" in list_annotation:
            union_list.append(image_name)
    
    disordered_index = np.random.permutation(range(len(union_list)))
    train_list = disordered_index[:np.int(len(union_list) * split_ratio)]
    val_list = disordered_index[np.int(len(union_list) * split_ratio):]

    import shutil
    for image_idx, image_name in enumerate(union_list):
        annotation_name = os.path.splitext(image_name)[0] + ".json"

        if image_idx in train_list:
            shutil.copy(os.path.join(image_path, image_name), os.path.join(image_train_folder, image_name))
            shutil.copy(os.path.join(annotation_path, annotation_name), os.path.join(annotation_train_folder, annotation_name))
        else:
            shutil.copy(os.path.join(image_path, image_name), os.path.join(image_val_folder, image_name))
            shutil.copy(os.path.join(annotation_path, annotation_name), os.path.join(annotation_val_folder, annotation_name))


class YoloCoco(object):
    def __init__(self, annotation_path=None, is_single_json_file=False):
        """
        dataset for YOLO, according with coco
        @ annotation_path: annotation path, filename if a single json, folder path is multiple jsons
        """
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.img_to_ann, self.cat_to_img = defaultdict(list), defaultdict(list)

        if annotation_path is not None:
            print("loading annotations into memory")
            tic = time.time()

            if is_single_json_file:
                # load annotations from single json
                with open(annotation_path, 'r') as f:
                    dataset = json.load(f)
            else:
                # load annotations from json files
                dataset = self.load_scattered_json(annotation_path)

            assert type(dataset)==dict, "annotation file format {} not supported".format(type(dataset))
            print("Done (t={:0.2f}s)".format(time.time()- tic))
            self.dataset = dataset
        else:
            raise ValueError("annotation_path should not be None")

        self.create_index()

    def _is_array_like(self, obj):
        return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

    def load_scattered_json(self, annotation_path):
        """
        merge annotation into a dataset, in accordancy with pycocotool
        """
        list_annotation = list(sorted(os.listdir(annotation_path)))

        dataset = {
            "images": list(),
            "annotations": list(),
            "categories": list(),
        }

        dict_category = dict()
        dict_image = dict()
        last_category_id = 0
        last_annotation_id = 0
        last_image_id = 0

        # for all json files
        for annotation_idx, annotation_filename in enumerate(list_annotation):
            with open(os.path.join(annotation_path, annotation_filename), 'r') as f:
                json_info = json.load(f)
            
            # process image info
            image_id = int(json_info["imageID"])
            if image_id not in dict_image:
                dict_image[image_id] = last_image_id
                last_image_id += 1

                image_info = {
                    "file_name": json_info["imagePath"],
                    "height": json_info["imageHeight"],
                    "width": json_info["imageWidth"],
                    "id": image_id,
                }

                dataset["images"].append(image_info)

            # process bounding box information
            for bounding_box_info in json_info["shapes"]:
                if bounding_box_info["label"] not in dict_category:
                    dict_category[bounding_box_info["label"]] = last_category_id

                    category_info = {
                        "id": last_category_id,
                        "name":bounding_box_info["label"],
                    }

                    dataset["categories"].append(category_info)
                    last_category_id += 1

                annotation_info = {
                    "image_id": image_id,
                    "bbox": list(np.array(np.concatenate((bounding_box_info["points"][0], bounding_box_info["points"][1]), axis=0), dtype=np.int)),
                    "category_id": dict_category[bounding_box_info["label"]],
                    "id": last_annotation_id,
                }
                last_annotation_id += 1

                dataset["annotations"].append(annotation_info)
        return dataset

    def create_index(self):
        print("creating index")
        anns, cats, imgs = dict(), dict(), dict()
        img_to_ann, cat_to_img = defaultdict(list), defaultdict(list)

        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                img_to_ann[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                imgs[img["id"]] = img

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                cat_to_img[ann["category_id"]].append(ann["image_id"])

        print("index created")

        # create class member
        self.anns = anns
        self.cats = cats
        self.imgs = imgs
        self.cat_to_img = cat_to_img
        self.img_to_ann = img_to_ann

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))

    def get_ann_id(self, img_id=[], cat_id=[], area_rng=[], is_crowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param: img_id  (int array)    get anns for given imgs
        :param: cat_id  (int array)    get anns for given cats
        :param: area_rng (float array) get anns for given area range (e.g. [0 inf])
        :param: is_crowd (boolean)     get anns for given crowd label (False or True)
        :return: ids (int array)       integer array of ann ids
        """
        img_id = img_id if self._is_array_like(img_id) else [img_id]
        cat_id = cat_id if self._is_array_like(cat_id) else [cat_id]

        if len(img_id) == len(cat_id) == len(area_rng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(img_id) == 0:
                lists = [self.img_to_ann[imgId] for imgId in img_id if imgId in self.img_to_ann]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(cat_id)  == 0 else [ann for ann in anns if ann['category_id'] in cat_id]
            anns = anns if len(area_rng) == 0 else [ann for ann in anns if ann['area'] > area_rng[0] and ann['area'] < area_rng[1]]
        if not is_crowd is None:
            ids = [ann['id'] for ann in anns if ann['is_crowd'] == is_crowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def get_cat_id(self, cat_nms=[], sup_nms=[], cat_id=[]):
        """
        filtering parameters. default skips that filter.
        :param: cat_nms (str array)  : get cats for given cat names
        :param: sup_nms (str array)  : get cats for given supercategory names
        :param: cat_id (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        cat_nms = cat_nms if self._is_array_like(cat_nms) else [cat_nms]
        sup_nms = sup_nms if self._is_array_like(sup_nms) else [sup_nms]
        cat_id = cat_id if self._is_array_like(cat_id) else [cat_id]

        if len(cat_nms) == len(sup_nms) == len(cat_id) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(cat_nms) == 0 else [cat for cat in cats if cat['name']          in cat_nms]
            cats = cats if len(sup_nms) == 0 else [cat for cat in cats if cat['supercategory'] in sup_nms]
            cats = cats if len(cat_id) == 0 else [cat for cat in cats if cat['id']            in cat_id]
        ids = [cat['id'] for cat in cats]
        return ids

    def get_img_id(self, img_id=[], cat_id=[]):
        """
        Get img ids that satisfy given filter conditions.
        :param: img_id (int array)  get imgs for given ids
        :param: cat_id (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        img_id = img_id if self._is_array_like(img_id) else [img_id]
        cat_id = cat_id if self._is_array_like(cat_id) else [cat_id]

        if len(img_id) == len(cat_id) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(img_id)
            for i, cat_id in enumerate(cat_id):
                if i == 0 and len(ids) == 0:
                    ids = set(self.cat_to_img[cat_id])
                else:
                    # original &=, but should be |=
                    ids &= set(self.cat_to_img[cat_id])
        return list(ids)

    def load_ann(self, ids=[]):
        """
        Load anns with the specified ids.
        :param: ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if self._is_array_like(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def load_cat(self, ids=[]):
        """
        Load cats with the specified ids.
        :param: ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if self._is_array_like(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def load_imgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param: ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if self._is_array_like(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]

    def load_numpy_annotation(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param:  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print('Converting ndarray to lists...')
        assert(type(data) == np.ndarray)
        print(data.shape)
        assert(data.shape[1] == 7)
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i,N))
            ann += [{
                'image_id'  : int(data[i, 0]),
                'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
                'score' : data[i, 5],
                'category_id': int(data[i, 6]),
                }]
        return ann



class YoloDataset(torch.utils.data.Dataset):
    """
    dataset of yolo
    """
    def __init__(self, image_path, annotation_path, is_single_json_file, filter_classes, is_train, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        self.root_path = image_path
        self.imgs = list(sorted(os.listdir(image_path)))
        self.annotation_path = annotation_path
        self.coco = YoloCoco(self.annotation_path, is_single_json_file=is_single_json_file)
        # eg: filter_classes: ['person', 'dog']
        self.cat_ids = self.coco.get_cat_id(cat_nms=filter_classes)
        self.ids = self.coco.get_img_id(cat_id=self.cat_ids)
        
        self.cat_to_label = {v: key+1 for key, v in enumerate(self.cat_ids)}
        self.label_to_cat = {key+1: v for key, v in enumerate(self.cat_ids)}

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment if is_train else False
        self.multiscale = multiscale if is_train else False
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        img_id = self.ids[index % len(self.ids)]
        ann_id = self.coco.get_ann_id(img_id=img_id)

        img_path = os.path.join(self.root_path, self.coco.load_imgs(img_id)[0]["file_name"])

        # Extract image as PyTorch tensor
        img = torchvision.transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------
        coco_annotation = self.coco.load_ann(ann_id)

        tmp_label = []
        box_info = []
        for ann in coco_annotation:
            if ann["category_id"] not in self.cat_ids:
                continue
            boxes = torch.zeros((1, 6), dtype=torch.float32)
            x1 = round(max(ann['bbox'][0], 0))
            y1 = round(max(ann['bbox'][1], 0))
            x2 = round(min(ann['bbox'][2], w - 1))
            y2 = round(min(ann['bbox'][3], h - 1))
            # x2 = round(min(x1 + ann['bbox'][2], w - 1))
            # y2 = round(min(y1 + ann['bbox'][3], h - 1))

            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            box_info.append(((x1, y1), (x2, y2)))

            # print(x1, x2, y1, y2, padded_h, padded_w)
            # Returns (x, y, w, h)
            boxes[0, 2] = (x2 + x1) / 2 / padded_w
            boxes[0, 3] = (y2 + y1) / 2 / padded_h
            boxes[0, 4] = (x2 - x1) / padded_w
            boxes[0, 5] = (y2 - y1) / padded_h
            boxes[0, 1] = self.cat_to_label[ann["category_id"]]
            tmp_label.append(boxes)
        
        # self.get_bounding_box(img, os.path.basename(img_path), box_info)

        # targets from list to tensor
        targets = torch.cat(tmp_label, dim=0)

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = self.horisontal_flip(img, targets)

        return img_path, img, targets

    def __len__(self):
        return len(self.ids)

    def _extract_zip(self, dataset_path, annotation_path):
        dataset_zipfile = zipfile.ZipFile(dataset_path, 'r')
        annotation_zipfile = zipfile.ZipFile(annotation_path, 'r')

        # create temp dir
        self.root_path = tempfile.TemporaryDirectory()

        # extract images and annotations
        dataset_zipfile.extractall(path=self.root_path.name)
        annotation_zipfile.extractall(path=self.root_path.name)
        imgs = list(sorted(os.listdir(os.path.join(self.root_path.name, self.img_folder_name))))
        annotation_file = os.path.join(self.root_path.name, "annotations", self.annotation_file_name)

        return imgs, annotation_file

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    # def get_bounding_box(self, img, basename, boxes, rect_th=3):
    #     """
    #     draw the bounding box on img
    #     """
    #     tmp = img.squeeze().detach().permute((1, 2, 0)).mul(255).clamp(0, 255).numpy()
    #     tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
    # 
    #     for rect_info in boxes:
    #         cv2.rectangle(tmp, rect_info[0], rect_info[1], (0, 255, 0), rect_th)
    # 
    #     cv2.imwrite('./rectangle_images/{}'.format(basename), tmp)

    def horisontal_flip(self, images, targets):
        images = torch.flip(images, [-1])
        targets[:, 2] = 1 - targets[:, 2]
        return images, targets



def main():
    pass


if "__main__" == __name__:
    main()
