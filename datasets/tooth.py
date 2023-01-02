# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os
from pathlib import Path
import json
import random

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

import torch
from .base_dataset import BaseDataset


class Tooth(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_classes=1,
                 multi_scale=False,
                 flip=True,
                 ignore_label=255,
                 base_size=384,
                 crop_size=(0, 0),
                 scale_factor=None,
                 mean=[0.485, 0.456, 0.406],
                 # mean=[0.261, 0.326, 0.624],
                 std=[0.229, 0.224, 0.225],
                 # std=[0.126, 0.155, 0.106],
                 bd_dilate_size=4):

        super(Tooth, self).__init__(ignore_label, base_size,
                                         crop_size, scale_factor, mean, std, )

        self.root = Path(root)
        self.json_path = self.root.joinpath(list_path)
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip

        # with open(self.json_path, 'r') as json_file:
        #     self.instances = json.load(json_file)
        self.instances = COCO(self.json_path)

        self.files = self.instances.imgs

        self.class_weights = torch.FloatTensor([1.0, 1.0]).cuda()

        self._set_group_flag()

    def convert_label(self, label):
        label[label == 0] = self.ignore_label
        label[label < self.ignore_label] = 0

        return label

    def __getitem__(self, index):
        index += 1
        item = self.instances.loadImgs(index)[0]
        img_id = item['id']
        img_name = item["file_name"]
        img_path = self.json_path.parents[1].joinpath('images', img_name)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img_size = (item['height'], item['width'])

        anns = self.instances.loadAnns(self.instances.getAnnIds(img_id, 3))
        # anns = self.instances.loadAnns(self.instances.getAnnIds(img_id, 1))
        label = np.zeros(img_size, dtype=np.uint8)
        for ann in anns:
            label += self.instances.annToMask(ann)
        # label = self.convert_label(label)

        image, label, edge = self.gen_sample(image, label, self.multi_scale, self.flip)

        return image.copy(), label.copy(), edge.copy(), np.array(img_size), img_name

    def gen_sample(self, image, label, multi_scale=True, is_flip=True, edge_pad=True, edge_size=4, city=True):

        edge = cv2.Canny(label, 0.1, 0.2)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        edge = (cv2.dilate(edge, kernel, iterations=1) > 50) * 1.0

        # if multi_scale:
        #     rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
        #     image, label, edge = self.multi_scale_aug(image, label, edge, rand_scale=rand_scale, rand_crop=False)

        h, w, _ = image.shape
        to_size = h if h > w else w
        image = self.pad_image(image, h, w, (to_size, to_size), 0)
        label = self.pad_image(label, h, w, (to_size, to_size), 0)
        edge = self.pad_image(edge, h, w, (to_size, to_size), 0)

        image = cv2.resize(image, (self.base_size, self.base_size), interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (self.base_size, self.base_size), interpolation=cv2.INTER_AREA)
        edge = cv2.resize(edge, (self.base_size, self.base_size), interpolation=cv2.INTER_AREA)

        # cv2.imshow('image', image)
        # cv2.imshow('label', label)
        # cv2.imshow('edge', edge)
        # cv2.waitKey()
        # exit(0)

        image = self.input_transform(image)
        label = self.label_transform(label)

        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            edge = edge[:, ::flip]

        return image, label, edge

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            i += 1
            img_info = self.files[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i-1] = 1

    def calc_mean_std(self):

        item = self.instances.loadImgs(1)[0]
        img_name = item["file_name"]
        img_path = self.json_path.parents[1].joinpath('images', img_name)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        h, w, _ = image.shape

        psum = np.array([0.0, 0.0, 0.0])
        psum_sq = np.array([0.0, 0.0, 0.0])

        for index in range(len(self)):
            index += 1
            item = self.instances.loadImgs(index)[0]
            img_name = item["file_name"]
            img_path = self.json_path.parents[1].joinpath('images', img_name)
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

            h, w, _ = image.shape
            to_size = h if h > w else w
            image = self.pad_image(image, h, w, (to_size, to_size), 0)

            image = image.astype(np.float32)
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))

            for i in range(3):
                psum[i] += image[i].sum()
                psum_sq[i] += (image[i] ** 2).sum()

        cnt = len(self) * h * w

        total_mean = psum / cnt
        total_var = (psum_sq / cnt) - (total_mean ** 2)
        total_std = np.sqrt(total_var)

        print(f"mean: {total_mean}")
        print(f"std: {total_std}")


class ToothDemo(Tooth):
    def __init__(self,
                 root,
                 num_classes=1,
                 multi_scale=False,
                 flip=True,
                 ignore_label=255,
                 base_size=384,
                 crop_size=(0, 0),
                 scale_factor=None,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        self.root = Path(root)
        self.files = list(self.root.iterdir())
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip

        self.mean = mean
        self.std = std
        self.ignore_label = ignore_label

        # self._set_group_flag()

    def __getitem__(self, index):
        img_path = self.files[index]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img_size = (image.shape[0], image.shape[1])
        image = self.gen_sample(image)
        return image, np.array(img_size), np.array(img_size), np.array(img_size), img_path.name

    def gen_sample(self, image, city=True):
        h, w, _ = image.shape
        to_size = h if h > w else w
        image = self.pad_image(image, h, w, (to_size, to_size), 0)
        image = self.input_transform(image)
        image = image.transpose((2, 0, 1))
        return image