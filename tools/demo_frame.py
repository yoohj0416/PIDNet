import argparse
import os
import pprint

import logging
import timeit
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
from datasets.tooth import ToothDemo
from configs import config
from configs import update_config
from utils.function import testval, test
from utils.utils import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('--data_root',
                        help='data root',
                        default="/nfs/DataArchive/tooth_segmentation_demo_frame",
                        type=str)
    parser.add_argument('--save_root',
                        help='save root',
                        default="./demo_frame",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = models.pidnet.get_seg_model(config, imgnet_pretrained=True)

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir, 'best.pt')

    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = model.cuda()

    dataset = ToothDemo(
        root=args.data_root,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE)

    testloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    test(config, dataset, testloader, model, sv_dir=args.save_root)


if __name__ == '__main__':
    main()