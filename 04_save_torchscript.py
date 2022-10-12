import argparse

import torch

import models
from configs import config
from configs import update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():

    args = parse_args()
    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    model = models.pidnet.get_seg_model(config, imgnet_pretrained=False)

    model_scripted = torch.jit.script(model)
    model_scripted.save('model_scripted_test.pt')


if __name__ == '__main__':
    main()