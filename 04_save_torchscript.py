import argparse

import torch
import torchvision.models.segmentation

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

    model = models.pidnet_for_script.get_seg_model(config, imgnet_pretrained=False)
    model.eval()

    example = torch.rand(6, 3, 384, 384)

    # out = model(example)
    # print(len(out))
    # print(out[1].size())
    # print(out.size())

    model_state_file = config.TEST.MODEL_FILE

    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model_scripted = torch.jit.script(model)
    model_scripted.save('scripted_pidnet_model_2022-10-14.pt')


if __name__ == '__main__':
    main()