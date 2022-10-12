from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import shutil
import copy


def reset_id_img_anns(images, anns):
    id_num = 1
    ann_num = 1
    new_images = []
    new_anns = []
    for image in images:
        cur_num = image['id']
        new_image = copy.deepcopy(image)
        new_image['id'] = id_num
        new_images.append(new_image)
        for ann in anns:
            if ann['image_id'] == cur_num:
                new_ann = copy.deepcopy(ann)
                new_ann['id'] = ann_num
                new_ann['image_id'] = id_num
                new_anns.append(new_ann)
                ann_num += 1
        id_num += 1

    return new_images, new_anns


def main():
    ori_dir = Path('/nfs/DataArchive/tooth_segmentation/released/2022-10-04')
    ori_ann_path = ori_dir.joinpath('annotations', 'instances_default.json')
    save_dir = Path('/nfs/DataArchive/tooth_segmentation/released/2022-10-04_released')

    with open(ori_ann_path, 'r') as ori_ann_json:
        ori_ann = json.load(ori_ann_json)

    print(f"ori ann type: {type(ori_ann)}")
    for key in ori_ann:
        print(key)
    print(f"ori ann info: {ori_ann['info']}")
    print(f"ori ann images length: {len(ori_ann['images'])}")
    # print(f"ori ann images: {ori_ann['images'][0]}")
    print(f"ori ann annotations length: {len(ori_ann['annotations'])}")
    # print(f"ori ann annotations: {ori_ann['annotations'][0]}")

    train_images, test_images = train_test_split(ori_ann['images'], test_size=0.2)
    print(f"train images length: {len(train_images)}")
    print(f"test images length: {len(test_images)}")

    train_anns, test_anns = [], []
    for image_dict in train_images:
        for ann_dict in ori_ann['annotations']:
            if ann_dict['image_id'] == image_dict['id']:
                train_anns.append(ann_dict)

    for image_dict in test_images:
        for ann_dict in ori_ann['annotations']:
            if ann_dict['image_id'] == image_dict['id']:
                test_anns.append(ann_dict)

    print(f"train anns length: {len(train_anns)}")
    print(f"test anns length: {len(test_anns)}")

    train_images, train_anns = reset_id_img_anns(train_images, train_anns)
    test_images, test_anns = reset_id_img_anns(test_images, test_anns)

    train_dict = {'licenses': ori_ann['licenses'], 'info': ori_ann['info'], 'categories': ori_ann['categories'],
                  'images': train_images, 'annotations': train_anns}
    test_dict = {'licenses': ori_ann['licenses'], 'info': ori_ann['info'], 'categories': ori_ann['categories'],
                 'images': test_images, 'annotations': test_anns}

    train_dir = save_dir.joinpath('train')
    train_img_dir = train_dir.joinpath('images')
    train_img_dir.mkdir(exist_ok=True, parents=True)
    train_ann_dir = train_dir.joinpath('annotations')
    train_ann_dir.mkdir(exist_ok=True, parents=True)

    for img_dict in train_images:
        ori_img_path = ori_dir.joinpath('images', img_dict['file_name'])
        shutil.copy(ori_img_path, train_img_dir)
    train_ann_file_path = train_ann_dir.joinpath('instances_default.json')
    with open(train_ann_file_path, 'w') as json_file:
        json.dump(train_dict, json_file)

    test_dir = save_dir.joinpath('test')
    test_img_dir = test_dir.joinpath('images')
    test_img_dir.mkdir(exist_ok=True, parents=True)
    test_ann_dir = test_dir.joinpath('annotations')
    test_ann_dir.mkdir(exist_ok=True, parents=True)

    for img_dict in test_images:
        ori_img_path = ori_dir.joinpath('images', img_dict['file_name'])
        shutil.copy(ori_img_path, test_img_dir)
    test_ann_file_path = test_ann_dir.joinpath('instances_default.json')
    with open(test_ann_file_path, 'w') as json_file:
        json.dump(test_dict, json_file)


if __name__ == '__main__':
    main()
