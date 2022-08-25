from pathlib import Path
import json


def main():
    # json_path = Path("/nfs/DataArchive/tooth_segmentation/demo/train/annotations/instances_default.json")
    json_path = Path("/nfs/DataArchive/tooth_segmentation/demo/test/annotations/instances_default.json")
    save_path = json_path.parent.joinpath("instances_default_1cls.json")
    remain_cat_name = 'tooth'

    with open(json_path, 'r') as json_file:
        instances = json.load(json_file)

    new_instances = {'licenses': instances['licenses'], 'info': instances['info'], 'images': instances['images']}
    # print(instances['licenses'])
    # print(instances['info'])
    # print(instances['categories'])
    # print(instances['images'][0])
    # print(instances['annotations'][0])

    new_cat = []
    for cat in instances['categories']:
        if cat['name'] == remain_cat_name:
            new_cat.append(cat)
            break
    new_instances['categories'] = new_cat

    new_anns = []
    for ann in instances['annotations']:
        if ann['category_id'] == new_cat[0]['id']:
            new_anns.append(ann)
    new_instances['annotations'] = new_anns

    with open(save_path, 'w') as json_file_w:
        json.dump(new_instances, json_file_w)


if __name__ == '__main__':
    main()