from pathlib import Path
import shutil


if __name__ == '__main__':
    ori_dir = Path("/nfs/DataArchive/tooth_segmentation/original/0_3")
    dst_dir = Path("/nfs/DataArchive/tooth_segmentation/sampled_img/2022-09-21")

    img_dir = dst_dir.joinpath('images')
    img_dir.mkdir(exist_ok=True)
    depth_dir = dst_dir.joinpath('depth')
    depth_dir.mkdir(exist_ok=True)
    ref_dir = dst_dir.joinpath('reference')
    ref_dir.mkdir(exist_ok=True)

    for data_path in ori_dir.iterdir():
        if data_path.suffix == '.txt':
            continue

        path_split = data_path.name.split('_')
        if path_split[0] == 'removed':
            seq_num = int(path_split[1])
            img_num = int(path_split[2])
            is_ref = True
        else:
            seq_num = int(path_split[0])
            img_num = int(path_split[1])
            is_ref = False

        if img_num % 10 != 0:
            continue

        print(data_path)

        if is_ref:
            ref_path_dst = ref_dir.joinpath(data_path.name)
            shutil.copy(data_path, ref_path_dst)
        else:
            new_img_name = f"{dst_dir.name}_{seq_num:03}_{img_num:04}_depthImage.bmp"

            img_path_dst = img_dir.joinpath(new_img_name)
            depth_name_src = data_path.stem + '.txt'
            depth_path_src = data_path.parent.joinpath(depth_name_src)
            depth_path_dst = depth_dir.joinpath(new_img_name.split('.')[0] + '.txt')
            shutil.copy(data_path, img_path_dst)
            shutil.copy(depth_path_src, depth_path_dst)
