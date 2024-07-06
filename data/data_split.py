import os
import json
import shutil
import random

def split_dataset(org_folder, train_folder, test_folder, list_path, split_ratio=0.8, seed=42):
    random.seed(seed)

    file_list = os.listdir(org_folder)
    random.shuffle(file_list)

    # 划分文件列表
    split_index = int(len(file_list) * split_ratio)
    train_files = file_list[:split_index]
    test_files = file_list[split_index:]

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    if not os.path.exists(list_path):
        os.makedirs(list_path)

    # 复制文件到对应目标路径
    for filename in train_files:
        src_path = os.path.join(org_folder, filename)
        dst_path = os.path.join(train_folder, filename)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy(src_path, dst_path)

    for filename in test_files:
        src_path = os.path.join(org_folder, filename)
        dst_path = os.path.join(test_folder, filename)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy(src_path, dst_path)

    train_files.sort()
    test_files.sort()

    train_list_path = os.path.join(list_path, "train_list.json")
    test_list_path = os.path.join(list_path, "test_list.json")

    with open(train_list_path, 'w') as f:
        json.dump(train_files, f, indent=4)

    with open(test_list_path, 'w') as f:
        json.dump(test_files, f, indent=4)


if __name__ == '__main__':
    org_data_folder = './BraTS2019-100case'
    train_folder = './BraTS2019-precessed/train/train_nii'
    test_folder = './BraTS2019-precessed/test/test_nii'
    list_path = './BraTS2019-precessed/div_log'

    split_dataset(org_data_folder, train_folder, test_folder, list_path)

    print("Data split && copy are DONE!")
