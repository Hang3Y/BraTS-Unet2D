import os
import json

"---------train"
data_dir = './data/BraTS2019-precessed/train/train_stack'
# 读取文件夹中的文件列表
image_files = os.listdir(os.path.join(data_dir, 'images'))
# 创建一个空字典，用于存储每个样本的信息
dataset = {}

for image_file in image_files:
    if image_file.endswith('.npz'):
        sample_name = image_file.split('.')[0]
        image_path = os.path.join(data_dir, 'images', image_file)
        mask_path = os.path.join(data_dir, 'masks', sample_name + '_mask.png')
        dataset[image_path] = {
            'mask_path': mask_path,
        }

json_file_path = os.path.join(data_dir, 'image2label_train.json')
with open(json_file_path, 'w') as json_file:
    json.dump(dataset, json_file, indent=4)
print("Train_json Done!")

"---------test"
data_dir = './data/BraTS2019-precessed/test/test_stack'
image_files = os.listdir(os.path.join(data_dir, 'images'))
dataset = {}

for image_file in image_files:
    if image_file.endswith('.npz'):
        sample_name = image_file.split('.')[0]
        image_path = os.path.join(data_dir, 'images', image_file)
        mask_path = os.path.join(data_dir, 'masks', sample_name + '_mask.png')
        dataset[mask_path] = {
            'image_path': image_path,
        }

json_file_path = os.path.join(data_dir, 'label2image_test.json')
with open(json_file_path, 'w') as json_file:
    json.dump(dataset, json_file, indent=4)

print("Test_json Done!")
