import os
import cv2
import torch
import json
import numpy as np
import albumentations as A

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2


def stack_dict_batched(batched_input):
    out_dict = {}  # 初始化一个空值字典
    for k, v in batched_input.items():
        if isinstance(v, list):  # 检查值 v 是否为列表类型。如果是列表类型，说明该数据不需要进行整理或堆叠，直接将其赋值给 out_dict
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])  # -1表示自动计算该维度的大小，而 *v.shape[2:] 表示保持剩余维度的大小不变。
    return out_dict

def train_transforms(img_size, ori_h, ori_w):
    transforms = []
    if ori_h < img_size and ori_w < img_size:
        transforms.append(
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    else:
        transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST))
    transforms.append(ToTensorV2(p=1.0))
    # ToTensorV2将图像0-255转成0-1，并且从numpy数组改成PyTorch张量，p=1是转换概率 即100%
    return A.Compose(transforms, p=1.0)


class MultiTestingDataset(Dataset):
    def __init__(self, data_path, image_size=256, mode='test', requires_name=True):
        self.image_size = image_size
        self.requires_name = requires_name

        dataset = json.load(open(os.path.join(data_path, f'label2image_{mode}.json'), "r"))
        self.mask_paths = list(dataset.keys())
        self.image_paths = [sample_info['image_path'] for sample_info in dataset.values()]

    def __getitem__(self, index):
        input_stack = {}
        masks_list = []

        imag_data = np.load(self.image_paths[index])
        image = imag_data['arr_0']  # npz info
        image = np.transpose(image, (1, 2, 0))
        mask_path = self.mask_paths[index]
        np_mask = cv2.imread(mask_path, 0)

        h, w = np_mask.shape
        transforms = train_transforms(self.image_size, h, w)

        augments = transforms(image=image, mask=np_mask)
        image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

        masks_list.append(mask_tensor)
        masks = torch.stack(masks_list, dim=0)
        input_stack["image"] = image_tensor.unsqueeze(0).float()
        input_stack["label"] = masks.unsqueeze(1).float()

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            input_stack["name"] = image_name
            return input_stack
        else:
            return input_stack

    def __len__(self):
        return len(self.mask_paths)


class MultiTrainingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True):
        self.image_size = image_size
        self.requires_name = requires_name

        dataset = json.load(open(os.path.join(data_dir, f'image2label_{mode}.json'), "r"))
        self.image_paths = list(dataset.keys())
        self.mask_paths = [sample_info['mask_path'] for sample_info in dataset.values()]

    def __getitem__(self, index):
        input_stack = {}
        masks_list = []

        imag_data = np.load(self.image_paths[index])
        image = imag_data['arr_0']  # npz info
        image = np.transpose(image, (1, 2, 0))
        mask_path = self.mask_paths[index]
        mask = cv2.imread(mask_path, 0)

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)

        augments = transforms(image=image, mask=mask)
        image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)
        masks_list.append(mask_tensor)
        masks = torch.stack(masks_list, dim=0)

        input_stack["image"] = image_tensor.unsqueeze(0).float()
        input_stack["label"] = masks.unsqueeze(1).float()

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            input_stack["name"] = image_name
            return input_stack
        else:
            return input_stack

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    "---MultiTrainingDataset---"
    train_dataset = MultiTrainingDataset("./data/BraTS2019-precessed/train/train_stack", image_size=256, mode='train',
                                         requires_name=True)
    print("Dataset:", len(train_dataset))
    train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=4, shuffle=False, num_workers=4)
    train_loader = tqdm(train_batch_sampler)
    for i, batched_image in enumerate(train_loader):
        if i == 2:
            break
        batched_image = stack_dict_batched(batched_image)
        image = batched_image["image"]
        masks = batched_image["label"]
        name = batched_image["name"]

        print("---image information:", image.shape)  # torch.Size([B, 4, 256, 256])
        print("---masks information:", masks.shape)  # torch.Size([B, 1, 256, 256])
        print("---the image name", name[0])

    "---MultiTestingDataset---"
    test_dataset = MultiTestingDataset("./data/BraTS2019-precessed/test/test_stack", image_size=256, mode='test',
                                       requires_name=True)
    print("Test Dataset: ", len(test_dataset))
    test_batch_sampler = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = tqdm(test_batch_sampler)
    for i, batched_image in enumerate(test_loader):
        if i == 2:
            break
        batched_image = stack_dict_batched(batched_image)
        image = batched_image["image"]  # torch.Size([B, 4, 256, 256])
        masks = batched_image["label"]  # torch.Size([B, 1, 256, 256])
        name = batched_image["name"]

        print("---image information:", image.shape)
        print("---masks information:", masks.shape)
        print("---the image name", name[0])
