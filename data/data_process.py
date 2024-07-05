import os
import cv2
import warnings

import numpy as np
import skimage.io as io
import SimpleITK as sitk

from tqdm import tqdm
from scipy.ndimage import zoom


warnings.filterwarnings("ignore")  # 忽略警告信息

"""
BraTs2019
0 其他背景， Other
1 肿瘤核心，necrotic tumor core (NCR)
2 瘤周水肿，edematous (ED)
4 增强肿瘤，enhancing tumor (ET)
"""
"---BraTs2019---"
flair_name = "_flair.nii.gz"
t1_name = "_t1.nii.gz"
t1ce_name = "_t1ce.nii.gz"
t2_name = "_t2.nii.gz"
mask_name = "_seg.nii.gz"

"---You should notice the dir and modify"
dir_root = "./BraTS2019-precessed/train/train_nii"
out_root = "./BraTS2019-precessed/train/train_stack"
# dir_root = "./BraTS2019-precessed/test/test_nii"
# out_root = "./BraTS2019-precessed/test/test_stack"

outputall_path = os.path.join(out_root, "images")
outputallmasks_path = os.path.join(out_root, "masks")

if not os.path.exists(outputall_path):
    os.makedirs(outputall_path)
if not os.path.exists(outputallmasks_path):
    os.makedirs(outputallmasks_path)


def file_name_path(file_dir):
    """
    get root path, sub_dirs, all_sub_files
    :param file_dir:
    :return: dir or file
    """
    path_list = []  # 存储所有文件路径的列表，获得的是路径下的子文件夹
    name_list = []  # 存储所有子目录名称的列表，到某个具体的文件（注意不是文件夹）
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            name_list = sorted(dirs)
        for f in files:
            path = os.path.join(root, f)
            path_list.append(path)
        path_list = sorted(path_list)
    return name_list, path_list


train_list, train_path_list = file_name_path(dir_root)
print("train_list:", len(train_list))
all_list = train_list


def normalize(slice):
    """
    对输入的切片进行归一化。
    :param slice: 输入的切片数据，通常是一个 NumPy 数组
    :return: 归一化后的切片数据，以 NumPy 数组形式返回
    """
    b = np.percentile(slice, 99)
    t = np.percentile(slice, 1)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        return tmp

def move(mask, croph):
    """
    在给定的 3D 切片数据中找到需要移动的距离，以尽量减少裁剪后的切片中包含的掩膜。
    :param mask: 3D 切片的掩膜数据，通常是一个 NumPy 数组
    :param croph: 需要裁剪的高度
    :return: 移动的距离
    """
    probe = 0
    height, width = mask[0].shape
    for probe in range(height//2-(croph//2)):
        bottom = height//2 + (croph//2) + probe
        if np.max(mask[:, bottom, :]) == 0:
            break
    if probe == 0:
        for probe in range(height//2-(croph//2)):
            up = height//2 - (croph//2) - probe
            if np.max(mask[:, up, :]) == 0 or np.max(mask[:, up+croph, :]) == 1:
                probe = 0-probe
                break
#     print("probe: ",probe)
    return probe

def crop_ceter(img, croph, cropw, move_value=0):
    """
    在给定的 3D 切片数据中裁剪中心区域。

    :param img: 3D 切片的数据，通常是一个 NumPy 数组
    :param croph: 需要裁剪的高度
    :param cropw: 需要裁剪的宽度
    :param move_value: 移动的距离，默认为 0
    :return: 裁剪后的 3D 切片数据，以 NumPy 数组形式返回
    """
    # for n_slice in range(img.shape[0]):
    height, width = img[0].shape
    starth = height//2-(croph//2) + move_value
    startw = width//2-(cropw//2)
    return img[:, starth:starth+croph, startw:startw+cropw]

def np_scaled(np_np):
    """np_np:输入的数组，需要进行归一化到[0,255]的那个数组"""
    min_value = np.min(np_np)
    max_value = np.max(np_np)
    return (np_np - min_value) / (max_value - min_value) * 255


def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def denormalize_image(image, original_min, original_max):
    return np.round(image * (original_max - original_min) + original_min).astype(np.uint8)


for subsetindex in tqdm(range(len(all_list)), desc="Processing images"):
    # if subsetindex == 1:
    #     break

    # 找到不同模态对应的具体路径
    flair_image = dir_root + '/' + all_list[subsetindex] + '/' + all_list[subsetindex] + flair_name
    t1_image = dir_root + '/' + all_list[subsetindex] + '/' + all_list[subsetindex] + t1_name
    t2_image = dir_root + '/' + all_list[subsetindex] + '/' + all_list[subsetindex] + t2_name
    t1ce_image = dir_root + '/' + all_list[subsetindex] + '/' + all_list[subsetindex] + t1ce_name
    mask_image = dir_root + '/' + all_list[subsetindex] + '/' + all_list[subsetindex] + mask_name

    # 使用 sitk.ReadImage 函数读取 FLAIR、T1、T1ce、T2 和掩膜图像文件。
    flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
    t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
    t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
    t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
    mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)

    # 使用 sitk.GetArrayFromImage 函数将 SimpleITK 图像对象转换为 NumPy 数组
    flair_array = sitk.GetArrayFromImage(flair_src)
    t1_array = sitk.GetArrayFromImage(t1_src)
    t1ce_array = sitk.GetArrayFromImage(t1ce_src)
    t2_array = sitk.GetArrayFromImage(t2_src)
    mask_array = sitk.GetArrayFromImage(mask)

    # 3. normalization
    flair_array_nor = normalize(flair_array)
    t1_array_nor = normalize(t1_array)
    t1ce_array_nor = normalize(t1ce_array)
    t2_array_nor = normalize(t2_array)

    # 4. cropping
    move_value = move(mask_array, 240)

    flair_crop = crop_ceter(flair_array_nor, 240, 240, move_value)
    t1_crop = crop_ceter(t1_array_nor, 240, 240, move_value)
    t1ce_crop = crop_ceter(t1ce_array_nor, 240, 240, move_value)
    t2_crop = crop_ceter(t2_array_nor, 240, 240, move_value)
    mask_crop = crop_ceter(mask_array, 240, 240, move_value)

    for n_slice in range(mask_crop.shape[0]):
        mask_np = mask_crop[n_slice, :, :]
        ET_label = np.empty((240, 240), np.uint8)
        TC_label = np.empty((240, 240), np.uint8)
        WT_label = np.empty((240, 240), np.uint8)

        if np.max(mask_np) == 0:
            # 当掩膜中所有像素值都为零时，跳出当前循环，执行下一次循环。
            continue

        mask_np_all = mask_np.copy()

        "--肿瘤坏死部分1:3, 瘤周水肿2:1, 增强肿瘤4:2"
        all_test_gt = np.zeros((240, 240), dtype=np.uint8)
        all_test_gt[mask_np_all == 4] = 2
        all_test_gt[mask_np_all == 2] = 1
        all_test_gt[mask_np_all == 1] = 3

        # 调整图像大小到 256x256
        all_test_gt = cv2.resize(all_test_gt, (256, 256), interpolation=cv2.INTER_NEAREST)
        # flair
        flair_np = flair_crop[n_slice, :, :]
        flair_np = flair_np.astype(np.float32)
        flair_np = np_scaled(flair_np)

        # t2
        t2_np = t2_crop[n_slice, :, :]
        t2_np = t2_np.astype(np.float32)
        t2_np = np_scaled(t2_np)

        # t1ce
        t1ce_np = t1ce_crop[n_slice, :, :]
        t1ce_np = t1ce_np.astype(np.float32)
        t1ce_np = np_scaled(t1ce_np)

        # t1
        t1_np = t1_crop[n_slice, :, :]
        t1_np = t1_np.astype(np.float32)
        t1_np = np_scaled(t1_np)

        # 获取原始最小值和最大值
        original_min = np.min([flair_np, t2_np, t1ce_np, t1_np])
        original_max = np.max([flair_np, t2_np, t1_np, t1_np])
        # print(original_min, original_max)  # 0.0--255.0
        # 归一化
        flair_np = normalize_image(flair_np)
        t2_np = normalize_image(t2_np)
        t1ce_np = normalize_image(t1ce_np)
        t1_np = normalize_image(t1_np)

        "----合并通道----"
        stacked4_img = np.stack([flair_np, t1_np, t2_np, t1ce_np], axis=0)
        stacked2_img = np.stack([flair_np, t1ce_np], axis=0)

        # 缩放因子
        zoom_factor = (1, 256 / 240, 256 / 240)  # 只在空间维度上缩放
        # 使用scipy的zoom函数进行缩放
        resized4_img = zoom(stacked4_img, zoom=zoom_factor, order=1)  # order=1表示双线性插值
        resized2_img = zoom(stacked2_img, zoom=zoom_factor, order=1)  # order=1表示双线性插值

        # 反归一化
        resized4_img = denormalize_image(resized4_img, original_min, original_max)
        resized2_img = denormalize_image(resized2_img, original_min, original_max)

        slice_num = n_slice + 1
        if len(str(slice_num)) == 1:
            new_slice_num = '00' + str(slice_num)
        elif len(str(slice_num)) == 2:
            new_slice_num = '0' + str(slice_num)
        else:
            new_slice_num = str(slice_num)

        io.imsave(outputallmasks_path + "/" + (all_list[subsetindex]) + "_" + str(new_slice_num) + "_mask.png", all_test_gt)
        np.savez_compressed((outputall_path + "/" + (all_list[subsetindex]) + "_" + str(new_slice_num)), resized4_img)

print("Done!")
