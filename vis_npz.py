import numpy as np


def load_and_print_npz_info(file_path):
    data = np.load(file_path)

    # 遍历文件中的所有数组
    for key in data.keys():
        array = data[key]

        # 打印数组的名称和形状
        print(f"Array name: {key}")
        print(f"Array shape: {array.shape}")
        print(f"Array dtype: {array.dtype}")
        print()  # 空行，以便于区分不同的数组信息


file_path = "./data/BraTS2019-precessed/test/test_stack/images/BraTS19_2013_1_1_045.npz"
load_and_print_npz_info(file_path)
