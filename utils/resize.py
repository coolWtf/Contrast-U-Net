from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import albumentations as A
import matplotlib.pyplot as plt


def read_file(path):
    """从文件夹中读取数据"""
    files_list = os.listdir(path)
    file_path_list = [os.path.join(path, img) for img in files_list]
    return file_path_list


class ResizeDataset(Dataset):
    def __init__(self, img_path):
        self.images = read_file(img_path)
        self.transforms = A.Resize(256, 256)

    def __getitem__(self, item):
        image = self.images[item]
        image = Image.open(image)
        image = np.array(image)
        image = self.transforms(image=image)['image']
        return image

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    image_path = r"E:\代码\unet\Datasets\Carotid\predict\image_crop"
    file_path_list = read_file(image_path)
    for i in range(len(file_path_list)):
        file_path_list[i] = file_path_list[i].split("\\")[-1]
    save_path = r"E:\代码\unet\Datasets\Carotid\predict\image_resize/"
    if os.path.exists(save_path) is not True:
        os.mkdir(save_path)

    dataset = ResizeDataset(image_path)
    for i in range(len(dataset)):
        Image.fromarray(dataset[i]).save(save_path + file_path_list[i])
