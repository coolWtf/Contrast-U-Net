import pandas as pd
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
import config as cfg
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class LabelProcessor:   # 1.处理标签文件中colormap的数据
    """对标签图像的编码"""

    def __init__(self, file_path):

        self.colormap = self.read_color_map(file_path)

        self.cm2lbl = self.encode_label_pix(self.colormap)

    # 静态方法装饰器， 可以理解为定义在类中的普通函数，可以用self.<name>方式调用
    # 在静态方法内部不可以示例属性和实列对象，即不可以调用self.相关的内容
    # 使用静态方法的原因之一是程序设计的需要（简洁代码，封装功能等）
    @staticmethod
    def read_color_map(file_path):  # data process and load.ipynb: 处理标签文件中colormap的数据
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []
        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)
        return colormap

    @staticmethod
    def encode_label_pix(colormap):     # data process and load.ipynb: 标签编码，返回哈希表
        cm2lbl = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):

        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')


class LoadDataset(Dataset):
    def __init__(self, file_path=[], crop_size=None, statu="train"):
        """para:
            file_path(list): 数据和标签路径,列表元素第一个为图片路径，第二个为标签路径
        """
        # 1 正确读入图片和标签路径
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径，图片路径在前")
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        # 2 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)
        # 3 初始化数据处理函数设置
        self.crop_size = crop_size
        self.statu = statu

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        # 从文件名中读取数据（图片和标签都是png格式的图像数据）
        img = Image.open(img).convert('RGB')
        label = Image.open(label).convert('RGB')

        # img, label = self.center_crop(img, label, self.crop_size)
        if self.statu == "train":
            trans = self.transforms_tatu()["train"]
        else:
            trans = self.transforms_tatu()["valid"]
        img, label = self.img_transform(img, label, trans)
        # print('处理后的图片和标签大小：',img.shape, label.shape)
        sample = {'img': img, 'label': label}

        return sample

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    def center_crop(self, data, label, crop_size):
        """裁剪输入的图片和标签大小"""
        "注释掉 不裁剪"
        #data = ff.center_crop(data, crop_size)
        #label = ff.center_crop(label, crop_size)
        return data, label

    def transforms_tatu(self):
        data_transforms = {
            "train": A.Compose([
                A.CenterCrop(self.crop_size[0], self.crop_size[1]),
                A.Resize(256, 256),
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
                # A.OneOf([
                #     A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                #     # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                #     A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
                # ], p=0.25),
                # A.CoarseDropout(max_holes=8, max_height=256 // 20, max_width=256 // 20,
                #                 min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
            ], p=1.0),
            "valid": A.Compose([
                A.CenterCrop(self.crop_size[0], self.crop_size[1]),
                A.Resize(256, 256),
            ], p=1.0)
        }
        return data_transforms

    def img_transform(self, img, label, trans):
        """对图片和标签做一些数值处理"""
        img = np.array(img)  # 以免不是np格式的数据
        # img = Image.fromarray(img.astype('uint8'))
        label = np.array(label)  # 以免不是np格式的数据
        # label = Image.fromarray(label.astype('uint8'))
        # transform_img = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ]
        # )
        data = trans(image=img, mask=label)
        img = data['image']
        label = data['mask']
        img = np.transpose(img, (2, 0, 1))
        # label = np.transpose(label, (2, 0, 1))
        label = label_processor.encode_label_img(label)
        label = t.from_numpy(label)

        return img, label


label_processor = LabelProcessor(cfg.class_dict_path)


def plot_batch(imgs, msks, size=3):
    plt.figure(figsize=(5*5, 5))
    for idx in range(size):
        plt.subplot(1, 5, idx+1)
        img = imgs[idx,].permute((1, 2, 0)).numpy()*255.0
        img = img.astype('uint8')
        msk = msks[idx,].permute((1, 2, 0)).numpy()*255.0
        show_img(img, msk)
    plt.tight_layout()
    plt.show()


def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     img = clahe.apply(img)
    #     plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='bone')

    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                   [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)
    plt.axis('off')


if __name__ == "__main__":
    train = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
    from torch.utils.data import DataLoader
    train_data = DataLoader(train, batch_size=5, shuffle=True)
    data = next(iter(train_data))
    imgs = data['img']
    msks = data['label']
    imgs.size(), msks.size()

    plot_batch(imgs, msks, size=5)