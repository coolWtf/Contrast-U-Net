
import numpy as np
import os
from PIL import Image

__all__ = ['SegmentationMetric']

"""
confusionMetric
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵n*n，初始值全0

    # 像素准确率PA，预测正确的像素/总像素
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + FN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def recall(self):
        TP = np.diag(self.confusionMatrix)
        FN = self.confusionMatrix.sum() - np.diag(self.confusionMatrix).sum() + np.diag(self.confusionMatrix) - self.confusionMatrix.sum(axis=1)
        re = TP / (TP + FN)
        return re

    # 类别像素准确率CPA，返回n*1的值，代表每一类，包括背景
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    # 类别平均像素准确率MPA，对每一类的像素准确率求平均
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    # MIoU
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    # 根据标签和预测图片返回其混淆矩阵
    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    # 更新混淆矩阵
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape  # 确认标签和预测值图片大小相等
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    # 清空混淆矩阵
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

def evaluate1(pre_path, label_path):
    acc_list = []
    macc_list = []
    mIoU_list = []
    fwIoU_list = []

    pre_imgs = os.listdir(pre_path)
    lab_imgs = os.listdir(label_path)

    for i, p in enumerate(pre_imgs):
        imgPredict = Image.open(pre_path + p).convert('L')
        imgPredict = ((np.array(imgPredict))/255).astype('int64')
        # print(imgPredict)
        # imgPredict = imgPredict[:,:,0]
        imgLabel = Image.open(label_path + lab_imgs[i]).convert('L')
        imgLabel =((np.array(imgLabel))/255).astype('int64')
        # imgLabel = imgLabel[:,:,0]
        metric = SegmentationMetric(2)  # 表示分类个数，包括背景
        metric.addBatch(imgPredict, imgLabel)
        acc = metric.pixelAccuracy()
        macc = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        fwIoU = metric.Frequency_Weighted_Intersection_over_Union()
        acc_list.append(acc)
        macc_list.append(macc)
        mIoU_list.append(mIoU)
        fwIoU_list.append(fwIoU)
        # print('{}: acc={}, macc={}, mIoU={}, fwIoU={}'.format(p, acc, macc, mIoU, fwIoU))

    return acc_list, macc_list, mIoU_list, fwIoU_list


def evaluate2(pre_path, label_path):
    pre_imgs = os.listdir(pre_path)
    lab_imgs = os.listdir(label_path)

    metric = SegmentationMetric(2)  # 表示分类个数，包括背景
    for i, p in enumerate(pre_imgs):
        imgPredict = Image.open(pre_path + p).convert('L')
        imgPredict = ((np.array(imgPredict))/255).astype('int64')
        imgLabel = Image.open(label_path + lab_imgs[i]).convert('L')
        imgLabel = ((np.array(imgLabel))/255).astype('int64')
        metric.addBatch(imgPredict, imgLabel)

    return metric

import pandas as pd
if __name__ == '__main__':
    # pre_path = '../input/New_G_A_Dice/train_image/'
    pre_path = r"E:\代码\unet\Results\test/"
    label_path = r'E:\代码\unet\Datasets\Carotid\predict\label_resize/'
    # label_path = r"../input/label/"
    # #  预测图像文件夹
    # PredictPath = r"../input/image_predict/"
    # 计算测试集每张图片的各种评价指标，最后求平均
    acc_list, macc_list, mIoU_list, fwIoU_list = evaluate1(pre_path, label_path)
    for i in range(len(acc_list)):
        print('测试图像{}: acc={:.5f}%, macc={:.3f}%, mIoU={:.3f}%, fwIoU={:.3f}%'.format(i+1,acc_list[i], macc_list[i],mIoU_list[i],fwIoU_list[i]))
    test = pd.DataFrame({'像素准确率PA': acc_list, '类别平均像素准确率(mPA)': macc_list,'mIoU': mIoU_list,'fwIoU': fwIoU_list})
    # test.to_csv('../input/number_evaluate/generation_train_pre_256.csv', index=False, encoding='utf8')
    # 加总测试机每张图片的混淆矩阵，对最终形成的这一个矩阵计算各种评价指标
    metric = evaluate2(pre_path, label_path)
    acc = metric.pixelAccuracy()
    macc = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    fwIoU = metric.Frequency_Weighted_Intersection_over_Union()

    print('总的：final2: acc={:.5f}%, macc={:.5f}%, mIoU={:.5f}%, fwIoU={:.5f}% '
          .format(acc * 100, macc * 100, mIoU * 100, fwIoU * 100))