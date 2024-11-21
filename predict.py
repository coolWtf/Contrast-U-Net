import cv2
import pandas as pd
import numpy as np
import torch
import torch as t
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader
from evalution_segmentaion import eval_semantic_segmentation
from PIL import Image
from dataset import LoadDataset
from model.unet import UNet
from model.CUnet import CUNet
from model.CUnetDouble import CUnetDouble
from model.CUnetDilated import CUNetDilated
from model.res_unet_plus import ResUnetPlusPlus
from model.UNetPlusPlus import NestedUNet
import segmentation_models_pytorch as smp
import albumentations as A
from sklearn.metrics import roc_auc_score
import config as cfg
import os

if __name__ == "__main__":
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    num_class = cfg.DATASET[1]

    Load_test = LoadDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size, statu="valid")
    # Load_test = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
    test_data = DataLoader(Load_test, batch_size=1, shuffle=False, num_workers=0)

    net = CUNet(3, num_class).to(device)
    # net = smp.UnetPlusPlus(in_channels=3, classes=num_class).to(device)
    # net = ResUnetPlusPlus(classes=num_class, channel=3).to(device)
    #net.load_state_dict(t.load("./Results/weights/our_moudle_test.pth"))
    net.load_state_dict(t.load("./Results/weights/CU-Net_carpet_moudle_test.pth"))
    net.eval()

    pd_label_color = pd.read_csv(cfg.class_dict_path, sep=',')
    name_value = pd_label_color['name'].values
    num_class = len(name_value)
    colormap = []
    for i in range(num_class):
        tmp = pd_label_color.iloc[i]
        color = [tmp['r'], tmp['g'], tmp['b']]
        colormap.append(color)

    cm = np.array(colormap).astype('uint8')

    dir = "./Results/test/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    train_acc = 0
    train_miou = 0
    train_class_acc = 0
    auroc = 0

    for i, sample in enumerate(test_data):
        valImg = sample['img'].float().to(device)
        valLabel = sample['label'].long().to(device)
        out = net(valImg)
        #测试auroc
        true_probabilty = out[0,1,:,:]-out[0,0,:,:]
        true_probabilty = torch.sigmoid(true_probabilty)
        #保存结果
        out = F.log_softmax(out, dim=1)
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label[pre_label==1]=255
        cv2.imwrite('E:/CAROTID/our_res/carpet/' + str(i).zfill(3) + '_77map.png', pre_label[0])
        #plt.imshow(pre_label, cmap='gray')

        #plt.savefig('E:/CAROTID/our_res/leather_fold' + str(i) + '_77map.png')  # 保存热力图为图片文件

        pre_true_pro = true_probabilty.data.cpu().numpy()

        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label[pre_label == 255] = 1
        true_label = valLabel.data.cpu().numpy()
        true_label_flat = true_label.reshape(-1)  # 将真实标签展平成一维数组
        pre_label_flat  = pre_true_pro.reshape(-1)
        auroc += roc_auc_score(true_label_flat,pre_label_flat)
        eval_metrix = eval_semantic_segmentation(pre_label, true_label)
        train_acc += eval_metrix['mean_class_accuracy']
        train_miou += eval_metrix['miou']
        train_class_acc += eval_metrix['class_accuracy']
        print("auroc",auroc)

    print("train_acc",train_acc/len(test_data))
    print("train_miou",train_miou/len(test_data))
    print("auroc",auroc/len(test_data))


    '''print(type(out_2))
        #re_1 = torch.sum(out_2, dim=0)
        re_1 = out_2[0]
        print(re_1)
        #sum_matrix = re_1.cpu().numpy().astype(np.float32)
        sum_matrix = re_1.cpu().detach().numpy()
        # 创建Normalize对象
        norm = Normalize(vmin=sum_matrix.min(), vmax=sum_matrix.max())

        # 缩放数据
        sum_matrix_norm = norm(sum_matrix)
        plt.imshow(sum_matrix_norm, cmap='Reds')  # 绘制热力图
        plt.colorbar()  # 添加颜色条
        plt.title('U-net')  # 设置标题
        plt.xlabel('X Label')  # 设置x轴标签
        plt.ylabel('Y Label')  # 设置y轴标签
        plt.savefig('E:/CAROTID/unet/Datasets/Carotid/U-net-heatmap/'+str(i)+'_first_heatmap.png')  # 保存热力图为图片文件
        plt.clf()'''
    '''out = F.log_softmax(out, dim=1)
        pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
        pre = cm[pre_label]
        pre1 = Image.fromarray(pre)

        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = valLabel.data.cpu().numpy()
        true_label = [i for i in true_label]
        eval_metrix = eval_semantic_segmentation(pre_label, true_label)
        train_acc += eval_metrix['mean_class_accuracy']
        train_miou += eval_metrix['miou']
        train_class_acc += eval_metrix['class_accuracy']

        pre1.save(dir + str(i+1) + '.png')
        print('Done')'''

    '''metric_description = '|Train Acc|: {:.5f}|Train Mean IU|: {:.5f}\n|Train_class_acc|:{:}'.format(
        train_acc / len(test_data),
        train_miou / len(test_data),
        train_class_acc / len(test_data),
    )
    print(metric_description)'''
