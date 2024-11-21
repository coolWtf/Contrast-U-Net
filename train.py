import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import LoadDataset
from evalution_segmentaion import eval_semantic_segmentation
from model.unet import UNet
from model.CUnet import CUNet
from model.UNetPlusPlus import NestedUNet
from model.CUnetplus import CUnetplus
from model.CUnetDilated import CUNetDilated
from model.CUnetDouble import CUnetDouble
from model.res_unet_plus import ResUnetPlusPlus
import config as cfg
from pytorch_toolbelt import losses as L
import segmentation_models_pytorch as smp
from tensorboardX import SummaryWriter
import numpy as np
import torch
import random
import os
def set_seed(seed = 0):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return random_state

random_state = set_seed(1997)

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
num_class = cfg.DATASET[1]

Load_train = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
# Load_val = LoadDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

train_data = DataLoader(Load_train, batch_size=cfg.BATCH_SIZE, shuffle=True,)
# val_data = DataLoader(Load_val, batch_size=cfg.BATCH_SIZE, shuffle=True) 
tensorboard_dir = './Unet'
if not os.path.exists(tensorboard_dir): 
    os.makedirs(tensorboard_dir)
writer = SummaryWriter(tensorboard_dir)
# unet = NestedUNet(channel=3, classes=num_class).to(device)
unet = CUNet(3, num_class).to(device)
# unet = smp.UnetPlusPlus(in_channels=3, classes=num_class).to(device)
loss = L.JointLoss(L.FocalLoss(), L.LovaszLoss(), 1, 0.4)
criterion = loss.to(device)  # FocalLoss().to(device)
optimizer = optim.Adam(unet.parameters(), lr=1e-4)


def train(model):
    best = [0]
    net = model.train()
    # 训练轮次
    for epoch in range(cfg.EPOCH_NUMBER):
        print('Epoch is [{}/{}]'.format(epoch + 1, cfg.EPOCH_NUMBER))
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5

        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0
        # 训练批次
        for i, sample in enumerate(train_data):
            # 载入数据
            img_data = Variable(sample['img'].float().to(device))
            img_label = Variable(sample['label'].float().to(device))
            print(img_label.dtype)
            # 训练
            out = net(img_data)
            print(out.dtype)
            # out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 评估
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrix = eval_semantic_segmentation(pre_label, true_label)
            train_acc += eval_metrix['mean_class_accuracy']
            train_miou += eval_metrix['miou']
            train_class_acc += eval_metrix['class_accuracy']

            print('|batch[{}/{}]|batch_loss {: .8f}|'.format(i + 1, len(train_data), loss.item()))
        writer.add_scalar('loss',train_loss,epoch)
        metric_description = '|Train Acc|: {:.5f}|Train Mean IU|: {:.5f}\n|Train_class_acc|:{:}'.format(
            train_acc / len(train_data),
            train_miou / len(train_data),
            train_class_acc / len(train_data),
        )
        print(type(train_acc))
        print(train_acc)
        writer.add_scalar('acc',train_acc / len(train_data),epoch)
        writer.add_scalar('miou',train_miou / len(train_data),epoch)
        #writer.add_scalar('class_acc',train_class_acc / len(train_data) / len(train_data),epoch)
        print(metric_description)
        if max(best) <= train_miou / len(train_data):
            best.append(train_miou / len(train_data))
            t.save(net.state_dict(), './Results/weights/CU-Net_carpet_moudle_test.pth')


# def evaluate(model):
#     net = model.eval()
#     eval_loss = 0
#     eval_acc = 0
#     eval_miou = 0
#     eval_class_acc = 0
#
#     prec_time = datetime.now()
#     for j, sample in enumerate(val_data):
#         valImg = Variable(sample['img'].float().to(device))
#         valLabel = Variable(sample['label'].float().long().to(device))
#
#         out = net(valImg)
#         out = F.log_softmax(out, dim=1)
#         loss = criterion(out, valLabel)
#         eval_loss = loss.item() + eval_loss
#         pre_label = out.max(dim=1)[1].data.cpu().numpy()
#         pre_label = [i for i in pre_label]
#
#         true_label = valLabel.data.cpu().numpy()
#         true_label = [i for i in true_label]
#
#         eval_metrics = eval_semantic_segmentation(pre_label, true_label)
#         eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
#         eval_miou = eval_metrics['miou'] + eval_miou
#
#     cur_time = datetime.now()
#     h, remainder = divmod((cur_time - prec_time).seconds, 3600)
#     m, s = divmod(remainder, 60)
#     time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
#
#     val_str = ('|Valid Loss|: {:.5f} \n|Valid Acc|: {:.5f} \n|Valid Mean IU|: {:.5f} \n|Valid Class Acc|:{:}'.format(
#         eval_loss / len(train_data),
#         eval_acc / len(val_data),
#         eval_miou / len(val_data),
#         eval_class_acc / len(val_data)))
#     print(val_str)
#     print(time_str)


if __name__ == "__main__":
    train(unet)
    # evaluate(unet)
