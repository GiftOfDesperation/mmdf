import open3d
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from dataset.eval_dataset import EvalDataset
from dataset.robotcar_dataset import Standardize
from models.PointNetVlad import PointNetVlad
from models.mmdf import FusionModel
import torch.utils.data as data
import torch.optim as optim
import os
import pickle
import time


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('gpu可用')
        import torch.backends.cudnn as cudnn
    else:
        print('gpu不可用')
        while True:
            pass

    batch_size = 32
    train_dataset = EvalDataset(list_files=
                                r"/home/omnisky/PycharmProjects/yx/PIFusion/test/Test_00.txt",
                                standardize=Standardize())

    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4,
        drop_last=False)

    model = FusionModel(num_points=5000)
    model.load_state_dict(torch.load('../trained_models/kitti_model_100.pth'))
    model.cuda()
    model.eval()
    torch.set_grad_enabled(False)
    print('start test...')
    p = 0
    n = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    thresh = 23
    y_label = np.array([])
    y_distance = np.array([])
    for i, data in enumerate(train_dataloader):
        points1, points2, img1, img2, xy1, xy2, target = data
        points1, points2, target = Variable(points1), Variable(points2), Variable(target[:, 0])
        img1, img2, xy1, xy2 = Variable(img1), Variable(img2), Variable(xy1).float(), Variable(xy2).float()
        points1 = points1.transpose(2, 1)  
        points2 = points2.transpose(2, 1)
        if torch.cuda.is_available():
            img1, img2, xy1, xy2, points1, points2, target = \
                img1.cuda(), img2.cuda(), xy1.cuda(), xy2.cuda(), \
                points1.cuda(), points2.cuda(), target.cuda()

        time1 = time.time()
        feat1 = model(points1, img1, xy1)
        time2 = time.time()
        feat2 = model(points2, img2, xy2)

        # print("time cost:", (time2-time1)*1000, "ms")

        euclidean_distance = F.pairwise_distance(feat1, feat2, keepdim=False)
        distance = euclidean_distance.data
        issimilar = target.data
        # print(distance, issimilar)
        distance = np.asarray(distance.cpu())
        issimilar = np.asarray(issimilar.cpu())
        y_label = np.concatenate((y_label, issimilar), axis=0)
        for dis in distance:
            score = dis
            # print(score)
            # print(dis)
            y_distance = np.append(y_distance, score)

    with open('distance_00_mmdf.pickle', 'wb') as f:
        pickle.dump([y_distance, y_label], f)

