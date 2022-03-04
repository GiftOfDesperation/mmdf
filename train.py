import open3d
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from dataset.kitti_odm_dataset import KittiOdmDataset
from dataset.robotcar_dataset import RobotCarDataset
from dataset.robotcar_dataset import Standardize
from models.mmdf import FusionModel
import torch.utils.data as data
import torch.optim as optim
from tensorboardX import SummaryWriter
import os


def eval_one_batch(feat1, feat2, target, batch_size, thresh):
    correct = 0
    euclidean_distance = F.pairwise_distance(feat1, feat2, keepdim=False)
    distance = euclidean_distance.data
    issimilar = target.data

    return distance, issimilar


def contrastive_loss(input1, input2, label, margin=50.0):
    l = label.float()
    euclidean_distance = F.pairwise_distance(input1, input2, keepdim=False)
    loss_contrastive = (l) * torch.pow(euclidean_distance, 2.0) \
                       + \
                       (1 - l) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)

    loss_contrastive_total = torch.mean(loss_contrastive)
    loss_pos = [value for index, value in enumerate(loss_contrastive) if l[index]]
    loss_neg = [value for index, value in enumerate(loss_contrastive) if not l[index]]
    return loss_contrastive_total, loss_pos, loss_neg


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('gpu_available')
        import torch.backends.cudnn as cudnn
    else:
        print('gpu_not_available')
        while True:
            pass

    batch_size = 8
    # train_dataset = RobotCarDataset(list_files=
    #                                 [r"/home/omnisky/PycharmProjects/yx/PIFusion/oxford_robotcar/RobotCar_Train.txt"],
    #                                 standardize=Standardize())
    train_dataset = KittiOdmDataset(list_files=[r"/home/omnisky/PycharmProjects/yx/PIFusion/TrainCmpDataPath_all.txt"],
                                    standardize=Standardize())
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4,
        drop_last=True)

    log_dir = './log'
    train_writer = SummaryWriter(os.path.join(log_dir, 'train'))
    model = FusionModel(num_points=5000)
    optimizer_cmp = optim.SGD(
        [{'params': model.parameters(), 'lr': 5e-6}],
        momentum=0.5)

    model.cuda()
    model.train()
    print('Start training...')
    total_epoch = 100
    for epoch in range(1, total_epoch + 1):
        for i, data in enumerate(train_dataloader):
            img1, img2, points1, points2, target, xy1, xy2 = data
            points1, points2, target = Variable(points1), Variable(points2), Variable(target[:, 0])
            img1, img2, xy1, xy2 = Variable(img1), Variable(img2), Variable(xy1).float(), Variable(xy2).float()
            points1 = points1.transpose(2, 1)  
            points2 = points2.transpose(2, 1)
            if torch.cuda.is_available():
                img1, img2, xy1, xy2, points1, points2, target = \
                    img1.cuda(), img2.cuda(), xy1.cuda(), xy2.cuda(), \
                    points1.cuda(), points2.cuda(), target.cuda()

            feat1 = model(points1, img1, xy1)
            feat2 = model(points2, img2, xy2)

            optimizer_cmp.zero_grad()
            loss, loss_pos, loss_neg = contrastive_loss(feat1, feat2, target)
            if i % 100 == 0:
                # train_writer.add_scalar("Loss", loss.cpu().item(), epoch * i + i)
                print(eval_one_batch(feat1, feat2, target, batch_size, thresh=15))
                # train_writer.add_scalar("Accuracy", one_batch_acc, (epoch+1) * i * batch_size)
            loss.backward()
            optimizer_cmp.step()

            if i % 20 == 0:
                print('[epoch %d: %5.2f%%] cmp train loss: %f' % (epoch, (i+1)*batch_size/len(train_dataset)*100, loss.item()))

        if epoch % 5 == 0:
            torch.save(model.state_dict(), './trained_models_mmdf_pro/kitti_model_%d_mmdf_pro_s1.pth' % epoch)
            pass
