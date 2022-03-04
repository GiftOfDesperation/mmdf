"""
MMDF deep fusion version
"""
import open3d
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from config import cfg
from models.pointnet import STN3d
from models.pointnet import STNkd
import random
from dataset.kitti_odm_dataset import KittiOdmDataset
from models.NetVLAD import NetVLADLoupe


def conv3x3(in_channels, out_channels, stride=1):
    "3x3 convoltion with padding, do not change the HxW"
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ImgBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ImgBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print('input', x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print('output', out.shape)
        return out


class PtsMlpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PtsMlpBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class FusionConv(nn.Module):
    def __init__(self, in_channels):
        super(FusionConv, self).__init__()
        self.conv1 = nn.Conv1d(2 * in_channels, in_channels, 1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        # directly sigmoid
        self.fc1 = nn.Linear(in_channels, 1)
        self.fc2 = nn.Linear(in_channels, 1)

        # self.pool = nn.MaxPool1d(num_points)

    def forward(self, pts_features, img_features):
        pts_features_BND = pts_features.transpose(1, 2)
        img_features_BND = img_features.transpose(1, 2)
        pts_w = self.fc1(pts_features_BND)
        img_w = self.fc2(img_features_BND)
        add_w = torch.add(pts_w, img_w)
        weight_map = torch.sigmoid(add_w)
        # print(weight_map.shape)
        weight_map = weight_map.transpose(1, 2)
        img_features = weight_map * img_features

        fusion_features = torch.cat((pts_features, img_features), dim=1)
        # print(fusion_features.shape)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))
        return fusion_features

class FinalFusion(nn.Module):
    def __init__(self, in_channels):
        super(FinalFusion, self).__init__()
        # without tanh, directly sigmoid
        self.fc1 = nn.Linear(in_channels, 1)
        self.fc2 = nn.Linear(in_channels, 1)

        # self.pool = nn.MaxPool1d(num_points)

    def forward(self, pts_features, img_features):
        # without tanh, directly sigmoid
        pts_features_BND = pts_features.transpose(1, 2)
        img_features_BND = img_features.transpose(1, 2)
        pts_w = self.fc1(pts_features_BND)
        img_w = self.fc2(img_features_BND)
        add_w = torch.add(pts_w, img_w)
        weight_map = torch.sigmoid(add_w)
        weight_map = weight_map.transpose(1, 2)
        img_features = weight_map * img_features

        fusion_features = torch.cat((pts_features, img_features), dim=1)
        # print(fusion_features.shape)
        return fusion_features

def feature_gather(feature_map, xy):
    """
    Gather the value of the feature_map at xy
    :param feature_map: image feature map
    :param xy: (B, N, 2) normalized to [-1, 1]
    :return: interpolate feature
    """
    # xy (B, N, 2) -> (B, 1, N, 2)
    xy = xy.unsqueeze(1)
    interpolate_feature = F.grid_sample(feature_map, xy)
    return interpolate_feature.squeeze(2)  # (B, C', N)


class FusionModel(nn.Module):
    def __init__(self, feature_transform=True, num_points=5000):
        super(FusionModel, self).__init__()
        self.feature_transform = feature_transform
        self.ImgStream = nn.ModuleList()
        self.PtsStream = nn.ModuleList()
        self.FusionStream = nn.ModuleList()
        self.stn = STN3d(3)
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.max_pooling = nn.MaxPool2d((12, 40), 1)
        self.fc = nn.Linear(512, 128)

        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.vlad = NetVLADLoupe(feature_size=1024, max_samples=num_points, cluster_size=64,
                                 output_dim=128)

        for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
            self.ImgStream.append(ImgBasicBlock(cfg.LI_FUSION.IMG_CHANNELS[i],
                                                cfg.LI_FUSION.IMG_CHANNELS[i + 1],
                                                stride=1))

        for j in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 2):
            self.PtsStream.append(PtsMlpBlock(cfg.LI_FUSION.IMG_CHANNELS[j + 1],
                                              cfg.LI_FUSION.IMG_CHANNELS[j + 2]))

        for t in range(0, 4):
            self.FusionStream.append(FusionConv(cfg.LI_FUSION.FUSION_CHANNELS[t]))

        self.FinalFusion = FinalFusion(512)

    @staticmethod
    def _break_up_pointcloud(pc):
        xyz = pc[..., 0:3].contiguous()
        feature = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, feature

    def forward(self, p, image=None, xy=None):
        trans = self.stn(p)
        p = p.transpose(2, 1)
        p = torch.bmm(p, trans)
        p = p.transpose(2, 1)
        p = F.relu(self.bn1(self.conv1(p)))

        if self.feature_transform:
            trans_feat = self.fstn(p)
            p = p.transpose(2, 1)
            p = torch.bmm(p, trans_feat)
            p = p.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = p

        # normalize xy to [-1, 1]
        size_range = [640.0, 192.0]
        #  B    N    D
        xy[:, :, 0] = xy[:, :, 0] / (size_range[0] - 1.0) * 2 - 1.0
        xy[:, :, 1] = xy[:, :, 1] / (size_range[1] - 1.0) * 2 - 1.0

        img = [image]

        l_pc_feature = [p]
        l_img_feature = []

        for i in range(0, 4):
            image_feature_map = self.ImgStream[i](img[i])
            image_gather_feature = feature_gather(image_feature_map, xy)
            l_img_feature.append(image_gather_feature)
            if i < 3:
                fusion_feature = self.FusionStream[i](l_pc_feature[i], l_img_feature[i])
                enhanced_pc_feature = self.PtsStream[i](fusion_feature)
                l_pc_feature.append(enhanced_pc_feature)
            img.append(image_feature_map)

        fusion_feature = self.FinalFusion(l_pc_feature[3], l_img_feature[3])

        global_feature = self.vlad(fusion_feature)

        return global_feature



if __name__ == '__main__':
    dataset = KittiOdmDataset(list_files=[r"/home/omnisky/PycharmProjects/yx/PIFusion/TrainCmpDataPath_00.txt"])
    index = random.randint(0, len(dataset))
    img1, img2, pts1, pts2, l, xy1, xy2 = dataset[index]
    pts1 = pts1.unsqueeze(0)
    pts1 = pts1.transpose(2, 1)
    print(pts1.shape)
    model = FusionModel()
    x = model(pts1, image=None, xy=xy1)
    print(x)
