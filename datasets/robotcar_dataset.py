import open3d
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import os
from PIL import Image
import random
import cv2
from oxford_robotcar.camera_model import CameraModel
import re
from oxford_robotcar.transform import build_se3_transform


class RobotCarDataset(data.Dataset):
    def __init__(self, list_files, split='Train', standardize=None):
        self.split = split
        self.standardize = standardize
        self.data_root = '/data/OxfordRobotCar'
        self.sequence1 = '2014-11-14-16-34-33'
        self.sequence2 = '2014-11-18-13-20-12'
        self.image_dir1 = os.path.join(self.data_root, self.sequence1, 'stereo', 'centre')
        self.pcd_dir1 = os.path.join(self.data_root, self.sequence1, 'processed_pcd')
        self.image_dir2 = os.path.join(self.data_root, self.sequence2, 'stereo', 'centre')
        self.pcd_dir2 = os.path.join(self.data_root, self.sequence2, 'processed_pcd')

        self.models_dir = '/data/OxfordRobotCar/robotcar-dataset-sdk-3.1/models'
        self.extrinsics_dir = '/data/OxfordRobotCar/robotcar-dataset-sdk-3.1/extrinsics'
        self.poses_file1 = '/data/OxfordRobotCar/2014-11-14-16-34-33/vo/vo.csv'
        self.poses_file2 = '/data/OxfordRobotCar/2014-11-18-13-20-12/vo/vo.csv'
        self.model = CameraModel(self.models_dir, self.image_dir1)

        extrinsics_path = os.path.join(self.extrinsics_dir, self.model.camera + '.txt')
        with open(extrinsics_path) as extrinsics_file:
            extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

        G_camera_vehicle = build_se3_transform(extrinsics)
        self.G_camera_posesource1 = None
        self.G_camera_posesource2 = None

        poses_type1 = re.search('(vo|ins|rtk)\.csv', self.poses_file1).group(1)
        if poses_type1 in ['ins', 'rtk']:
            with open(os.path.join(self.extrinsics_dir, 'ins.txt')) as extrinsics_file:
                extrinsics = next(extrinsics_file)
                self.G_camera_posesource1 = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
        else:
            # VO frame and vehicle frame are the same
            self.G_camera_posesource1 = G_camera_vehicle

        poses_type2 = re.search('(vo|ins|rtk)\.csv', self.poses_file2).group(1)
        if poses_type2 in ['ins', 'rtk']:
            with open(os.path.join(self.extrinsics_dir, 'ins.txt')) as extrinsics_file:
                extrinsics = next(extrinsics_file)
                self.G_camera_posesource2 = G_camera_vehicle * build_se3_transform(
                    [float(x) for x in extrinsics.split(' ')])
        else:
            # VO frame and vehicle frame are the same
            self.G_camera_posesource2 = G_camera_vehicle

        self.npoints = 5000

        self.pairfile_paths = []  
        self.label = []  # class label

        for list_file in list_files:
            with open(list_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    path1, path2, issimilar = line.split(' ')
                    self.pairfile_paths.append([path1, path2])
                    self.label.append(int(issimilar))
                   

    def __getitem__(self, index):
        o3pts1 = open3d.io.read_point_cloud(self.pairfile_paths[index][0])
        o3pts2 = open3d.io.read_point_cloud(self.pairfile_paths[index][1])

        points1_raw = np.asarray(o3pts1.points, dtype=np.float32)
        points2_raw = np.asarray(o3pts2.points, dtype=np.float32)

        if self.standardize:
            o3pts1 = self.standardize(o3pts1)
            o3pts2 = self.standardize(o3pts2)

        points1 = np.asarray(o3pts1.points, dtype=np.float32)
        points2 = np.asarray(o3pts2.points, dtype=np.float32)

        img1_tensor = self.get_image(self.pairfile_paths[index][0])
        img2_tensor = self.get_image(self.pairfile_paths[index][1])

        # resample
        point_count = len(points1)
        choice = np.random.choice(point_count, self.npoints)
        points1 = points1[choice, :]
        points1_raw = points1_raw[choice, :]
        points1_tensor = torch.from_numpy(points1)

        point_count = len(points2)
        choice = np.random.choice(point_count, self.npoints)
        points2 = points2[choice, :]
        points2_raw = points2_raw[choice, :]
        points2_tensor = torch.from_numpy(points2)

        # xy1 (array x, array y)
        points1_raw = points1_raw.transpose((1, 0))
        points2_raw = points2_raw.transpose((1, 0))
        if points1_raw.shape[0] == 3:
            points1_raw = np.vstack((points1_raw, np.ones((1, points1_raw.shape[1]))))
        if points2_raw.shape[0] == 3:
            points2_raw = np.vstack((points2_raw, np.ones((1, points2_raw.shape[1]))))

        points1_raw = np.dot(self.G_camera_posesource1, points1_raw)
        points2_raw = np.dot(self.G_camera_posesource2, points2_raw)
        # print(points1.shape)
        xy1, _, _ = self.model.project(points1_raw, (1280, 960))
        xy2, _, _ = self.model.project(points2_raw, (1280, 960))
        xy1 = xy1.transpose((1, 0))
        xy2 = xy2.transpose((1, 0))
        xy1 = np.asarray(xy1)
        xy2 = np.asarray(xy2)
        xy1[:, 0] = xy1[:, 0] / 1280 * 640
        xy1[:, 1] = xy1[:, 1] / 960 * 192
        xy2[:, 0] = xy2[:, 0] / 1280 * 640
        xy2[:, 1] = xy2[:, 1] / 960 * 192
        l = self.label[index]
        l = torch.from_numpy(np.array([l]).astype(np.int64))

        return img1_tensor, img2_tensor, points1_tensor, points2_tensor, l, xy1, xy2

    def __len__(self):
        return len(self.pairfile_paths)

    def get_image(self, path):
        sequence = path.split('/')[3]
        name = path.split('/')[5].replace('.pcd', '')
        img_path = os.path.join(self.data_root, sequence, 'stereo/centre', name + '.png')
        assert os.path.exists(img_path)
        img = Image.open(img_path).convert('RGB')
        img = np.array(img).astype(np.float)
        img = self.rescale(img)
        # img = img / 255.0
        # img -= mean
        # img /= std
        imgback = np.zeros([192, 640, 3], dtype=np.float)
        imgback[:img.shape[0], :img.shape[1], :] = img  
        imgback = imgback.transpose((2, 0, 1))
        imgback_tensor = torch.from_numpy(imgback).float()
        return imgback_tensor

    def rescale(self, image, output_size_cfg=(192, 640)):
        if isinstance(output_size_cfg, int):
            output_size = (output_size_cfg, output_size_cfg)
        else:
            output_size = output_size_cfg
        h, w = image.shape[:2]
        if output_size == (h, w):
            return image

        new_h, new_w = output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)

        return img


class Standardize(object):
    """
    将点云去均值
    """

    def __init__(self):
        # do nothing
        pass

    def __call__(self, pts):
        max_bound = pts.get_max_bound()
        scale_coeff = 1 / max(max_bound)

        (mean, covariance) = pts.compute_mean_and_covariance()
        # print("mean1:{}".format(mean))

        # pts.translate(-mean)
        T_mat = \
           [[1, 0, 0, -mean[0]],
            [0, 1, 0, -mean[1]],
            [0, 0, 1, -mean[2]],
            [0, 0, 0, 1]]

        pts.transform(T_mat)

        return pts


if __name__ == '__main__':
    print('test start')

    dataset = RobotCarDataset(list_files=[r"/home/omnisky/PycharmProjects/yx/PIFusion/oxford_robotcar/RobotCarPairs_01.txt"])
    print('data length: {}'.format(len(dataset)))
    for i in range(100):
        index = random.randint(0, len(dataset))
        img1, img2, pts1, pts2, l, xy1, xy2 = dataset[index]
        print(img1.shape, img2.shape, pts1.shape, pts2.shape)
        print(xy1)

