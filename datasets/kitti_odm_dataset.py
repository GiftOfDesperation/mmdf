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
import utils.kitti_calibration as calib
import cv2


class KittiOdmDataset(data.Dataset):
    def __init__(self, list_files, split='Train', standardize=None):
        self.split = split
        self.standardize = standardize
        self.data_root = '/data/semanticKITTI/sequences'
        # self.sequence = '00'
        # self.image_dir = os.path.join(self.data_root, self.sequence, 'image_2')
        # self.pcd_dir = os.path.join(self.data_root, self.sequence, 'processed_pcd')
        # self.bin_dir = os.path.join(self.data_root, self.sequence, 'velodyne')
        self.calib_path_00 = os.path.join(self.data_root, '00', 'calib.txt')
        self.calib_00 = calib.Calibration(self.calib_path_00)
        self.calib_path_02 = os.path.join(self.data_root, '02', 'calib.txt')
        self.calib_02 = calib.Calibration(self.calib_path_02)
        self.calib_path_05 = os.path.join(self.data_root, '05', 'calib.txt')
        self.calib_05 = calib.Calibration(self.calib_path_05)
        self.calib_path_06 = os.path.join(self.data_root, '06', 'calib.txt')
        self.calib_06 = calib.Calibration(self.calib_path_06)
        self.calib_path_07 = os.path.join(self.data_root, '07', 'calib.txt')
        self.calib_07 = calib.Calibration(self.calib_path_07)
        self.calib_path_09 = os.path.join(self.data_root, '09', 'calib.txt')
        self.calib_09 = calib.Calibration(self.calib_path_09)
        self.npoints = 5000

        self.pairfile_paths = []  # full path of two file,
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
        if self.pairfile_paths[index][0].split('/')[4] == '07':
            xy1 = self.calib_07.lidar_to_img(points1_raw)[0]
            xy1[:, 0] = xy1[:, 0] / 1226 * 640
            xy1[:, 1] = xy1[:, 1] / 370 * 192
            xy2 = self.calib_07.lidar_to_img(points2_raw)[0]
            xy2[:, 0] = xy2[:, 0] / 1226 * 640
            xy2[:, 1] = xy2[:, 1] / 370 * 192
        elif self.pairfile_paths[index][0].split('/')[4] == '09':
            xy1 = self.calib_09.lidar_to_img(points1_raw)[0]
            xy1[:, 0] = xy1[:, 0] / 1226 * 640
            xy1[:, 1] = xy1[:, 1] / 370 * 192
            xy2 = self.calib_09.lidar_to_img(points2_raw)[0]
            xy2[:, 0] = xy2[:, 0] / 1226 * 640
            xy2[:, 1] = xy2[:, 1] / 370 * 192
        elif self.pairfile_paths[index][0].split('/')[4] == '00':
            xy1 = self.calib_00.lidar_to_img(points1_raw)[0]
            xy1[:, 0] = xy1[:, 0] / 1241 * 640
            xy1[:, 1] = xy1[:, 1] / 376 * 192
            xy2 = self.calib_00.lidar_to_img(points2_raw)[0]
            xy2[:, 0] = xy2[:, 0] / 1241 * 640
            xy2[:, 1] = xy2[:, 1] / 376 * 192
        elif self.pairfile_paths[index][0].split('/')[4] == '02':
            xy1 = self.calib_02.lidar_to_img(points1_raw)[0]
            xy1[:, 0] = xy1[:, 0] / 1241 * 640
            xy1[:, 1] = xy1[:, 1] / 376 * 192
            xy2 = self.calib_02.lidar_to_img(points2_raw)[0]
            xy2[:, 0] = xy2[:, 0] / 1241 * 640
            xy2[:, 1] = xy2[:, 1] / 376 * 192
        elif self.pairfile_paths[index][0].split('/')[4] == '05':
            xy1 = self.calib_05.lidar_to_img(points1_raw)[0]
            xy1[:, 0] = xy1[:, 0] / 1226 * 640
            xy1[:, 1] = xy1[:, 1] / 370 * 192
            xy2 = self.calib_05.lidar_to_img(points2_raw)[0]
            xy2[:, 0] = xy2[:, 0] / 1226 * 640
            xy2[:, 1] = xy2[:, 1] / 370 * 192
        elif self.pairfile_paths[index][0].split('/')[4] == '06':
            xy1 = self.calib_06.lidar_to_img(points1_raw)[0]
            xy1[:, 0] = xy1[:, 0] / 1226 * 640
            xy1[:, 1] = xy1[:, 1] / 370 * 192
            xy2 = self.calib_06.lidar_to_img(points2_raw)[0]
            xy2[:, 0] = xy2[:, 0] / 1226 * 640
            xy2[:, 1] = xy2[:, 1] / 370 * 192
        
        l = self.label[index]
        l = torch.from_numpy(np.array([l]).astype(np.int64))

        return img1_tensor, img2_tensor, points1_tensor, points2_tensor, l, xy1, xy2

    def __len__(self):
        return len(self.pairfile_paths)

    def get_image(self, path):
        img_path = path.replace('processed_pcd1', 'image_2').replace('.pcd', '.png')
        # print(img_path)
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


if __name__ == '__main__':
    print('test start')

    dataset = KittiOdmDataset(list_files=[r"/home/omnisky/PycharmProjects/yx/PIFusion/TrainCmpDataPath_all.txt"])
    print('data length: {}'.format(len(dataset)))
    for i in range(100):
        index = random.randint(0, len(dataset))
        img1, img2, pts1, pts2, l, xy1, xy2 = dataset[index]
        print(img1.shape, img2.shape, pts1.shape, pts2.shape)
        print(xy1.shape)

