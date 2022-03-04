#coding=utf-8
import open3d
import os
import copy
import random
import time
import pickle
import numpy as np
from sklearn.neighbors import KDTree


def convert_number_to_name(number):
    name = str(number)
    if len(name) < 6:
        for j in range(6 - len(name)):
            name = '0' + name
    return name


def read_calib(fn):
    with open(fn, 'r') as file:
        lines = file.readlines()
        Tr_line = lines[4]
        sp = Tr_line.strip().split(' ')
        sp = sp[1:]
        sp = [float(v) for v in sp]
        sp.append(0)
        sp.append(0)
        sp.append(0)
        sp.append(1)

        Tr = np.asarray(sp).reshape((4, 4))

    return Tr


def read_gt(fn):
    poses = []
    rots = []
    trans = []
    with open(fn, 'r') as file:
        while True:
            line = file.readline().strip()
            if line == '':
                break
            sp = line.split(' ')
            sp = [float(v) for v in sp]
            sp.append(0)
            sp.append(0)
            sp.append(0)
            sp.append(1)

            Tr = np.asarray(sp).reshape((4, 4))
            rot = Tr[0:3, 0:3]
            tran = Tr[0:3, 3]

            poses.append(Tr)
            rots.append(rot)
            trans.append(tran)

    return poses, rots, trans


def read_gt_in_lidar_coord(fn, calibTr):
    poses, rots, trans = read_gt(fn)
    poses_l = []
    rots_l = []
    trans_l = []

    for i in range(len(poses)):
        pose_l = np.dot(np.linalg.inv(calibTr), poses[i])

        rot_l = pose_l[0:3, 0:3]
        tran_l = pose_l[0:3, 3]

        poses_l.append(pose_l)
        rots_l.append(rot_l)
        trans_l.append(tran_l)

    return poses_l, rots_l, trans_l


if __name__ == '__main__':

    sequence_no = 6

    calib_file_path = '/data/semanticKITTI/sequences/%02d/calib.txt' % sequence_no
    Tr = read_calib(calib_file_path)
    # print(Tr)
    #
    gt_file_path = '/data/semanticKITTI/poses/%02d.txt' % sequence_no
    poses, rots, trans = read_gt(gt_file_path)
    # print(len(poses), len(rots), len(trans))
    #
    poses_l, rots_l, trans_l = read_gt_in_lidar_coord(gt_file_path, Tr)
    print(trans)
    print(trans_l)
    pcd_file_path = '/data/semanticKITTI/sequences/%02d/pcd/' % sequence_no

    if sequence_no == 5:
        # database frames: 0~1200
        tree = KDTree(np.asarray(trans[0:1200]))
        # query frames: 1300~2700
        neighbor_3m = tree.query_radius(np.asarray(trans[1300:2700:2]), r=3)
        neighbor_5m = tree.query_radius(np.asarray(trans[1300:2700:2]), r=5)
        neighbor_2m = tree.query_radius(np.asarray(trans[1300:2700:2]), r=2)
        neighbor_10m = tree.query_radius(np.asarray(trans[1300:2700:2]), r=10)
        neighbor_15m = tree.query_radius(np.asarray(trans[1300:2700:2]), r=15)
        neighbor_25m = tree.query_radius(np.asarray(trans[1300:2700:2]), r=25)
        neighbor_50m = tree.query_radius(np.asarray(trans[1300:2700:2]), r=50)

        # test_sets = {}
        test_data_path = []
        for i in range(len(neighbor_5m)):
            query = 1300 + i*2
 
            positives = neighbor_5m[i]
            point_count = len(positives)
            if point_count > 5:
                choice = np.random.choice(point_count, 5)
                positives = positives[choice]
            positives = positives.tolist()
            
            negatives = np.setdiff1d(np.asarray([a for a in range(0, 1200)]), neighbor_50m[i])
            point_count = len(negatives)
            if point_count > 5 * len(positives):
                choice = np.random.choice(point_count, 5 * len(positives))
                negatives = negatives[choice]

            negatives = negatives.tolist()
            query_path = '/data/semanticKITTI/sequences/%02d/processed_pcd/' % sequence_no \
                         + convert_number_to_name(query) + '.pcd'
            for positive in positives:
                positive_path = '/data/semanticKITTI/sequences/%02d/processed_pcd/' % sequence_no \
                         + convert_number_to_name(positive) + '.pcd'
                test_data_path.append([query_path, positive_path, 1])
            for negative in negatives:
                negative_path = '/data/semanticKITTI/sequences/%02d/processed_pcd/' % sequence_no \
                         + convert_number_to_name(negative) + '.pcd'
                test_data_path.append([query_path, negative_path, 0])

        
        with open('TestDataPairs05_1_5.txt', 'w') as f:
            for pairs in test_data_path:
                print(pairs)
                f.write(pairs[0])
                f.write(' ')
                f.write(pairs[1])
                f.write(' ')
                f.write(str(pairs[2]))
                f.write('\n')

    else:
        pass
