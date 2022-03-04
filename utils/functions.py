import open3d
import pcl
import numpy as np
import math
import os


# filter points outside x m
def range_filter(cloud_in, r_thresh):
    cloud_out = []
    for point in cloud_in:
        dist = point[0] * point[0] + point[1] * point[1]
        if dist < r_thresh * r_thresh:
            cloud_out.append(point)
    return np.asarray(cloud_out, dtype=np.float32)


# filter out ground points
def get_ground_label(cloud_in, h_thresh):
    # pre-allocate lists
    grid = []
    grid_label = []
    grid_h = []
    for i in range(0, 720):
        grid.append([])
        grid_label.append([])
        grid_h.append([])
        for j in range(0, 250):
            grid[i].append([])
            grid_label[i].append(0)
            grid_h[i].append(0)

    # fill in grids
    for index in range(0, len(cloud_in)):
        x = cloud_in[index][0]
        y = cloud_in[index][1]
        z = cloud_in[index][2]

        ang = math.atan2(y, x) * 180 / math.pi
        if ang < 0:
            ang = ang + 360
        ang_bin = math.floor(ang * 2)

        dis = math.sqrt(x * x + y * y)
        if dis >= 50:
            continue
        dis_bin = math.floor(dis * 5)
        grid[ang_bin][dis_bin].append(index)
        grid_label[ang_bin][dis_bin] = 0

    # first height thresh
    for i in range(0, 720):
        for j in range(0, 250):
            # find min max
            zmax = -1000
            zmin = 1000
            zmean = 0
            for k in range(0, len(grid[i][j])):
                index = grid[i][j][k]
                zmean = zmean + cloud_in[index][2]
                if cloud_in[index][2] > zmax:
                    zmax = cloud_in[index][2]
                if cloud_in[index][2] < zmin:
                    zmin = cloud_in[index][2]
            zmean = zmean / (len(grid[i][j]) + 0.00001)

            if (zmax == -1000) or (zmin == 1000) or (zmean == 0):
                continue

            grid_h[i][j] = zmean

            delta_h = zmax - zmin
            if delta_h < h_thresh:
                grid_label[i][j] = 1

    # ----------------------------------------------------------
    # calculate coarse grid height
    coarse_grid_h = []
    need_repair_i = []
    need_repair_j = []
    for i in range(0, 72):
        coarse_grid_h.append([])
        for j in range(0, 25):
            coarse_grid_h[i].append(0)
    for i in range(0, 72):
        for j in range(0, 25):
            mean = 0
            cnt = 0
            for ii in range(i * 10, i * 10 + 10):
                for jj in range(j * 10, j * 10 + 10):
                    if grid_label[ii][jj] == 1:
                        mean = mean + grid_h[ii][jj]
                        cnt = cnt + 1

            if cnt < 3:
                need_repair_i.append(i)
                need_repair_j.append(j)
            else:
                mean = mean / (cnt + 0.000001)
                coarse_grid_h[i][j] = mean

    # 4-nearest mean
    for k in range(0, len(need_repair_i)):
        iidx = need_repair_i[k]
        jjdx = need_repair_j[k]
        ii = [iidx - 1, iidx + 1]
        jj = [jjdx - 1, jjdx + 1]
        if iidx >= 71:
            ii[1] = 71
        if iidx <= 0:
            ii[0] = 0
        if jjdx >= 24:
            jj[1] = 24
        if jjdx <= 0:
            jj[0] = 0
        mean_mean_h = coarse_grid_h[ii[0]][jj[0]] + \
                      coarse_grid_h[ii[1]][jj[0]] + \
                      coarse_grid_h[ii[0]][jj[1]] + \
                      coarse_grid_h[ii[1]][jj[1]]
        mean_mean_h = mean_mean_h / 4
        coarse_grid_h[iidx][jjdx] = mean_mean_h
    # end calculate coarse grid height
    # ----------------------------------------------------------

    # second height thresh
    for i in range(0, 72):
        for j in range(0, 25):
            mi = i / 10
            mj = j / 10
            ii = [int(mi), int(mi + 1)]
            jj = [int(mj), int(mj + 1)]
            if mi == 71:
                ii[1] = 70
            if mj == 24:
                jj[1] = 23
            res = coarse_grid_h[ii[0]][jj[0]] + \
                  coarse_grid_h[ii[1]][jj[0]] + \
                  coarse_grid_h[ii[0]][jj[1]] + \
                  coarse_grid_h[ii[1]][jj[1]]
            res = res / 4
            if (grid_label[i][j] == 1) and (grid_h[i][j] - res > h_thresh):
                grid_label[i][j] = 0

    # output
    outIsGroundLabels = []
    for i in range(len(cloud_in)):
        outIsGroundLabels.append([])
    for i in range(0, 720):
        for j in range(0, 250):
            for k in range(0, len(grid[i][j])):
                if grid_label[i][j] == 0:
                    outIsGroundLabels[grid[i][j][k]] = 0
                else:
                    outIsGroundLabels[grid[i][j][k]] = 1

    return outIsGroundLabels


def ground_filter(cloud_in, ground_label):
    out_cloud = []
    for point, label in zip(cloud_in, ground_label):
        if label == 0:
            out_cloud.append(point)
        else:
            pass

    np_points = np.asarray(out_cloud, dtype=np.float32)
    pcl_cloud = pcl.PointCloud()
    pcl_cloud.from_array(np_points)
    # print(pcl_cloud.size)
    return pcl_cloud


def np2o3(np_points):
    o3_cloud = open3d.geometry.PointCloud()
    o3_cloud.points = open3d.utility.Vector3dVector(np_points)
    return o3_cloud


def o32np(o3_cloud):
    np_points = np.asarray(o3_cloud.points, dtype=np.float32)
    return np_points


def np2pcl(np_points):
    pcl_cloud = pcl.PointCloud()
    pcl_cloud.from_array(np_points)
    return pcl_cloud


def pcl2np(pcl_cloud):
    np_points = np.asarray(pcl_cloud, dtype=np.float32)
    return np_points


def o32pcl(o3_cloud):
    np_points = np.asarray(o3_cloud.points, dtype=np.float32)
    pcl_cloud = pcl.PointCloud()
    pcl_cloud.from_array(np_points)
    return pcl_cloud


def pcl2o3(pcl_cloud):
    np_points = np.asarray(pcl_cloud, dtype=np.float32)
    o3_cloud = open3d.geometry.PointCloud()
    o3_cloud.points = open3d.utility.Vector3dVector(np_points)
    return o3_cloud


if __name__ == '__main__':
    import time
    import random

    # yx
    pcd_file = '/data/semanticKITTI/sequences/00/pcd/001116.pcd'
    o3_cloud = open3d.io.read_point_cloud(pcd_file, format='pcd')
    pcl_cloud = o32pcl(o3_cloud)

    # o3_cloud = pcl2o3(pcl_cloud)
    # open3d.io.write_point_cloud('test0.pcd', o3_cloud, write_ascii=True)

    # vg = pcl_cloud.make_voxel_grid_filter()
    # vg.set_leaf_size(0.15, 0.15, 0.15)
    # pcl_cloud_filtered = vg.filter()

    # o3_cloud = pcl2o3(pcl_cloud_filtered)
    # open3d.io.write_point_cloud('test1.pcd', o3_cloud, write_ascii=True)

    pcl_cloud_filtered = range_filter(pcl_cloud, 50)

    # o3_cloud = pcl2o3(pcl_cloud_filtered)
    # open3d.io.write_point_cloud('test2.pcd', o3_cloud, write_ascii=True)

    ground_labels = get_ground_label(pcl_cloud_filtered, 0.15)
    pcl_cloud_filtered = ground_filter(pcl_cloud_filtered, ground_labels)

    o3_cloud = pcl2o3(pcl_cloud_filtered)
    # open3d.io.write_point_cloud('test3.pcd', o3_cloud, write_ascii=True)
    clusters = get_clusters(pcl_cloud_filtered, 0.3, 100, 100000)
    print("get {} clusters".format(len(clusters)))
    o3_clusters = []
    for i, cluster in enumerate(clusters, 0):
        # print("cluster {} has {} points.".format(i, len(cluster)))
        o3_cluster = np2o3(cluster)
        o3_cluster.paint_uniform_color([random.random(), random.random(), random.random()])
        o3_clusters.append(o3_cluster)

    open3d.visualization.draw_geometries(o3_clusters)
    print('---------------------------------')
    time.sleep(2)

    # hwc
    print('test start')
    pcd_files = os.listdir('/data/kitti_3d/pcd/training/')
    for pcd_file in pcd_files:
        pcd_file = '/data/kitti_3d/pcd/training/' + pcd_file
        o3_cloud = open3d.io.read_point_cloud(pcd_file, format='pcd')
        pcl_cloud = o32pcl(o3_cloud)

        # o3_cloud = pcl2o3(pcl_cloud)
        # open3d.io.write_point_cloud('test0.pcd', o3_cloud, write_ascii=True)

        # vg = pcl_cloud.make_voxel_grid_filter()
        # vg.set_leaf_size(0.15, 0.15, 0.15)
        # pcl_cloud_filtered = vg.filter()

        # o3_cloud = pcl2o3(pcl_cloud_filtered)
        # open3d.io.write_point_cloud('test1.pcd', o3_cloud, write_ascii=True)

        pcl_cloud_filtered = range_filter(pcl_cloud, 50)

        # o3_cloud = pcl2o3(pcl_cloud_filtered)
        # open3d.io.write_point_cloud('test2.pcd', o3_cloud, write_ascii=True)

        ground_labels = get_ground_label(pcl_cloud_filtered, 0.15)
        pcl_cloud_filtered = ground_filter(pcl_cloud_filtered, ground_labels)

        o3_cloud = pcl2o3(pcl_cloud_filtered)
        # open3d.io.write_point_cloud('test3.pcd', o3_cloud, write_ascii=True)

        clusters = get_clusters(pcl_cloud_filtered, 0.3, 100, 100000)
        print("get {} clusters".format(len(clusters)))
        o3_clusters = []
        for i, cluster in enumerate(clusters, 0):
            # print("cluster {} has {} points.".format(i, len(cluster)))
            o3_cluster = np2o3(cluster)
            o3_cluster.paint_uniform_color([random.random(), random.random(), random.random()])
            o3_clusters.append(o3_cluster)

        open3d.visualization.draw_geometries(o3_clusters)
        print('---------------------------------')
        time.sleep(2)
