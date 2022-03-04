import open3d
import utils.functions as functions
import os
import pcl
from utils.kitti_calibration import Calibration
import numpy as np


def limit_point_valid_flag(point, calib):
    x_range = [0, 1226]
    y_range = [0, 370]
    point = [point]
    np_point = np.asarray(point)
    point_out, _ = calib.lidar_to_img(np_point)
    # print(point_out)
    point_out = point_out[0]
    if x_range[0] <= point_out[0] <= x_range[1] and y_range[0] <= point_out[1] <= y_range[1]:
        return point
    else:
        return None


if __name__ == '__main__':
    root = '/data/semanticKITTI/sequences/00/pcd/'
    save_path = '/data/semanticKITTI/sequences/00/processed_pcd/'
    calib_path = '/data/semanticKITTI/sequences/00/calib.txt'
    files = os.listdir(root)

    calib = Calibration(calib_path)

    for file in files:
        file_path = root + file
        cloud = open3d.io.read_point_cloud(file_path)

        pcl_cloud = functions.o32pcl(cloud)
        pcl_cloud_filtered = functions.range_filter(pcl_cloud, 50)
        ground_labels = functions.get_ground_label(pcl_cloud_filtered, 0.15)
        pcl_cloud_filtered = functions.ground_filter(pcl_cloud_filtered, ground_labels)
        # print(pcl_cloud_filtered)
        pcl_limit_range = []
        for point in pcl_cloud_filtered:
            _point = limit_point_valid_flag(point, calib)
            if _point is not None:
                pcl_limit_range.append(_point[0])

        pcl_limit_range = np.array(pcl_limit_range, dtype=np.float32)
        pcl_limit_range = functions.np2pcl(pcl_limit_range)

        # o3pcf = functions.pcl2o3(pcl_cloud_filtered)
        # print(save_path+file)
        # pcl.save(pcl_limit_range, '../processed_pcd/' + file, format='pcd')
        print("Save %s success" % file)
        # open3d.io.write_point_cloud(save_path + file, o3pcf)
    print('Saving done.')
