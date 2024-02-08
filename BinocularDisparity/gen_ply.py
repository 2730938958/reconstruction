import cv2
import numpy as np
import open3d as o3d

def depToPcd(depth_map, camera_matrix, flatten=False, depth_scale=1000):
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
    z = depth_map / depth_scale
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy
    xyz = np.dstack((x, y, z)) if flatten == False else np.dstack((x, y, z)).reshape(-1, 3)
    # xyz=cv2.rgbd.depthTo3d(depth_map,camera_matrix)
    return xyz


def generate(disp_pred):
    camera_matrix = np.array([[540, 0, 320],
                                 [0, 540, 240],
                                 [0, 0, 1]])
    pc = depToPcd(disp_pred, camera_matrix)  # generate pc
    pc_flatten = pc.reshape(-1, 3)  # equals to pc = depToPcd(depth_map, camera_matrix, flatten=True)
    # height, width = image.shape
    # x, y = np.meshgrid(np.arange(width), np.arange(height))
    # z = image
    # points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    # create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_flatten)
    o3d.io.write_point_cloud('11.ply', pcd)

