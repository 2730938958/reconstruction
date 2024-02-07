import open3d as o3d
import numpy as np
import cv2

from matplotlib import pyplot as plt

disp=o3d.io.read_image("disp.png")
img=cv2.imread("disp.png")
size=img.shape
height,width=size[0],size[1]
fx, fy, cx, cy = 545.102, 545.102, 312.62, 242.808
#nump=np.asarray(disp)
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
pcd = o3d.geometry.PointCloud().create_from_depth_image(disp, intrinsic)

# Flip it, otherwise the pointcloud will be upside down
# pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])
