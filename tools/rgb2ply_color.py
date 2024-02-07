import cv2
import numpy as np
import open3d as o3d

# read image
image = cv2.imread('disp.png', cv2.IMREAD_GRAYSCALE)

# create point cloud
height, width = image.shape
x, y = np.meshgrid(np.arange(width), np.arange(height))
z = image
pts = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

#color generation
colors = np.zeros([pts.shape[0], 3])
height_max = np.max(pts[:, 2])
height_min = np.min(pts[:, 2])
delta_c = abs(height_max - height_min) / (255 * 2)
for j in range(pts.shape[0]):
    color_n = (pts[j, 2] - height_min) / delta_c
    if color_n <= 255:
        colors[j, :] = [0, 1 - color_n / 255, 1]
    else:
        colors[j, :] = [(color_n - 255) / 255, 0, 1]
pcd.colors = o3d.utility.Vector3dVector(colors)



# create point cloud object
o3d.io.write_point_cloud('12.ply', pcd)