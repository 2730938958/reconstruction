import numpy as np
import open3d as o3d

pcd_load = o3d.io.read_point_cloud("data/bun000.ply")

pcd_load.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

# convert Open3D.o3d.geometry.PointCloud to numpy array
xyz_numpy = np.asarray(pcd_load.points)
xyz_normals = np.asarray(pcd_load.normals)

xyz = np.concatenate((xyz_normals, xyz_numpy), axis=1)

# o3d.visualization.draw_geometries([pcd_load],point_show_normal=True)

print(xyz)