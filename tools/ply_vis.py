import open3d as o3d
import numpy as np
# visualization of point clouds.
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    # 选中的点为灰色，未选中点为红色
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # 可视化
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def remove_outlier(pcd, voxel_size=0.025, n_points=16, radius=0.25):
    pcd = pcd.voxel_down_sample(voxel_size)
    cloud, index = pcd.remove_radius_outlier(n_points, radius)
    cloud = cloud.select_by_index(index)
    return cloud

pcd = o3d.io.read_point_cloud('./ascend.ply')
# pcd = o3d.io.read_triangle_mesh('save_mesh.ply')


pcd = remove_outlier(pcd, 0.02, 16, 0.25)
# display_inlier_outlier(pcd, index)

# mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)   #泊松表面重建
# vertices_to_remove = densities < np.quantile(densities, 0.005)
# mesh.remove_vertices_by_mask(vertices_to_remove)

# out_arr = np.asarray(pcd.points)
# o3d.visualization.draw_geometries([pcd, mesh], point_show_normal=False, mesh_show_wireframe=True, mesh_show_back_face=True)
o3d.visualization.draw_geometries([pcd], point_show_normal=False, mesh_show_wireframe=True, mesh_show_back_face=True)
# o3d.io.write_point_cloud('../save.ply', cloud, write_ascii=True)
# o3d.io.write_triangle_mesh('../save_mesh.ply', mesh, write_ascii=True)