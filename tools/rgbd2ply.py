import open3d as o3d

color_raw = o3d.io.read_image("jpg_img/Color_1702793924519_1.png")
depth_raw = o3d.io.read_image("jpg_img/Depth_1702793920644_0.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)



inter = o3d.camera.PinholeCameraIntrinsic()
inter.set_intrinsics(640, 480, 545.102, 545.102, 312.62, 242.808)
pcd = o3d.geometry.PointCloud().create_from_rgbd_image(
    rgbd_image, inter)

# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])
