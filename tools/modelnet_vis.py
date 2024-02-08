#coding=utf-8
import h5py
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm

def visualize_with_label(points, labels):
    assert points.shape[0] == labels.shape[0]
    bs = points.shape[0]
    for b in tqdm(range(bs)):
        # label = labels[b]
        point = points[b]
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))

        pt = o3d.geometry.PointCloud()
        pt.points = o3d.utility.Vector3dVector(point)
        pt.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([pt], point_show_normal=False, mesh_show_wireframe=True,
                                          mesh_show_back_face=True)


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


points, labels = load_h5("D:\\AI\\BinocularDisparity\\tools\\datasets\\mydata\\ply_data_train0.h5")


points = points[..., :]
labels = labels[...]
visualize_with_label(points, labels)