"""Evaluate RPMNet. Also contains functionality to compute evaluation metrics given transforms

Example Usages:
    1. Visualize RPMNet
        python vis.py --noise_type crop --resume [path-to-model.pth] --dataset_path [your_path]/modelnet40_ply_hdf5_2048

"""
import os
from tools.ply2mesh import pcd_to_mesh
import open3d as o3d
import random
import numpy as np
from tqdm import tqdm
import torch
import open3d as o3d
from arguments import rpmnet_eval_arguments
from common.misc import prepare_logger
from common.torch import dict_all_to_device, CheckPointManager, to_numpy
from common.math_torch import se3
from data_loader.datasets import get_test_datasets
import models.multi_view_rpmnet


def vis(npys):
    pcds = []
    colors = [[1.0, 0, 0],
              [0, 1.0, 0],
              [0, 0, 1.0]]
    for ind, npy in enumerate(npys):
        color = colors[ind] if ind < 3 else [random.random() for _ in range(3)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(npy)
        pcd.paint_uniform_color(color)
        # mesh = pcd_to_mesh(pcd)
        # o3d.io.write_triangle_mesh(
        #     "D:\\AI\\3Dpointcloud\\Point-cloud-registration\\RPMNet\\RPMNet\\src\\tools\\computer.ply",
        #     mesh, write_ascii=True)
        pcds.append(pcd)
    return pcds


def recompute_norm(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

    # convert Open3D.o3d.geometry.PointCloud to numpy array
    xyz_numpy = np.asarray(pcd.points)
    xyz_normals = np.asarray(pcd.normals)
    xyz_src = np.concatenate((xyz_numpy, xyz_normals), axis=1)
    return xyz_src


def registration(ref, src_list, model:torch.nn.Module):
    pcd_ref = o3d.io.read_point_cloud(ref)
    xyz_ref = torch.tensor(recompute_norm(pcd_ref), device=_device).float()
    xyz_ref, norm_ref = xyz_ref[None, :, :3], xyz_ref[None, :, 3:6]

    concatenate_tensor = xyz_ref

    for src in src_list:
        pcd_src = o3d.io.read_point_cloud(src)
        xyz_src = torch.tensor(recompute_norm(pcd_src), device=_device).float()
        xyz_src, norm_src = xyz_src[None, :, :3], xyz_src[None, :, 3:6]
        pred_transforms, endpoints = model(xyz_src, xyz_ref, norm_src, norm_ref, _args.num_reg_iter)
        src_transformed = se3.transform(pred_transforms[-1], xyz_src)
        concatenate_tensor = torch.cat((src_transformed, concatenate_tensor), axis=1)

    concatenate_cpp = torch.squeeze(concatenate_tensor).cpu().detach()
    return concatenate_cpp


def inference_vis(data_loader, model: torch.nn.Module):

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader):
            # dict_all_to_device(data,_device)
            #gt_transforms = data['transform_gt']

            points_src = data['points_src'][..., :3]
            points_ref = data['points_ref'][..., :3].squeeze()

            src = o3d.geometry.PointCloud()
            src.points = o3d.utility.Vector3dVector(points_src.squeeze().numpy())
            pcd_src = torch.tensor(recompute_norm(src), device=_device).float()
            xyz_src, norm_src = pcd_src[None, :, :3].cuda(), pcd_src[None, :, 3:6].cuda()

            ref = o3d.geometry.PointCloud()
            ref.points = o3d.utility.Vector3dVector(points_ref.squeeze().numpy())
            pcd_ref = torch.tensor(recompute_norm(ref), device=_device).float()
            xyz_ref, norm_ref = pcd_ref[None, :, :3].cuda(), pcd_ref[None, :, 3:6].cuda()
            #points_raw = data['points_raw'][..., :3]

            pred_transforms, endpoints = model(xyz_src, xyz_ref, norm_src, norm_ref, _args.num_reg_iter)
            # pred_transforms, endpoints = model(data, _args.num_reg_iter)
            src_transformed = se3.transform(pred_transforms[-1], points_src.cuda())

            src_np = torch.squeeze(points_src).cpu().detach()
            src_transformed_np = torch.squeeze(src_transformed).cpu().detach()
            ref_np = torch.squeeze(points_ref).cpu().detach()

            # pcds = vis([src_np, src_transformed_np, ref_np])
            pcds = vis([src_transformed_np, ref_np])
            o3d.visualization.draw_geometries(pcds)




    # with torch.no_grad():
    #     for data in tqdm(data_loader):
    #         dict_all_to_device(data,_device)
    #         #gt_transforms = data['transform_gt']
    #         points_src = data['points_src'][..., :3]
    #         points_ref = data['points_ref'][..., :3]
    #         #points_raw = data['points_raw'][..., :3]
    #
    #         xyz_ref, norm_ref = data['points_ref'][:, :, :3], data['points_ref'][:, :, 3:6]
    #         xyz_src, norm_src = data['points_src'][:, :, :3], data['points_src'][:, :, 3:6]
    #
    #         # pred_transforms, endpoints = model(data, _args.num_reg_iter)
    #         pred_transforms, endpoints = model(xyz_src, xyz_ref, norm_src, norm_ref, _args.num_reg_iter)
    #
    #         src_transformed = se3.transform(pred_transforms[-1], points_src)
    #
    #         # src_np = torch.squeeze(points_src).cpu().detach()
    #         src_transformed_np = torch.squeeze(src_transformed).cpu().detach()
    #         ref_np = torch.squeeze(points_ref).cpu().detach()
    #
    #         # pcds = vis([src_np, src_transformed_np, ref_np])
    #         pcds = vis([src_transformed_np, ref_np])
    #         o3d.visualization.draw_geometries(pcds)


    # with torch.no_grad():
    #     pcd_src = o3d.io.read_point_cloud("drill/data/drill_1.6mm_60_cyb.ply")
    #     pcd_ref = o3d.io.read_point_cloud("drill/data/drill_1.6mm_0_cyb.ply")
    #
    #
    #     xyz_src = torch.tensor(recompute_norm(pcd_src), device=_device).float()
    #     xyz_ref = torch.tensor(recompute_norm(pcd_ref), device=_device).float()
    #
    #
    #     # points_src = xyz_src[..., :3]
    #     # points_ref = xyz_ref[..., :3]
    #
    #
    #     xyz_ref, norm_ref = xyz_ref[None, :, :3], xyz_ref[None, :, 3:6]
    #     xyz_src, norm_src = xyz_src[None, :, :3], xyz_src[None, :, 3:6]
    #
    #     # xyz_ref, norm_ref = xyz_ref[:, :3], xyz_ref[:, 3:6]
    #     # xyz_src, norm_src = xyz_src[:, :3], xyz_src[:, 3:6]
    #
    #
    #     pred_transforms, endpoints = model(xyz_src, xyz_ref, norm_src, norm_ref, _args.num_reg_iter)
    #     src_transformed = se3.transform(pred_transforms[-1], xyz_src)
    #
    #     # src_transformed_tensor = torch.squeeze(src_transformed).cpu().detach()
    #     # ref_tensor = torch.squeeze(xyz_ref).cpu().detach()
    #
    #     concatenate_tensor = torch.cat((src_transformed,xyz_ref), axis=1)
    #
    #     concatenate = torch.squeeze(concatenate_tensor).cpu().detach()
    #
    #     # pcds = vis([src_np, src_transformed_np, ref_np])
    #     # pcds = vis([src_transformed_np, ref_np])
    #     pcds = vis([concatenate])
    #     o3d.visualization.draw_geometries(pcds)


def vis_registration(model: torch.nn.Module):
    model.eval()
    with torch.no_grad():
        src_list = ["tools/src.ply"]
        ref = "tools/ref.ply"
        concatenate_cpp = registration(ref, src_list, model)
        pcds = vis([concatenate_cpp])
        o3d.visualization.draw_geometries(pcds)



def get_model():
    _logger.info('Computing transforms using {}'.format(_args.method))
    assert _args.resume is not None
    # model = models.rpmnet.get_model(_args)
    model = models.multi_view_rpmnet.get_model(_args)
    model.to(_device)
    if _device == torch.device('cpu'):
        model.load_state_dict(
            torch.load(_args.resume, map_location=torch.device('cpu'))['state_dict'])
    else:
        model.load_state_dict(torch.load(_args.resume)['state_dict'])
    return model


def main():
    # Load data_loader
    test_dataset = get_test_datasets(_args)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1, shuffle=False)

    model = get_model()
    inference_vis(test_loader, model)  # Feedforward transforms

    # vis_registration(model)
    _logger.info('Finished')


if __name__ == '__main__':
    # Arguments and logging
    parser = rpmnet_eval_arguments()
    _args = parser.parse_args()
    _logger, _log_path = prepare_logger(_args, log_path=_args.eval_save_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
    if _args.gpu >= 0 and (_args.method == 'rpm' or _args.method == 'rpmnet'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
        _device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    else:
        _device = torch.device('cpu')
    main()
