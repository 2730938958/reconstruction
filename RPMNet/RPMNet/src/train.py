""" Train RPMNet

Example usage:
    python train.py --noise_type crop
    python train.py --noise_type jitter --train_batch_size 4
"""
from collections import defaultdict
import os
import random
from typing import Dict, List

from matplotlib.pyplot import cm as colormap
import numpy as np
import open3d  # Ensure this is imported before pytorch
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from arguments import rpmnet_train_arguments
from common.colors import BLUE, ORANGE
from common.misc import prepare_logger
from common.torch import dict_all_to_device, CheckPointManager, TorchDebugger
from common.math_torch import se3
from data_loader.datasets import get_train_datasets
from eval import compute_metrics, summarize_metrics, print_metrics
# from models.rpmnet import get_model
from models.multi_view_rpmnet import get_model
# import models.multi_view_rpmnet
"""
设置参数：
--noise_type clean 
--train_batch_size 4
"""

# Set up arguments and logging
parser = rpmnet_train_arguments()
_args = parser.parse_args()
_logger, _log_path = prepare_logger(_args)
if _args.gpu >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
    _device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
else:
    _device = torch.device('cpu')


def main():
    train_set, val_set = get_train_datasets(_args)
    run(train_set, val_set)

#
# def compute_losses(data: Dict, pred_transforms: List, endpoints: Dict,
#                    loss_type: str = 'mae', reduction: str = 'mean') -> Dict:
def compute_losses(source, pred_transforms: List, endpoints: Dict, gt,
                       loss_type: str = 'mae', reduction: str = 'mean') -> Dict:
    """Compute losses

    Args:
        data: Current mini-batch data
        pred_transforms: Predicted transform, to compute main registration loss
        endpoints: Endpoints for training. For computing outlier penalty
        loss_type: Registration loss type, either 'mae' (Mean absolute error, used in paper) or 'mse'
        reduction: Either 'mean' or 'none'. Use 'none' to accumulate losses outside
                   (useful for accumulating losses for entire validation dataset)

    Returns:
        losses: Dict containing various fields. Total loss to be optimized is in losses['total']

    """

    losses = {}
    num_iter = len(pred_transforms)

    # Compute losses
    # gt_src_transformed = se3.transform(data['transform_gt'], data['points_src'][..., :3])
    gt_src_transformed = se3.transform(gt, source)
    if loss_type == 'mse':
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = nn.MSELoss(reduction=reduction)
        for i in range(num_iter):
            # pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :3])
            pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :3])
            if reduction.lower() == 'mean':
                losses['mse_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
            elif reduction.lower() == 'none':
                losses['mse_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                        dim=[-1, -2])
    elif loss_type == 'mae':
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = nn.L1Loss(reduction=reduction)
        for i in range(num_iter):
            # pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :3])
            pred_src_transformed = se3.transform(pred_transforms[i], source)
            if reduction.lower() == 'mean':
                losses['mae_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
            elif reduction.lower() == 'none':
                losses['mae_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                        dim=[-1, -2])
    else:
        raise NotImplementedError

    # Penalize outliers
    for i in range(num_iter):
        ref_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=1)) * _args.wt_inliers
        src_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=2)) * _args.wt_inliers
        if reduction.lower() == 'mean':
            losses['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength) + torch.mean(src_outliers_strength)
        elif reduction.lower() == 'none':
            losses['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength, dim=1) + \
                                             torch.mean(src_outliers_strength, dim=1)

    discount_factor = 0.5  # Early iterations will be discounted
    total_losses = []
    for k in losses:
        discount = discount_factor ** (num_iter - int(k[k.rfind('_')+1:]) - 1)
        total_losses.append(losses[k] * discount)
    losses['total'] = torch.sum(torch.stack(total_losses), dim=0)

    return losses


def save_summaries(writer: SummaryWriter, data: Dict, predicted: List, endpoints: Dict = None,
                   losses: Dict = None, metrics: Dict = None, step: int = 0):
    """Save tensorboard summaries"""

    subset = [0, 1]
    # subset = [0]

    with torch.no_grad():
        # Save clouds
        if 'points_src' in data:

            points_src = data['points_src'][subset, ..., :3]
            points_ref = data['points_ref'][subset, ..., :3]

            colors = torch.from_numpy(
                np.concatenate([np.tile(ORANGE, (*points_src.shape[0:2], 1)),
                                np.tile(BLUE, (*points_ref.shape[0:2], 1))], axis=1))

            iters_to_save = [0, len(predicted)-1] if len(predicted) > 1 else [0]

            # Save point cloud at iter0, iter1 and after last iter
            concat_cloud_input = torch.cat((points_src, points_ref), dim=1)
            writer.add_mesh('iter_0', vertices=concat_cloud_input, colors=colors, global_step=step)
            for i_iter in iters_to_save:
                src_transformed_first = se3.transform(predicted[i_iter][subset, ...], points_src)
                concat_cloud_first = torch.cat((src_transformed_first, points_ref), dim=1)
                writer.add_mesh('iter_{}'.format(i_iter+1), vertices=concat_cloud_first, colors=colors, global_step=step)

            if endpoints is not None and 'perm_matrices' in endpoints:
                color_mapper = colormap.ScalarMappable(norm=None, cmap=colormap.get_cmap('coolwarm'))
                for i_iter in iters_to_save:
                    ref_weights = torch.sum(endpoints['perm_matrices'][i_iter][subset, ...], dim=1)
                    ref_colors = color_mapper.to_rgba(ref_weights.detach().cpu().numpy())[..., :3]
                    writer.add_mesh('ref_weights_{}'.format(i_iter), vertices=points_ref,
                                    colors=torch.from_numpy(ref_colors) * 255, global_step=step)

        if endpoints is not None:
            if 'perm_matrices' in endpoints:
                for i_iter in range(len(endpoints['perm_matrices'])):
                    src_weights = torch.sum(endpoints['perm_matrices'][i_iter], dim=2)
                    ref_weights = torch.sum(endpoints['perm_matrices'][i_iter], dim=1)
                    writer.add_histogram('src_weights_{}'.format(i_iter), src_weights, global_step=step)
                    writer.add_histogram('ref_weights_{}'.format(i_iter), ref_weights, global_step=step)

        # Write losses and metrics
        if losses is not None:
            for l in losses:
                writer.add_scalar('losses/{}'.format(l), losses[l], step)
        if metrics is not None:
            for m in metrics:
                writer.add_scalar('metrics/{}'.format(m), metrics[m], step)

        writer.flush()


def repeat(tensor):
    copy = tensor
    repeat_result = torch.cat((tensor,copy),0)
    return repeat_result

def validate(data_loader, model: torch.nn.Module, summary_writer: SummaryWriter, step: int):
    """Perform a single validation run, and saves results into tensorboard summaries"""
    _logger.info('Starting validation run...')
    with torch.no_grad():
        all_val_losses = defaultdict(list)
        all_val_metrics_np = defaultdict(list)
        for val_data in data_loader:
            # dict_all_to_device(val_data, _device)
            val_data = val_data
            val_data = val_data[0]
            pcd0 = val_data[0]
            pcd1 = val_data[1]
            T_gt = val_data[2][:3, :]
            T_gt = torch.tensor(T_gt.reshape(1, 3, 4)).type(torch.float32).to(_device)

            down0 = prepare_dataset(pcd0)
            down1 = prepare_dataset(pcd1)
            xyz_src, norm_src = downsample(down0)
            xyz_ref, norm_ref = downsample(down1)
            # pred_transforms, endpoints = model(xyz_src, xyz_ref, norm_src, norm_ref, _args.num_reg_iter)
            pred_test_transforms, endpoints = model(xyz_src, xyz_ref, norm_src, norm_ref, _args.num_reg_iter)
            val_losses = compute_losses(xyz_src, pred_test_transforms, endpoints,T_gt,
                                        loss_type=_args.loss_type, reduction='none')
            val_data = {'transform_gt': T_gt, 'points_src': xyz_src, 'points_ref': xyz_ref, 'points_raw': xyz_src}
            val_metrics = compute_metrics(val_data, pred_test_transforms[-1])

            for k in val_losses:
                all_val_losses[k].append(val_losses[k])
            for k in val_metrics:
                all_val_metrics_np[k].append(val_metrics[k])

        all_val_losses = {k: torch.cat(all_val_losses[k]) for k in all_val_losses}
        all_val_metrics_np = {k: np.concatenate(all_val_metrics_np[k]) for k in all_val_metrics_np}
        mean_val_losses = {k: torch.mean(all_val_losses[k]) for k in all_val_losses}

    # Rerun on random and worst data instances and save to summary
    rand_idx = random.randint(0, all_val_losses['total'].shape[0] - 1)
    worst_idx = torch.argmax(all_val_losses['{}_{}'.format(_args.loss_type, _args.num_reg_iter - 1)]).cpu().item()
    # indices_to_rerun = [rand_idx, worst_idx]
    indices_to_rerun = [rand_idx]
    # data_to_rerun = defaultdict(list)
    data_to_rerun = None
    for i in indices_to_rerun:
        cur = data_loader.dataset[i]
        # cur = cur
        # cur = cur[0]
        pcd0 = cur[0]
        pcd1 = cur[1]
        T_gt = cur[2][:3, :]
        T_gt = torch.tensor(T_gt.reshape(1, 3, 4)).type(torch.float32).to(_device)

        down0 = prepare_dataset(pcd0)
        down1 = prepare_dataset(pcd1)
        xyz_src, norm_src = downsample(down0)
        xyz_ref, norm_ref = downsample(down1)
        cur = {'transform_gt': repeat(T_gt), 'points_src': repeat(xyz_src), 'points_ref': repeat(xyz_ref), 'points_raw': repeat(xyz_src), 'norm_src': repeat(norm_src), 'norm_ref': repeat(norm_ref)}
        # for k in cur:
            # data_to_rerun[k].append(cur[k].cpu())
            # data_to_rerun[k].append(cur[k])
        data_to_rerun = cur
    # for k in data_to_rerun:
    #     data_to_rerun[k] = torch.from_numpy(np.stack(data_to_rerun[k], axis=0))
    # dict_all_to_device(data_to_rerun, _device)
    # data_to_rerun = data_to_rerun[0]
    pred_transforms, endpoints = model(data_to_rerun['points_src'], data_to_rerun['points_ref'],
                                       data_to_rerun['norm_src'], data_to_rerun['norm_ref'], _args.num_reg_iter)

    summary_metrics = summarize_metrics(all_val_metrics_np)
    losses_by_iteration = torch.stack([mean_val_losses['{}_{}'.format(_args.loss_type, k)]
                                       for k in range(_args.num_reg_iter)]).cpu().numpy()
    print_metrics(_logger, summary_metrics, losses_by_iteration, 'Validation results')

    save_summaries(summary_writer, data=data_to_rerun, predicted=pred_transforms, endpoints=endpoints,
                   losses=mean_val_losses, metrics=summary_metrics, step=step)

    score = -summary_metrics['chamfer_dist']
    return score

def make_open3d_point_cloud(xyz, color=None):
  pcd = open3d.geometry.PointCloud()
  pcd.points = open3d.utility.Vector3dVector(xyz)
  if color is not None:
    if len(color) != len(xyz):
      color = np.tile(color, (len(xyz), 1))
    pcd.colors = open3d.utility.Vector3dVector(color)
  return pcd

def prepare_dataset(pcd0):
    source = make_open3d_point_cloud(pcd0)
    source = source.voxel_down_sample(0.1)
    return source

def recompute_norm(pcd):
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    xyz_numpy = np.asarray(pcd.points)
    xyz_normals = np.asarray(pcd.normals)
    xyz_src = np.concatenate((xyz_numpy, xyz_normals), axis=1)
    return xyz_src

def downsample(source):
    xyz_src = torch.tensor(recompute_norm(source), device=_device).float()
    xyz_src, norm_src = xyz_src[None, :, :3], xyz_src[None, :, 3:6]
    return xyz_src, norm_src

def run(train_set, val_set):
    """Main train/val loop"""

    _logger.debug('Trainer (PID=%d), %s', os.getpid(), _args)

    model = get_model(_args)
    model.to(_device)
    global_step = 0

    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_set,collate_fn=lambda x: x,pin_memory=False,
                                               batch_size=_args.train_batch_size, shuffle=True, num_workers=_args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_set,collate_fn=lambda x: x,pin_memory=False,
                                             batch_size=_args.val_batch_size, shuffle=False, num_workers=_args.num_workers)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=_args.lr)

    # Summary writer and Checkpoint manager
    train_writer = SummaryWriter(os.path.join(_log_path, 'train'), flush_secs=10)
    val_writer = SummaryWriter(os.path.join(_log_path, 'val'), flush_secs=10)
    saver = CheckPointManager(os.path.join(_log_path, 'ckpt', 'model'), keep_checkpoint_every_n_hours=0.5)
    if _args.resume is not None:
        global_step = saver.load(_args.resume, model, optimizer)

    # trainings
    torch.autograd.set_detect_anomaly(_args.debug)
    model.train()

    steps_per_epoch = len(train_loader)
    if _args.summary_every < 0:
        _args.summary_every = abs(_args.summary_every) * steps_per_epoch
    if _args.validate_every < 0:
        _args.validate_every = abs(_args.validate_every) * steps_per_epoch

    for epoch in range(0, _args.epochs):
        _logger.info('Begin epoch {} (steps {} - {})'.format(epoch, global_step, global_step + len(train_loader)))
        tbar = tqdm(total=len(train_loader), ncols=100)
        for train_data in train_loader:

            global_step += 1

            optimizer.zero_grad()

            # Forward through neural network
            # dict_all_to_device(train_data, _device)

            train_data = train_data
            train_data = train_data[0]
            pcd0 = train_data[0]
            pcd1 = train_data[1]
            T_gt = train_data[2][:3, :]
            T_gt = torch.tensor(T_gt.reshape(1, 3, 4)).type(torch.float32).to(_device)

            down0 = prepare_dataset(pcd0)
            down1 = prepare_dataset(pcd1)
            xyz_src, norm_src = downsample(down0)
            xyz_ref, norm_ref = downsample(down1)
            pred_transforms, endpoints = model(xyz_src, xyz_ref, norm_src, norm_ref, _args.num_reg_iter)
            # pred_transforms, endpoints = model(train_data, _args.num_train_reg_iter)  # Use less iter during training

            # Compute loss, and optimize
            train_losses = compute_losses(xyz_src, pred_transforms, endpoints, T_gt,
                                          loss_type=_args.loss_type, reduction='mean')
            if _args.debug:
                with TorchDebugger():
                    train_losses['total'].backward()
            else:
                train_losses['total'].backward()
            optimizer.step()

            tbar.set_description('Loss:{:.3g}'.format(train_losses['total']))
            tbar.update(1)

            if global_step % _args.summary_every == 0:  # Save tensorboard logs
                save_summaries(train_writer, data=train_data, predicted=pred_transforms, endpoints=endpoints,
                               losses=train_losses, step=global_step)

            if global_step % _args.validate_every == 0:  # Validation loop. Also saves checkpoints
                model.eval()
                val_score = validate(val_loader, model, val_writer, global_step)
                saver.save(model, optimizer, step=global_step, score=val_score)
                model.train()

        tbar.close()

    _logger.info('Ending training. Number of steps = {}.'.format(global_step))


if __name__ == '__main__':
    main()
