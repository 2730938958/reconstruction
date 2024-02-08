from PIL import Image
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from gen_ply import generate
from module.sttr import STTR
from dataset.preprocess import normalization, compute_left_occ_region
from utilities.misc import NestedTensor
from memory_profiler import profile
sys.path.append('../') # add relative path

# Default parameters
args = type('', (), {})() # create empty args
args.channel_dim = 128
args.position_encoding='sine1d_rel'
args.num_attn_layers=6
args.nheads=8
args.regression_head='ot'
args.context_adjustment_layer='cal'
args.cal_num_blocks=8
args.cal_feat_dim=16
args.cal_expansion_ratio=4


#模型加载
model = STTR(args).cuda().eval()
#加载预训练模型
model_file_name = "pre/kitti_finetuned_model.pth.tar"
checkpoint = torch.load(model_file_name)
pretrained_dict = checkpoint['state_dict']
model.load_state_dict(pretrained_dict, strict=False)

#读入图像数据
left = np.array(Image.open('data/KITTI_disp/training/image_2/000046_10.png'))
right = np.array(Image.open('data/KITTI_disp/training/image_3/000046_10.png'))

#normalize
input_data = {'left': left, 'right':right}
input_data = normalization(**input_data)

#下采样attention
h, w, _ = left.shape
bs = 1
downsample = 3
col_offset = int(downsample / 2)
row_offset = int(downsample / 2)
sampled_cols = torch.arange(col_offset, w, downsample)[None,].expand(bs, -1).cuda()
sampled_rows = torch.arange(row_offset, h, downsample)[None,].expand(bs, -1).cuda()

#调整输入格式
input_data = NestedTensor(input_data['left'].cuda()[None,],input_data['right'].cuda()[None,],
                          sampled_cols=sampled_cols, sampled_rows=sampled_rows)

#模型计算
output = model(input_data)

#产生深度图并调用点云生成接口
disp_pred = output['disp_pred'].data.cpu().numpy()[0]
occ_pred = output['occ_pred'].data.cpu().numpy()[0] > 0.5
disp_pred[occ_pred] = 0.0

generate(disp_pred)

