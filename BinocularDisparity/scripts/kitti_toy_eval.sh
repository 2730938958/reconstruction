#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --batch_size 1\
                --checkpoint kitti_toy_eval\
                --num_workers 1\
                --eval\
                --dataset kitti_toy\
                --dataset_directory data/KITTI_disp\
                --resume ./pre/kitti_finetuned_model.pth.tar
#                --resume sttr_light_sceneflow_pretrained_model.pth.tar
#                --resume kitti_finetuned_model.pth.tar