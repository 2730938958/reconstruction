#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --epochs 10\
                --batch_size 1\
                --checkpoint kitti_ft\
                --num_workers 1\
                --dataset kitti\
                --dataset_directory sample_data/KITTI_2015\
                --ft\
                --resume sceneflow_pretrained_model.pth.tar