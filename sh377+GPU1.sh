#!/bin/bash

echo "Start running Python scripts..."

CUDA_VISIBLE_DEVICES=1 nohup python train.py --cfg ./configs/human_nerf/zju_mocap/377/adventure.yaml > 377_split_train 2>&1 &  
wait
echo "377trained , evaling"
CUDA_VISIBLE_DEVICES=1 nohup python eval-excel.py --cfg ./configs/human_nerf/zju_mocap/377/adventure.yaml > 377_split_eval 2>&1 &  
wait
echo "command 377 done"




