import os
import sys
import os.path as osp
import numpy as np
import argparse
import json
import cv2
import body_model
import glob
from smpl_numpy import SMPL
import pickle

try:
    import smplx
    import torch
except ImportError:
    print('smplx and torch are needed for visualization of SMPL vertices.')


def rm(root_dir, outputpath=None, view_id=0, 
              visualize_smpl=False, smpl_model_path=None):
    # load color image
    png_files = glob.glob(os.path.join(root_dir, 'images','*.png'))
    png_files.sort(key=lambda x:int(x[-8:-4]))
    skip=8
    # 遍历文件列表，每12张图片取一张保留，其余的删除
    for i, file_name in enumerate(png_files):
        if i % skip == 0:
            continue
        # file_path = os.path.join(root_dir,'images', file_name)
        # print(file_name)
        os.remove(file_name)

    
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str,
                        help='root directory in which data is stored.')
    args = parser.parse_args()

    rm(
        root_dir=args.root_dir,
        # view_id=args.view_id,
        # visualize_smpl=args.visualize_smpl,
    )
