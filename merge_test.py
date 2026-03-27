import os
import sys
import numpy as np

def full_demo(lib_dir, camera_data_dir):
    sys.path.append(os.path.join(lib_dir, 'graspnet-baseline'))
    sys.path.append(os.path.join(lib_dir, 'lingbot-depth'))

    from lingbot_test import treatData
    from anygrasp_test import graspDemo
    import open3d as o3d
    import argparse
    import scipy.io as scio
    from PIL import Image

    import torch
    import cv2
    import importlib


    # data_dir = 'CAMERA_DATA'
    treatData(lib_dir, camera_data_dir)
    graspDemo(camera_data_dir)


if __name__=='__main__':
    lib_dir = ''
    camera_data_dir = 'CAMERA_DATA'
    full_demo(lib_dir, camera_data_dir)