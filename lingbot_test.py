import torch
import cv2
import numpy as np
import importlib
import os

scale = 1000.0

def treatData(lib_dir, camera_data_dir): # colorimage, depthimage, intrinsics):
    # Import the module as a string
    v2 = importlib.import_module(os.path.join(lib_dir, 'lingbot-depth.mdm.model.v2'))

    # Load model
    odevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = v2.MDMModel.from_pretrained('robbyant/lingbot-depth-pretrain-vitl-14-v0.5').to(odevice)

    # Load and prepare inputs
    image = cv2.cvtColor(cv2.imread(os.path.join(camera_data_dir, 'color.png')), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    image = torch.tensor(image / 255, dtype=torch.float32, device=odevice).permute(2, 0, 1)[None]

    depth = cv2.imread(os.path.join(camera_data_dir, 'raw_depth.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / scale
    depth = torch.tensor(depth, dtype=torch.float32, device=odevice)[None]

    intrinsics = np.loadtxt(os.path.join(camera_data_dir, 'intrinsics.txt'))
    intrinsics[0] /= w  # Normalize fx and cx by width
    intrinsics[1] /= h  # Normalize fy and cy by height
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=odevice)[None]

    # Run inference
    output = model.infer(
        image,
        depth_in=depth,
        use_fp16=False, # because pallas GPU is too old, doesn't support ADA, fp16
        intrinsics=intrinsics)

    depth_pred = output['depth'].squeeze().cpu().numpy()  # Refined depth map - data is in meters
    depth_pred = (depth_pred.astype(np.float32) * scale).astype(np.uint16)
    # Replace invalid values with 0
    depth_pred = np.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0)
    
    points = output['points']      # 3D point cloud
    treated_depth_image_path = os.path.join(camera_data_dir, 'depth.png') # CV_16U
    treated_depth_success = cv2.imwrite(treated_depth_image_path, depth_pred)
    if not treated_depth_success:
        print("color_success: Error saving image")

    return points


if __name__=='__main__':
    treatData('CAMERA_DATA')


