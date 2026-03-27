import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Create alignment object
align_to = rs.stream.color
align = rs.align(align_to)

try:
    # Wait for a coherent pair of frames
    frames = pipeline.wait_for_frames()
    
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        raise Exception("Could not get frames")

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Colorize depth map (convert 16-bit to 8-bit RGB)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # Save images
    cv2.imwrite(os.path.join('CAMERA_DATA','color_image.png'), color_image)
    cv2.imwrite(os.path.join('CAMERA_DATA', 'depth_image.png'), depth_colormap)
    cv2.imwrite(os.path.join('CAMERA_DATA', 'raw_depth.png'), depth_image)
    print("Saved color_image png and depth_image png")

finally:
    pipeline.stop()


