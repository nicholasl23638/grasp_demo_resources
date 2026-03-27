import sys
import os
import rclpy
from rclpy.node import Node
sys.path.append("lib/MARCH25")
from merge_test import full_demo

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import qos_profile_sensor_data

class DepthImageSaver(Node):
    def __init__(self):
        super().__init__('depth_image_saver')
        # Subscribe to the depth image topic
        # The topic might be different based on your launch file, check with `ros2 topic list`
        #qos_profile = QoSProfile(
        #    reliability=ReliabilityPolicy.RELIABLE, # Key setting for best effort
        #    durability=DurabilityPolicy.TRANSIENT_LOCAL,
        #    history=HistoryPolicy.KEEP_LAST,
        #)
        self.depth_subscription = self.create_subscription(
            Image,
            '/gripper_camera/depth/image_rect_raw', #'/gripper_camera/aligned_depth_to_color/image_raw', # Common topic for raw depth, adjust if needed
            self.depth_callback,
            qos_profile_sensor_data)
        self.color_subscription = self.create_subscription(
            Image,
            '/gripper_camera/color/image_rect_raw',
            self.color_callback,
            qos_profile_sensor_data
            )
        timer_period = 3.0
        # Create a timer that calls the 'timer_callback' function every 'timer_period' seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.bridge = CvBridge()
        self.get_logger().info('Depth Image Saver Node Started')
        self.color_image = None
        self.depth_image = None
        self.depth_image_arr = []
        self.depth_image_count = 0

    def color_callback(self, msg):
        self.get_logger().info('Received color image...')
        if self.color_image is not None:
            self.get_logger().info('Already have color image - destroying subscriber...')
            self.destroy_subscription(self.color_subscription)
            return
        try:
            # Convert ROS Image message to OpenCV image (NumPy array).
            # D435i color images typically use "bgr8" encoding in ROS.
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
            return
        # OpenCV uses BGR color order, but PIL uses RGB. We must convert the color space.
        #rgb_image = cv_image #cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        detection_box_image = cv_image #cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)
        # Convert the NumPy array to a PIL Image
        self.color_image = detection_box_image

    def depth_callback(self, msg):
        self.get_logger().info('Received depth image...')
        if self.depth_image is not None:
            self.get_logger().info('Already have depth image - destroying subscriber...')
            self.destroy_subscription(self.depth_subscription)
            return

        try:
            # Convert ROS Image message to OpenCV image
            # '16UC1' is the typical format for raw RealSense depth data (unsigned 16-bit, 1 channel)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        except Exception as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return
        # Save the raw 16-bit data as a 16UC1 grayscale PNG (preserves data)
        # The resulting image will look almost black/white as depth is > 255mm
        rgb_image = cv_image #cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        detection_box_image = rgb_image # cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)
        self.depth_image = detection_box_image
        
        self.get_logger().info('Finished processing depth image!')


    def timer_callback(self):
        """Method that is periodically called by the timer."""
        self.get_logger().info('Timer called...')
        if (self.color_image is not None and self.depth_image is not None):
            #self.get_logger().info(f'Height: {self.depth_image.shape[0]}')
            #self.get_logger().info(f'Width: {self.depth_image.shape[1]}')
            #self.get_logger().info(f'Test: {self.depth_image.shape[2]}')
            #self.get_logger().info(f'HeightColor: {self.color_image.shape[0]}')
            #self.get_logger().info(f'WidthColor: {self.color_image.shape[1]}')
            #self.get_logger().info(f'TestColor: {self.color_image.shape[2]}')
                        
            lib_dir = 'lib/MARCH25'
            camera_data_dir = os.path.join(lib_dir, 'CAMERA_DATA')
            
            color_image_path = os.path.join(camera_data_dir, 'color.png')
            color_success = cv2.imwrite(color_image_path, self.color_image)
            if not color_success:
                print("color_success: Error saving image")
            
            depth_image_path = os.path.join(camera_data_dir,'raw_depth.png')
            depth_success = cv2.imwrite(depth_image_path, self.depth_image)
            if not depth_success:
                print("depth_success: Error saving image")
            
            full_demo(lib_dir, camera_data_dir) #self.color_image, self.depth_image)

def main(args=None):
    rclpy.init(args=args)
    depth_image_saver = DepthImageSaver()
    try:
        rclpy.spin(depth_image_saver)
    except KeyboardInterrupt:
        pass
    finally:
        depth_image_saver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
