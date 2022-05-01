#################################################################################
#   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.          #
#                                                                               #
#   Licensed under the Apache License, Version 2.0 (the "License").             #
#   You may not use this file except in compliance with the License.            #
#   You may obtain a copy of the License at                                     #
#                                                                               #
#       http://www.apache.org/licenses/LICENSE-2.0                              #
#                                                                               #
#   Unless required by applicable law or agreed to in writing, software         #
#   distributed under the License is distributed on an "AS IS" BASIS,           #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#   See the License for the specific language governing permissions and         #
#   limitations under the License.                                              #
#################################################################################

"""
object_detection_node.py
This module creates the object_detection_node which is responsible for collecting
sensor data (camera images) from sensor_fusion_pkg and running object detection,
on specified object, providing normalized delta from target for
ftl_navigation_pkg.
The node defines:
    image_subscriber: A subscriber to the /sensor_fusion_pkg/sensor_msg published
                      by the sensor_fusion_pkg with sensor data.
    display_image_publisher: A publisher to publish the Image message using
                             web_video_server.
    delta_publisher: A publisher to publish the normalized error (delta) of the
                     detected object from the target (reference) position
                     with respect to x and y axes.
"""
import time
import signal
import threading
import cv2
import numpy as np
from pyzbar import pyzbar
from pyzbar.pyzbar import decode 
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (QoSProfile,
                       QoSHistoryPolicy,
                       QoSReliabilityPolicy)
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from deepracer_interfaces_pkg.msg import (EvoSensorMsg,
                                          DetectionDeltaMsg)
from openvino.inference_engine import IECore
import ngraph as ng
from object_detection_pkg import (constants,
                                  utils)


class ObjectDetectionNode(Node):
    """Node responsible for collecting sensor data (camera images) from sensor_fusion_pkg
       and running object detection on specified object, providing normalized delta from target for
       ftl_navigation_pkg.
    """

    def __init__(self, qos_profile):
        """Create a ObjectDetectionNode.
        """
        super().__init__('object_detection_node')
        self.get_logger().info("object_detection_node started.")

        # Double buffer to hold the input images for inference.
        self.input_buffer = utils.DoubleBuffer(clear_data_on_get=True)
        # Get DEVICE parameter (CPU/MYRIAD) from launch file.
        self.declare_parameter("DEVICE")
        self.device = self.get_parameter("DEVICE").get_parameter_value().string_value
        if not self.device:
            self.device = constants.DEVICE
        # Check if the inference output needs to be published to localhost using web_video_server
        self.declare_parameter("PUBLISH_DISPLAY_OUTPUT")
        self.publish_display_output = \
            self.get_parameter("PUBLISH_DISPLAY_OUTPUT").get_parameter_value().bool_value
        self.get_logger().info(f"Publish output set to {self.publish_display_output}")
        # Initialize Intel Inference Engine
        self.init_network()
        # Calculate target position for bounding box center.
        self.target_x, self.target_y = self.calculate_target_center(self.w, self.h)

        # Create subscription to sensor messages from camera.
        self.image_subscriber = self.create_subscription(EvoSensorMsg,
                                                         constants.SENSOR_FUSION_TOPIC,
                                                         self.on_image_received_cb,
                                                         qos_profile)

        # Creating publisher for display_image.
        self.display_image_publisher = \
            self.create_publisher(Image,
                                  constants.DISPLAY_IMAGE_PUBLISHER_TOPIC,
                                  10)

        # Creating publisher for error (delta) from target bb position.
        self.delta_publisher = self.create_publisher(DetectionDeltaMsg,
                                                     constants.DELTA_PUBLISHER_TOPIC,
                                                     qos_profile)
        self.bridge = CvBridge()


        # Launching a separate thread to run inference.
        self.stop_thread = False
        self.thread_initialized = False
        self.thread = threading.Thread(target=self.run_inference)
        self.thread.start()
        self.thread_initialized = True
        self.get_logger().info(f"Waiting for input images on {constants.SENSOR_FUSION_TOPIC}")

    def init_network(self):
        """Function which initializes Intel Inference Engine.
        """
        # Load OpenVINO Inference Engine.
        self.get_logger().info(f"Loading Inference Engine on {self.device}")
        self.ie = IECore()

        # Read and load the network.
        self.net = self.ie.read_network(model=constants.MODEL_XML, weights=constants.MODEL_BIN)
        self.func = ng.function_from_cnn(self.net)
        self.ops = self.func.get_ordered_ops()
        self.exec_net = self.ie.load_network(network=self.net, device_name=self.device)

        # Read expected input image info from network and prepare input blobs.
        # n: batch size, c: no. of channels, h: input height, w: input width
        for self.input_key in self.net.input_info:
            self.input_name = self.input_key
            self.n, self.c, self.h, self.w = self.net.input_info[self.input_key].input_data.shape
        # Initializing to float for optimizing in later functions
        self.h = float(self.h)
        self.w = float(self.w)

        # Prepare output blobs
        self.out_blob = next(iter(self.net.outputs))

    def wait_for_thread(self):
        """Function which joins the created background thread.
        """
        if self.thread_initialized:
            self.thread.join()
            self.get_logger().info("Thread joined")

    def thread_shutdown(self):
        """Function which sets the flag to shutdown background thread.
        """
        self.stop_thread = True

    def on_image_received_cb(self, sensor_data):
        """Call back for adding to the input double buffer whenever
           new sensor image is received from sensor_fusion_node.
        Args:
            sensor_data (EvoSensorMsg): Message containing sensor images and lidar data.
        """
        self.input_buffer.put(sensor_data)

    def preprocess(self, sensor_data):
        """Method that preprocesses the input data to be provided for inference to network.
        Args:
            sensor_data (EvoSensorMsg): Contains sensor images and lidar data.
        Returns:
            image: Preprosessed image expected by the network.
        """
        image = self.bridge.imgmsg_to_cv2(sensor_data.images[0])
        ih, iw = image.shape[:-1]
        # Resize to required input size
        if (ih, iw) != (int(self.h), int(self.w)):
            image = cv2.resize(image, (int(self.w), int(self.h)))
        # Change data layout from HWC to CHW.
        image = image.transpose((2, 0, 1))
        return image

    def calculate_target_center(self,image_width, image_height):
        target_x = float(image_width)/2.0
        target_y = float(image_height)/3.0
        return target_x, target_y

    def calculate_delta(self, target_x, target_y, center_x, center_y, width, height):
        """Method that calculates the normalized error (delta) of the
           detected object from the target (reference) position
           with respect to x and y axes.
        Args:
            target_x (float): Target x co-ordinate.
            target_y (float): Target y co-ordinate.
            bb_center_x (float): x co-ordinate of center of detected bounding box.
            bb_center_y (float): y co-ordinate of center of detected bounding box.
        Returns:
            delta (DetectionDeltaMsg): Normalized Error (delta) in x and y respectively
            returned as a list of floats and converted to ObjectDetectionErrorMsg.
        """
        delta_x = (center_x - target_x) / self.w
        delta_y = (center_y - target_y) / self.h
        delta = DetectionDeltaMsg()
        delta.delta = [delta_x, delta_y,float(center_x),float(center_y), float(width), float(height)]
        self.get_logger().debug(f"Delta from target position: {delta_x} {delta_y}")
        return delta

    def read_barcode(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        barcodes = decode(gray)
        bb_center_x, bb_center_y, width, height = 0, 0, 0, 0
        x, y, w, h = 0, 0, 0, 0
        if len(barcodes)==0:
            detected = False
        else:
            for barcode in barcodes:
                (x, y, w, h) = barcode.rect
                detected = True
                if w>=width or h>=height:
                     bb_center_x, bb_center_y = int(x+w/2), int(y+h/2)
            self.get_logger().info(f"Saw smg!: {bb_center_x} {bb_center_y}") 
        return (detected, bb_center_x, bb_center_y, w, h)
    
    def show_barcodes(self, frame, x_pos, y_pos, width, height):
        cv2.rectangle(frame, (x_pos,y_pos), (x_pos+width,y_pos+height), (232, 35, 244),2)
        cv2.circle(frame, (int(self.target_x), int(self.target_y)),
                   5,
                   (0, 255, 0),
                   -1)
        display_image = self.bridge.cv2_to_imgmsg(np.array(frame), "bgr8")
        self.display_image_publisher.publish(display_image)

    def run_inference(self):
        """Method for running inference on received input image.
        """
        try:
            while not self.stop_thread:
                # Get an input image from double buffer.
                sensor_data = self.input_buffer.get()
                start_time = time.time()

                frame = self.preprocess(sensor_data).transpose((1, 2, 0))
                detected, bb_center_x, bb_center_y, width, height = self.read_barcode(frame)

                if detected:
                    delta = self.calculate_delta(self.target_x, self.target_y, bb_center_x, bb_center_y, width, height)
                    self.delta_publisher.publish(delta)
                else:
                    delta = self.calculate_delta(self.target_x, self.target_y, self.target_x, self.target_y, 0, 0)
                    self.delta_publisher.publish(delta)
                    
                #if self.publish_display_output:
                #    self.show_barcodes(frame, bb_center_x, bb_center_y, width, height)
                
                self.get_logger().info(f"Total execution time = {time.time() - start_time}")
        except Exception as ex:
            self.get_logger().error(f"Failed inference step: {ex}")
            # Destroy the ROS Node running in another thread as well.
            self.destroy_node()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    qos = QoSProfile(reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
                     depth=1,
                     history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST)

    try:
        object_detection_node = ObjectDetectionNode(qos)
        executor = MultiThreadedExecutor()

        def signal_handler(signum, frame):
            """Callback function to handle registered signal handler
               to join and stop executing running thread created.
            Args:
                signum: The signal number
                frame: the current stack frame (None or a frame object)
            """
            object_detection_node.get_logger().info("Signal Handler initiated")
            object_detection_node.thread_shutdown()
            object_detection_node.wait_for_thread()

        # Register SIGINT handler
        signal.signal(signal.SIGINT, signal_handler)

        rclpy.spin(object_detection_node, executor)

    except Exception as ex:
        object_detection_node.get_logger().error(f"Exception in Object Detection Node: {ex}")
        object_detection_node.destroy_node()
        rclpy.shutdown()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    object_detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
