#!/usr/bin/env python3
"""Latest-only compressed UI preview. Never modifies the source camera topic."""
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


class CameraPreview(Node):
    def __init__(self):
        super().__init__('orchid_ui_camera_preview')
        self.declare_parameter('input_topic', os.getenv('ORCHID_UI_CAMERA_INPUT', '/remote/head_rgbd_sensor/rgb/image_rect_color'))
        # self.declare_parameter('input_topic', os.getenv('ORCHID_UI_CAMERA_INPUT', '/head_rgbd_sensor/rgb/image_rect_color'))
        self.declare_parameter('output_topic', os.getenv('ORCHID_UI_CAMERA_OUTPUT', '/orchid_ui/camera/compressed'))
        self.latest = None
        self.last_encoded = 0.
        self.publisher = self.create_publisher(CompressedImage, self.get_parameter('output_topic').value, QoSProfile(depth=1))
        self.subscription = None
        self.demand_timer = self.create_timer(.5, self.sync_subscription,
            clock=Clock(clock_type=ClockType.STEADY_TIME))

    def sync_subscription(self):
        """Stop receiving raw images when no browser needs the preview."""
        if self.publisher.get_subscription_count():
            if self.subscription is None:
                self.subscription = self.create_subscription(Image, self.get_parameter('input_topic').value, self.receive,
                    QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        elif self.subscription is not None:
            self.destroy_subscription(self.subscription)
            self.subscription = None
            self.latest = None

    def receive(self, message):
        if not self.publisher.get_subscription_count():
            self.latest = None
            return
        self.latest = message
        now = time.monotonic()
        if now - self.last_encoded < .5:
            return
        self.last_encoded = now
        try:
            preview = encode_preview(self.latest)
            self.publisher.publish(preview)
        except (ValueError, cv2.error) as exc:
            self.get_logger().warning(f'Camera preview rejected: {exc}', throttle_duration_sec=5.)
        finally:
            self.latest = None


def encode_preview(message):
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(message.encoding)
    if not channels or message.step < message.width * channels:
        raise ValueError(f'unsupported image encoding/stride: {message.encoding}')
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, :message.width * channels].reshape(message.height, message.width, channels)
    if message.encoding == 'rgb8': image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif message.encoding == 'rgba8': image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif message.encoding == 'bgra8': image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif channels == 1: image = image[:, :, 0]
    if message.width > 640:
        image = cv2.resize(image, (640, max(1, round(message.height * 640 / message.width))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok: raise ValueError('JPEG encoding failed')
    result = CompressedImage()
    result.header = message.header
    result.format = 'jpeg'
    result.data = encoded.tobytes()
    return result


def main():
    rclpy.init()
    node = CameraPreview()
    try: rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__': main()
