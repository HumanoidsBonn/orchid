#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


class TFFilter(Node):
    def __init__(self):
        super().__init__('recorded_tf_filter')

        self.pub = self.create_publisher(TFMessage, '/tf', 100)

        self.sub = self.create_subscription(
            TFMessage,
            '/tf_bag',
            self.callback,
            100
        )

        self.removed = 0

    def callback(self, msg):
        out = TFMessage()

        for tf in msg.transforms:
            parent = tf.header.frame_id.lstrip('/')
            child = tf.child_frame_id.lstrip('/')

            if parent == 'map' and child == 'odom':
                self.removed += 1
                continue

            out.transforms.append(tf)

        if out.transforms:
            self.pub.publish(out)


def main():
    rclpy.init()
    node = TFFilter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    print(f"Removed {node.removed} map->odom transforms")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
