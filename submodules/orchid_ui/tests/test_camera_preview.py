import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from sensor_msgs.msg import Image

spec = importlib.util.spec_from_file_location('camera_preview', Path(__file__).parents[1] / 'scripts/camera_preview.py')
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


def frame(width=1280, height=720):
    message = Image(width=width, height=height, encoding='rgb8', step=width*3+4)
    rows = np.zeros((height, message.step), dtype=np.uint8)
    rows[:, :width*3] = np.tile([210, 40, 20], width)
    message.data = rows.tobytes()
    return message


class PreviewTests(unittest.TestCase):
    def test_raw_subscription_follows_preview_demand(self):
        node = SimpleNamespace(publisher=Mock(), subscription=None, latest='old frame',
            create_subscription=Mock(return_value='raw-subscription'), destroy_subscription=Mock(),
            get_parameter=Mock(return_value=SimpleNamespace(value='/camera')), receive=Mock())
        node.publisher.get_subscription_count.return_value = 0
        preview.CameraPreview.sync_subscription(node)
        node.create_subscription.assert_not_called()
        node.publisher.get_subscription_count.return_value = 2
        preview.CameraPreview.sync_subscription(node)
        preview.CameraPreview.sync_subscription(node)
        node.create_subscription.assert_called_once()
        node.publisher.get_subscription_count.return_value = 1
        preview.CameraPreview.sync_subscription(node)
        node.destroy_subscription.assert_not_called()
        node.publisher.get_subscription_count.return_value = 0
        preview.CameraPreview.sync_subscription(node)
        node.destroy_subscription.assert_called_once_with('raw-subscription')
        self.assertIsNone(node.subscription)
        self.assertIsNone(node.latest)

    def test_dimensions_stride_colour_and_compression(self):
        original = frame(); raw = bytes(original.data)
        compressed = preview.encode_preview(original)
        image = cv2.imdecode(np.frombuffer(compressed.data, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, (360, 640, 3))
        self.assertLess(abs(int(image[100, 100, 2])-210), 8)
        self.assertLess(len(compressed.data), len(raw)//10)
        self.assertEqual(bytes(original.data), raw)

    def test_subscribers_and_frame_rate_gate(self):
        publisher = Mock(); publisher.get_subscription_count.return_value = 0
        node = SimpleNamespace(publisher=publisher, latest=None, last_encoded=0.)
        message = frame(64, 48)
        with patch.object(preview, 'encode_preview', return_value='jpeg') as encode:
            preview.CameraPreview.receive(node, message); encode.assert_not_called()
            publisher.get_subscription_count.return_value = 1
            with patch.object(preview.time, 'monotonic', side_effect=[10.,10.1,10.2,10.5]):
                for _ in range(4): preview.CameraPreview.receive(node, message)
            self.assertEqual(encode.call_count, 2)
            self.assertEqual(publisher.publish.call_count, 2)

    def test_reject_unsupported_encoding(self):
        message = frame(); message.encoding = '16UC1'
        with self.assertRaises(ValueError): preview.encode_preview(message)


if __name__ == '__main__': unittest.main()
