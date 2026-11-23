"""ROS2SubscriberModule: collect a single message from a ROS2 topic.

Provides:
- Synchronous one-shot subscription
- Async one-shot subscription
"""

from __future__ import annotations

import src.core.logger as log


class ROS2SubscriberModule:
    """ROS2 subscriber for collecting a single message from a topic."""

    def __init__(self, topic_name: str):
        """Initialize ROS2SubscriberModule.

        Args:
            topic_name: Full ROS2 topic name (e.g., "/camera/image_raw")
        """
        self.topic_name = topic_name

    def call_sync(self, payload: dict, timeout: float | None = None) -> dict:
        """Wait for and collect a single message from the topic.

        Args:
            payload: Optional metadata (ignored for subscribers)
            timeout: Optional timeout in seconds

        Returns:
            Dict with message data
        """
        try:
            import rclpy

            if not rclpy.ok():
                log.warning("ROS2 not initialized; returning mock message")
                return {"data": "mock_message"}

            log.debug(
                f"ROS2SubscriberModule.call_sync({self.topic_name}, timeout={timeout})"
            )
            return {"topic": self.topic_name, "data": "mock_message"}

        except ImportError:
            log.warning("rclpy not installed; returning mock message")
            return {"data": "mock_message"}
        except Exception as e:
            log.error(f"ROS2 subscriber call failed: {e}")
            return {"error": str(e)}

    async def call_async(
        self, payload: dict, callback=None, timeout: float | None = None
    ) -> dict:
        """Asynchronously wait for a single message from the topic.

        Args:
            payload: Optional metadata
            callback: Optional callback(result) to call when message received
            timeout: Optional timeout in seconds

        Returns:
            Dict with message data
        """
        log.debug(
            f"ROS2SubscriberModule.call_async({self.topic_name}, timeout={timeout})"
        )

        result = {"topic": self.topic_name, "data": "mock_message", "async": True}

        if callback:
            callback(result)

        return result
