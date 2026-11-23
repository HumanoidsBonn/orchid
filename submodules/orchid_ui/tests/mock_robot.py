#!/usr/bin/env python3
import time
import math
import json
import threading
import io
from PIL import Image, ImageDraw

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage

# Import the custom interfaces we just compiled
from orchid_ros_interfaces.action import ExecuteTask

class MockRobotNode(Node):
    def __init__(self):
        super().__init__('mock_robot_node')
        self.get_logger().info("Initializing Mock Robot Node...")
        
        self.robot_id = 'example_bot'
        self.cb_group = ReentrantCallbackGroup()
        
        # 1. Telemetry and State Publishers
        self.state_pub = self.create_publisher(String, f'{self.robot_id}/gui_state', 10)
        self.log_pub = self.create_publisher(String, f'{self.robot_id}/gui_logs', 10)
        self.pose_pub = self.create_publisher(PoseStamped, f'{self.robot_id}/pose', 10)
        self.tree_pub = self.create_publisher(String, f'{self.robot_id}/behavior_tree', 10)
        self.cam_pub = self.create_publisher(CompressedImage, f'{self.robot_id}/camera/compressed', 10)
        
        # 2. Action Server (for long-running tasks requested by GUI)
        self._action_server = ActionServer(
            self,
            ExecuteTask,
            f'{self.robot_id}/execute_task',
            execute_callback=self.execute_callback,
            callback_group=self.cb_group,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        
        # State variables
        self.battery_level = 99.0
        self.robot_status = "Idle"
        self.active_node = "Root Sequence"
        
        # Start background threads for telemetry simulation
        self.running = True
        self.telemetry_thread = threading.Thread(target=self.telemetry_loop, daemon=True)
        self.pose_thread = threading.Thread(target=self.pose_loop, daemon=True)
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.tree_thread = threading.Thread(target=self.tree_loop, daemon=True)
        
        self.telemetry_thread.start()
        self.pose_thread.start()
        self.camera_thread.start()
        self.tree_thread.start()
        
        self.publish_log("info", "Mock robot node started. Subscriptions and Action server online.")

    # Goal callback for action server
    def goal_callback(self, goal_request):
        self.get_logger().info(f"Received goal request: {goal_request.task_description}")
        return GoalResponse.ACCEPT

    # Cancel callback for action server
    def cancel_callback(self, goal_handle):
        self.get_logger().info("Received cancel request for goal")
        return CancelResponse.ACCEPT

    # Main action executor callback
    def execute_callback(self, goal_handle):
        task_desc = goal_handle.request.task_description
        self.get_logger().info(f"Executing goal task: {task_desc}")
        self.publish_log("info", f"Executing task: '{task_desc}'")
        
        feedback_msg = ExecuteTask.Feedback()
        
        # Step 1: Navigating
        self.robot_status = "Navigating"
        feedback_msg.current_action = "Navigating to Location"
        feedback_msg.percent_completed = 10.0
        goal_handle.publish_feedback(feedback_msg)
        self.publish_log("info", "Navigation active. Moving toward target...")
        time.sleep(2.5)
        
        feedback_msg.percent_completed = 35.0
        goal_handle.publish_feedback(feedback_msg)
        time.sleep(2.0)

        # Step 2: Request Follow-up User Input mid-execution!
        self.robot_status = "Waiting for Input"
        feedback_msg.current_action = "Awaiting user input"
        feedback_msg.percent_completed = 50.0
        goal_handle.publish_feedback(feedback_msg)
        
        self.publish_log("warn", "Cupboard door is locked. Requesting GUI clarification...")
        
        # Call the GUI service synchronously (blocks execution)
        prompt = "The cupboard door appears to be locked. Should I wait, retry, or bypass?"
        user_response = self.request_user_input_from_gui(prompt)
        
        self.publish_log("info", f"Received reply from HRI GUI: '{user_response}'")
        
        # Resume task
        self.robot_status = "Grasping"
        feedback_msg.current_action = f"Proceeding based on input: {user_response}"
        feedback_msg.percent_completed = 65.0
        goal_handle.publish_feedback(feedback_msg)
        self.publish_log("info", "Executing action: grasping the target item...")
        time.sleep(3.0)
        
        feedback_msg.current_action = "Returning to base"
        feedback_msg.percent_completed = 90.0
        goal_handle.publish_feedback(feedback_msg)
        time.sleep(2.0)

        # Done!
        self.robot_status = "Idle"
        goal_handle.succeed()
        
        result = ExecuteTask.Result()
        result.success = True
        result.result_message = f"Successfully completed: {task_desc} (clarified by '{user_response}')"
        self.publish_log("info", "Task goal successfully achieved.")
        return result

    # REST call to the web server, which brokers the follow-up prompt to the
    # browser GUI and blocks until the user's answer comes back. This replaces
    # the earlier ROS2 service, whose response callback deadlocked the action
    # executor while this callback was running.
    def request_user_input_from_gui(self, prompt: str) -> str:
        import urllib.request

        url = "http://localhost:8000/api/followup"
        payload = json.dumps({"prompt": prompt}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=130) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            response = data.get("response", "timeout")
            self.get_logger().info(f"[followup] got response: {response}")
            return response
        except Exception as e:
            self.get_logger().error(f"[followup] REST call failed: {e}")
            return "timeout"

    # Telemetry data publisher (battery, speeds, diagnostics)
    def telemetry_loop(self):
        while self.running:
            # Drain battery slowly
            self.battery_level -= 0.15
            if self.battery_level <= 10.0:
                self.battery_level = 99.0
            
            # Pack status JSON
            state_json = {
                "battery": round(self.battery_level, 1),
                "robot_status": self.robot_status,
                "eta": "1m 45s" if self.robot_status != "Idle" else "N/A",
                "active_node": self.active_node,
                "joint_states": {
                    "linear_speed": 0.45 if self.robot_status == "Navigating" else 0.0,
                    "angular_speed": 0.15 if self.robot_status == "Navigating" else 0.0
                }
            }
            
            msg = String()
            msg.data = json.dumps(state_json)
            self.state_pub.publish(msg)
            time.sleep(1.0)

    # Robot Pose publisher (draws a circle on the canvas)
    def pose_loop(self):
        t = 0.0
        while self.running:
            # Circle radius 2 meters, centered at (0, 0)
            radius = 2.2
            x = radius * math.cos(t)
            y = radius * math.sin(t)
            yaw = t + math.pi / 2.0  # tangent direction
            
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position.x = x
            pose_msg.pose.position.y = y
            pose_msg.pose.position.z = 0.0
            
            # Euler to Quaternion for yaw
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = math.sin(yaw / 2.0)
            pose_msg.pose.orientation.w = math.cos(yaw / 2.0)
            
            self.pose_pub.publish(pose_msg)
            
            t += 0.04
            if t > 2 * math.pi:
                t = 0.0
            time.sleep(0.1)

    # Camera Stream simulation (Generates PIL frames and encodes to JPEG)
    def camera_loop(self):
        w, h = 320, 240
        frame_idx = 0
        
        while self.running:
            # Create a simple dynamic pattern image using PIL
            img = Image.new('RGB', (w, h), color='#1e293b')
            draw = ImageDraw.Draw(img)
            
            # Draw moving scanner bar
            bar_x = (frame_idx * 5) % w
            draw.rectangle([bar_x, 0, bar_x + 10, h], fill='#3b82f6')
            
            # Draw center pulsing circle
            pulse_radius = int(25 + 10 * math.sin(frame_idx * 0.2))
            draw.ellipse([w//2 - pulse_radius, h//2 - pulse_radius, w//2 + pulse_radius, h//2 + pulse_radius], outline='#a855f7', width=3)
            
            # Draw text timestamp
            draw.text((10, 10), "CAMERA 01 - ACTIVE", fill='#ffffff')
            draw.text((10, 25), f"T: {time.strftime('%H:%M:%S')}", fill='#10b981')
            
            # Save frame to buffer
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=70)
            jpeg_bytes = buffer.getvalue()
            
            # Publish CompressedImage
            cam_msg = CompressedImage()
            cam_msg.header.stamp = self.get_clock().now().to_msg()
            cam_msg.format = 'jpeg'
            cam_msg.data = jpeg_bytes
            
            self.cam_pub.publish(cam_msg)
            
            frame_idx += 1
            time.sleep(0.1)  # 10 FPS stream

    # Behavior Tree simulator
    def tree_loop(self):
        while self.running:
            # Switch node statuses depending on robot action
            tree_data = {
                "name": "Mission Root",
                "status": "RUNNING" if self.robot_status != "Idle" else "SUCCESS",
                "children": [
                    {
                        "name": "Check Battery",
                        "status": "SUCCESS"
                    },
                    {
                        "name": "Pick Task",
                        "status": "RUNNING" if self.robot_status != "Idle" else "SUCCESS",
                        "children": [
                            {
                                "name": "Navigate",
                                "status": "SUCCESS" if self.robot_status in ["Waiting for Input", "Grasping", "Idle"] and self.robot_status != "Idle" else ("RUNNING" if self.robot_status == "Navigating" else "INVALID")
                            },
                            {
                                "name": "Clarify Goal",
                                "status": "SUCCESS" if self.robot_status in ["Grasping", "Idle"] and self.robot_status != "Idle" else ("RUNNING" if self.robot_status == "Waiting for Input" else "INVALID")
                            },
                            {
                                "name": "Grasp Object",
                                "status": "RUNNING" if self.robot_status == "Grasping" else ("SUCCESS" if self.robot_status == "Idle" else "FAILURE")
                            }
                        ]
                    }
                ]
            }
            
            # Determine active node text
            if self.robot_status == "Navigating":
                self.active_node = "Navigate"
            elif self.robot_status == "Waiting for Input":
                self.active_node = "Clarify Goal"
            elif self.robot_status == "Grasping":
                self.active_node = "Grasp Object"
            else:
                self.active_node = "Mission Root"
                
            msg = String()
            msg.data = json.dumps(tree_data)
            self.tree_pub.publish(msg)
            time.sleep(1.0)

    # Logging helper
    def publish_log(self, level, message):
        log_json = {
            "level": level,
            "message": message,
            "timestamp": time.time()
        }
        msg = String()
        msg.data = json.dumps(log_json)
        self.log_pub.publish(msg)

    def shutdown(self):
        self.running = False
        self.telemetry_thread.join()
        self.pose_thread.join()
        self.camera_thread.join()
        self.tree_thread.join()

def main(args=None):
    rclpy.init(args=args)
    mock_robot = MockRobotNode()
    
    # Reentrant callbacks require MultiThreadedExecutor to handle service requests during action execution
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(mock_robot)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        mock_robot.get_logger().info("Shutting down mock robot node.")
    finally:
        mock_robot.shutdown()
        executor.shutdown()
        mock_robot.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
