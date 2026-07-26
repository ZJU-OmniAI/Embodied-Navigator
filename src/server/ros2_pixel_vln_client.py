#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 Foxy client script that:
1) Subscribes to /cam{id}/image_raw (id=0..3).
2) Reads camera/vehicle poses from TF (frames: camera0-3, vehicle).
3) Calls the HTTP agent service (FastAPI) step-by-step:
   - /reset: set a new instruction
   - /step : upload obs + sensor_poses, get MOVE_TO pixel or STOP
4) Publishes the action to /pixel_vln_cmd.

Camera id mapping (given by user):
  camera0: back
  camera1: front
  camera2: left
  camera3: right

This is a single-file script (not a full ROS package). Run it with ROS2:
  ros2 run ...  (if you wrap it) or just python3 this file with env sourced.

Dependencies on robot machine:
  - rclpy, tf2_ros, sensor_msgs, std_msgs (ROS2 Foxy)
  - pillow, numpy (for JPEG/base64 encoding)
  - no requests dependency (uses urllib).
"""

from __future__ import annotations

import base64
import io
import json
import math
import threading
import time
import uuid
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String, Bool

import tf2_ros


SENSOR_ORDER = ("rgb_back", "rgb_front", "rgb_left", "rgb_right")
CAM_ID_TO_SENSOR = {
    0: "rgb_back",
    1: "rgb_front",
    2: "rgb_left",
    3: "rgb_right",
}
SENSOR_TO_CAM_ID = {v: k for k, v in CAM_ID_TO_SENSOR.items()}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    # Yaw around Z, ROS uses (x,y,z,w).
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _http_post_json(url: str, payload: Dict[str, Any], timeout_s: float = 10.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _ros_img_to_rgb8(msg: RosImage) -> np.ndarray:
    """
    Convert common ROS Image encodings to an HxWx3 uint8 RGB numpy array.
    Supports: rgb8, bgr8, rgba8, bgra8, mono8.
    """
    h = int(msg.height)
    w = int(msg.width)
    enc = (msg.encoding or "").lower()

    if enc in ("rgb8", "bgr8"):
        c = 3
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, c))
        if enc == "bgr8":
            arr = arr[..., ::-1]
        return arr

    if enc in ("rgba8", "bgra8"):
        c = 4
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, c))
        arr = arr[..., :3]
        if enc == "bgra8":
            arr = arr[..., ::-1]
        return arr

    if enc == "mono8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w))
        return np.repeat(arr[:, :, None], 3, axis=2)

    raise ValueError(f"Unsupported image encoding: {msg.encoding!r}")


def _encode_jpeg_b64(rgb: np.ndarray, quality: int = 85) -> str:
    rgb8 = rgb.astype(np.uint8, copy=False)
    im = Image.fromarray(rgb8, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class LatestImage:
    msg: RosImage
    ts_ms: int


class PixelVlnClientNode(Node):
    def __init__(self) -> None:
        super().__init__("dthink_pixel_vln_client")

        self.declare_parameter("api", "http://127.0.0.1:11451")
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("vehicle_frame", "vehicle")
        self.declare_parameter(
            "camera_frames", ["camera0", "camera1", "camera2", "camera3"]
        )
        self.declare_parameter("jpeg_quality", 85)
        self.declare_parameter("rate_hz", 0.1)

        self._api_base = str(self.get_parameter("api").value).rstrip("/")
        self._world_frame = str(self.get_parameter("world_frame").value)
        self._vehicle_frame = str(self.get_parameter("vehicle_frame").value)

        camera_frames_param = self.get_parameter("camera_frames").value
        if isinstance(camera_frames_param, str):
            camera_frames = [s.strip()
                             for s in camera_frames_param.split(",") if s.strip()]
        else:
            camera_frames = list(camera_frames_param)

        default_frames = ["camera0", "camera1", "camera2", "camera3"]
        if len(camera_frames) != 4:
            self.get_logger().warn(
                f"camera_frames should have 4 entries; got {len(camera_frames)}."
            )
        if len(camera_frames) < 4:
            camera_frames = camera_frames + default_frames[len(camera_frames):]
        if len(camera_frames) > 4:
            camera_frames = camera_frames[:4]

        self._camera_frames = tuple(camera_frames)
        self._jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        # Images
        self._img_lock = threading.Lock()
        self._latest: Dict[int, Optional[LatestImage]] = {
            0: None, 1: None, 2: None, 3: None}

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._subs = []
        for cam_id in range(4):
            topic = f"/cam{cam_id}/image_raw"
            self._subs.append(
                self.create_subscription(
                    RosImage, topic, self._make_cam_cb(cam_id), qos)
            )
            self.get_logger().info(f"Subscribing: {topic}")

        # Publish command as JSON string.
        self._pub = self.create_publisher(String, "/pixel_vln_cmd", 10)

        # Goal status trigger.
        self._goal_status_lock = threading.Lock()
        self._goal_status_event = threading.Event()
        self._last_goal_status: Optional[bool] = True
        self._saw_false_since_true = False
        self._goal_status_sub = self.create_subscription(
            Bool, "/far_reach_goal_status", self._goal_status_cb, 10
        )

        # TF
        self._tf_buffer = tf2_ros.Buffer(
            cache_time=rclpy.duration.Duration(seconds=5.0))
        self._tf_listener = tf2_ros.TransformListener(
            self._tf_buffer, self, spin_thread=False)

    def _make_cam_cb(self, cam_id: int):
        def cb(msg: RosImage) -> None:
            with self._img_lock:
                self._latest[cam_id] = LatestImage(msg=msg, ts_ms=_now_ms())
        return cb

    def _lookup_pose_xytheta(self, target_frame: str) -> Optional[Tuple[float, float, float]]:
        """
        Pose of `target_frame` in `world_frame` as (x, y, theta).
        """
        try:
            tfm = self._tf_buffer.lookup_transform(
                self._world_frame,
                target_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except Exception:
            return None

        t = tfm.transform.translation
        q = tfm.transform.rotation
        yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        return float(t.x), float(t.y), float(yaw)

    def _goal_status_cb(self, msg: Bool) -> None:
        status = bool(msg.data)
        with self._goal_status_lock:
            if status == self._last_goal_status:
                return

            if self._last_goal_status is True and status is False:
                self._saw_false_since_true = True
            elif self._last_goal_status is False and status is True and self._saw_false_since_true:
                self._saw_false_since_true = False
                self._goal_status_event.set()

            self._last_goal_status = status

    def get_sensor_poses(self) -> Dict[str, Any]:
        """
        Build sensor_poses mapping for the HTTP agent.
        DThinkPixelAgent only consumes keys containing "rgb_".
        """
        out: Dict[str, Any] = {}
        for cam_id, frame in enumerate(self._camera_frames):
            sensor = CAM_ID_TO_SENSOR.get(cam_id)
            if not sensor:
                continue
            p = self._lookup_pose_xytheta(frame)
            if p is None:
                self.get_logger().warn(f"frame {frame} tf lost")
                continue
            x, y, theta = p
            out[sensor] = {"x": x, "y": y, "theta": theta}

        # Vehicle pose is sometimes useful to log/debug; agent ignores non-rgb_ keys.
        vp = self._lookup_pose_xytheta(self._vehicle_frame)
        if vp is not None:
            x, y, theta = vp
            out["vehicle"] = {"x": x, "y": y, "theta": theta}
        return out

    def get_observation(self, require_all: bool = True) -> Optional[Dict[str, Any]]:
        """
        Return obs payload:
          {"images": {"rgb_front": {"b64": "...", "encoding": "jpg", "colorspace": "RGB"}, ...}}
        """
        with self._img_lock:
            latest = dict(self._latest)

        images: Dict[str, Any] = {}
        for cam_id, entry in latest.items():
            if entry is None:
                if require_all:
                    return None
                continue
            sensor = CAM_ID_TO_SENSOR.get(cam_id)
            if not sensor:
                continue
            try:
                rgb = _ros_img_to_rgb8(entry.msg)
                b64 = _encode_jpeg_b64(rgb, quality=self._jpeg_quality)
            except Exception as e:
                # If decode fails, skip this sensor.
                self.get_logger().warn(
                    f"cam{cam_id} decode/encode failed: {e}")
                if require_all:
                    return None
                continue
            images[sensor] = {"b64": b64,
                              "encoding": "jpg", "colorspace": "RGB"}

        if require_all and len(images) < 4:
            return None

        return {"images": images}

    def api_reset(
        self,
        *,
        instruction: str,
        first_obs: Optional[Dict[str, Any]],
        sensor_poses: Optional[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> str:
        payload = {
            "session_id": session_id,
            "instruction": instruction,
            "episode_id": str(uuid.uuid4()),
            "first_obs": first_obs,
            "sensor_poses": sensor_poses,
        }
        url = f"{self._api_base}/reset"
        resp = _http_post_json(url, payload, timeout_s=30.0)
        return str(resp["session_id"])

    def api_step(self, *, session_id: str, obs: Dict[str, Any], sensor_poses: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._api_base}/step"
        payload = {
            "session_id": session_id,
            "obs": obs,
            "sensor_poses": sensor_poses,
            "return_model_text": True,
            "return_model_out": False,
        }
        return _http_post_json(url, payload, timeout_s=30.0)

    def publish_cmd(self, msg_obj: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(msg_obj, ensure_ascii=False)
        self._pub.publish(msg)

    def reset_goal_status_cycle(self) -> None:
        with self._goal_status_lock:
            self._goal_status_event.clear()
            self._saw_false_since_true = False

    def wait_for_goal_status_cycle(self, timeout_s: float = 0.1) -> bool:
        return self._goal_status_event.wait(timeout=timeout_s)

    def consume_goal_status_cycle(self) -> None:
        self._goal_status_event.clear()

    @property
    def camera_frames(self) -> Tuple[str, str, str, str]:
        return self._camera_frames


def main() -> None:
    rclpy.init()
    node = PixelVlnClientNode()

    # Spin in background so we can use blocking input() in main thread.
    spin_stop = threading.Event()

    def _spin():
        while rclpy.ok() and not spin_stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)

    th = threading.Thread(target=_spin, daemon=True)
    th.start()

    node.get_logger().info("Ready. Type an instruction and press Enter.")

    try:
        while rclpy.ok():
            try:
                instruction = "go to the box in front of you"
            except EOFError:
                break
            if instruction == "":
                continue
            if instruction.lower() in ("exit", "quit"):
                break

            # Wait for initial images.
            node.get_logger().info("Waiting for camera images...")
            first_obs = None
            t0 = time.time()
            while rclpy.ok():
                first_obs = node.get_observation(require_all=True)
                if first_obs is not None:
                    break
                if time.time() - t0 > 10.0:
                    node.get_logger().warn("Timeout waiting for all 4 cameras; continuing with partial views.")
                    first_obs = node.get_observation(require_all=False)
                    break
                time.sleep(0.05)

            sensor_poses = node.get_sensor_poses()

            node.get_logger().info("Calling /reset ...")
            session_id = node.api_reset(
                instruction=instruction,
                first_obs=first_obs,
                sensor_poses=sensor_poses,
                session_id=None,
            )
            node.get_logger().info(f"session_id={session_id}")

            # Step loop until STOP.
            node.reset_goal_status_cycle()
            first_step = True
            while rclpy.ok():
                if not first_step:
                    while rclpy.ok():
                        if node.wait_for_goal_status_cycle(timeout_s=0.1):
                            node.consume_goal_status_cycle()
                            break
                    if not rclpy.ok():
                        break
                else:
                    first_step = False

                while rclpy.ok():
                    obs = node.get_observation(require_all=False)
                    if obs is None:
                        time.sleep(0.05)
                        continue

                    sensor_poses = node.get_sensor_poses()
                    try:
                        resp = node.api_step(
                            session_id=session_id, obs=obs, sensor_poses=sensor_poses)
                        break
                    except Exception as e:
                        node.get_logger().error(f"/step failed: {e}")
                        time.sleep(0.1)
                        continue
                if not rclpy.ok():
                    break
                node.get_logger().info(f"/step success: {resp}")
                action = resp.get("action") or {}
                a_type = (action.get("type") or "CMD").upper()

                if a_type == "MOVE_TO":
                    sensor = action.get("choice") or action.get("sensor") or ""
                    cam_id = SENSOR_TO_CAM_ID.get(sensor, -1)
                    pixel = action.get("pixel") or action.get(
                        "action") or [None, None]
                    cmd = {
                        "type": "MOVE_TO",
                        "session_id": session_id,
                        "camera_id": cam_id,
                        "camera_frame": (node.camera_frames[cam_id] if 0 <= cam_id < 4 else ""),
                        "sensor": sensor,
                        "pixel": pixel,
                        "raw_action": action,
                        "env_action": resp.get("env_action") or {},
                        "ts_ms": _now_ms(),
                    }
                    node.publish_cmd(cmd)
                else:
                    cmd = {
                        "type": "STOP",
                        "session_id": session_id,
                        "raw_action": action,
                        "env_action": resp.get("env_action") or {},
                        "ts_ms": _now_ms(),
                    }
                    node.publish_cmd(cmd)
                    node.get_logger().info("Received STOP. Waiting for next instruction...")
                    break
    finally:
        spin_stop.set()
        th.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
