#!/usr/bin/env python3
"""
RGB/RGB-D 实时采集 + 可视化 + 位姿模型对接 Demo

支持：
- Orbbec Astra 系列（通过 OpenNI2: primesense.openni2）
- Intel RealSense（通过 pyrealsense2）

可视化：
- OpenCV：叠加 2D 坐标轴与 2D 边界框
- Open3D：显示点云、三轴、3D 边界框（基于位姿）

位姿：
- 预留 PoseEstimator 接口：传入 (rgb, depth, K)，输出 R(3x3), t(3,), bbox_3d(8x3)

使用方法：
1) 安装依赖（按需）：
   pip install open3d opencv-python numpy
   # RealSense：pip install pyrealsense2
   # OpenNI2：pip install primesense  或 使用 Orbbec 官方 SDK 的 Python 包（ob）

2) 根据你的相机修改 CONFIG 中的 BACKEND 和内参 K（或走自动获取）。
3) 运行：python rgbd_pose_demo.py
   键盘：q 退出；o 切换 Open3D 可视化开/关。

注意：
- 如果没有位姿模型，示例会用单位姿态（R=I, t=[0,0,0.5]m）示范坐标轴与立方体。
- Astra S 的深度单位通常是毫米，记得转换为米。
"""
from __future__ import annotations
import time
import math
import sys
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

# ====== 可选依赖（按需导入） ======
try:
    import pyrealsense2 as rs  # RealSense
except Exception:
    rs = None

try:
    from primesense import openni2  # OpenNI2 for Orbbec/Astra
except Exception:
    openni2 = None

try:
    import open3d as o3d
except Exception:
    o3d = None


# ================== 配置 ==================
@dataclass
class Config:
    BACKEND: str = "astra"   # "astra" | "realsense"
    WIDTH: int = 640
    HEIGHT: int = 480
    FPS: int = 30
    # 相机内参（fx, fy, cx, cy），单位像素；若为 None 则尝试从驱动读取
    INTRINSICS: Optional[Tuple[float, float, float, float]] = None
    # 深度尺度（把原始深度值乘以该比例 => 米）
    DEPTH_SCALE: float = 0.001  # Astra 常见为 0.001（mm->m）；RealSense 自动读取
    # Open3D 可视化
    USE_O3D: bool = True

CONFIG = Config()


# =============== 相机抽象接口 ===============
class RGBDCamera:
    def start(self):
        raise NotImplementedError

    def read(self) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Returns:
            rgb: HxWx3, uint8, BGR (for OpenCV)
            depth: HxW, float32, 单位米
            depth_scale: float, 深度尺度（已应用后返回同 CONFIG）
            K: 3x3 内参矩阵
        """
        raise NotImplementedError

    def stop(self):
        pass


# =============== RealSense 实现 ===============
class RealSenseCam(RGBDCamera):
    def __init__(self, width=640, height=480, fps=30):
        if rs is None:
            raise RuntimeError("未安装 pyrealsense2，请改用 Astra 或安装依赖")
        self.width, self.height, self.fps = width, height, fps
        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = None
        self.depth_scale = None
        self.K = None

    def start(self):
        self.profile = self.pipeline.start(self.cfg)
        # 深度尺度
        dpt_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(dpt_sensor.get_depth_scale())  # 已是米比例
        # 内参
        intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        dpt = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not dpt or not color:
            raise RuntimeError("RealSense 未获取到帧")
        depth = np.asanyarray(dpt.get_data()).astype(np.float32) * self.depth_scale
        rgb = np.asanyarray(color.get_data())  # BGR
        return rgb, depth, self.depth_scale, self.K.copy()

    def stop(self):
        self.pipeline.stop()


# =============== Astra (OpenNI2) 实现 ===============
class AstraCam(RGBDCamera):
    def __init__(self, width=640, height=480, fps=30, depth_scale=0.001, intrinsics=None):
        if openni2 is None:
            raise RuntimeError("未安装 OpenNI2(primesense)，请安装或使用 Orbbec 官方 SDK Python 包")
        self.width, self.height, self.fps = width, height, fps
        self.depth_scale = float(depth_scale)
        self._intr = intrinsics  # (fx, fy, cx, cy) or None
        self.dev = None
        self.color_stream = None
        self.depth_stream = None

    def start(self):
        # 初始化 OpenNI2（若运行失败，可传入驱动库路径）
        if not openni2.is_initialized():
            openni2.initialize()  # 可换成 openni2.initialize('/path/to/OpenNI2/Redist')
        self.dev = openni2.Device.open_any()
        # Color
        self.color_stream = self.dev.create_color_stream()
        self.color_stream.set_video_mode(openni2.OniVideoMode(
            pixelFormat=openni2.PIXEL_FORMAT_RGB888, resolutionX=self.width, resolutionY=self.height, fps=self.fps
        ))
        self.color_stream.start()
        # Depth
        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.set_video_mode(openni2.OniVideoMode(
            pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM, resolutionX=self.width, resolutionY=self.height, fps=self.fps
        ))
        self.depth_stream.start()
        # 注意：OpenNI 的 RGB 是 RGB 顺序，需要转为 BGR 给 OpenCV
        # 内参：OpenNI2 不直接给标定内参，这里用传入或简单估计（请替换为你的标定值）
        if self._intr is None:
            fx = 570.0 * (self.width / 640.0)
            fy = 570.0 * (self.height / 480.0)
            cx = self.width / 2.0
            cy = self.height / 2.0
            self._intr = (fx, fy, cx, cy)
        self.K = np.array([[self._intr[0], 0, self._intr[2]], [0, self._intr[1], self._intr[3]], [0, 0, 1]], dtype=np.float32)

    def read(self):
        dframe = self.depth_stream.read_frame()
        cframe = self.color_stream.read_frame()
        d_data = np.frombuffer(dframe.get_buffer_as_uint16(), dtype=np.uint16).reshape(self.height, self.width)
        c_data = np.frombuffer(cframe.get_buffer_as_uint8(), dtype=np.uint8).reshape(self.height, self.width, 3)
        depth = d_data.astype(np.float32) * self.depth_scale  # 米
        rgb = cv2.cvtColor(c_data, cv2.COLOR_RGB2BGR)
        return rgb, depth, self.depth_scale, self.K.copy()

    def stop(self):
        try:
            if self.color_stream: self.color_stream.stop()
            if self.depth_stream: self.depth_stream.stop()
            if openni2.is_initialized(): openni2.shutdown()
        except Exception:
            pass


# =============== 位姿估计接口（占位示例） ===============
class PoseEstimator:
    """将你的模型封装到这里：load_model / infer"""
    def __init__(self):
        # TODO: 加载你的权重/引擎
        pass

    def infer(self, rgb: np.ndarray, depth: np.ndarray, K: np.ndarray):
        """输入：BGR, float深度(米), K
        输出：R(3x3), t(3,), bbox3d(8x3)
        这里先返回一个演示用的单位立方体与简单位姿。
        """
        R = np.eye(3, dtype=np.float32)
        t = np.array([0.0, 0.0, 0.5], dtype=np.float32)  # 相机前方 0.5m
        # 一个边长 0.1m 的立方体（以物体坐标原点为中心）
        s = 0.05
        corners = np.array([
            [-s, -s, -s], [ s, -s, -s], [ s,  s, -s], [-s,  s, -s],
            [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
        ], dtype=np.float32)
        return R, t, corners


# =============== 可视化工具 ===============
def project_points(pts3d: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """将 3D 点投影到像素坐标"""
    P = (R @ pts3d.T).T + t[None, :]
    x = P[:, 0] / np.maximum(P[:, 2], 1e-6)
    y = P[:, 1] / np.maximum(P[:, 2], 1e-6)
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], axis=1)

def draw_axes(img: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, axis_len: float = 0.1):
    """在图像上画 3D 坐标轴 (X-红, Y-绿, Z-蓝)。OpenCV 默认 BGR，这里颜色按照 BGR 传入。"""
    origin = np.zeros((1, 3), dtype=np.float32)
    axes = np.array([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]], dtype=np.float32)
    pts = np.concatenate([origin, axes], axis=0)
    uv = project_points(pts, K, R, t)
    o = tuple(np.round(uv[0]).astype(int))
    x = tuple(np.round(uv[1]).astype(int))
    y = tuple(np.round(uv[2]).astype(int))
    z = tuple(np.round(uv[3]).astype(int))
    cv2.line(img, o, x, (0, 0, 255), 2)
    cv2.line(img, o, y, (0, 255, 0), 2)
    cv2.line(img, o, z, (255, 0, 0), 2)


def draw_bbox2d(img: np.ndarray, uv8: np.ndarray):
    """根据 8 个角点的投影画 3D 立方体的 2D 线框"""
    uv = uv8.round().astype(int)
    # 边索引
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a,b in edges:
        pa, pb = tuple(uv[a]), tuple(uv[b])
        cv2.line(img, pa, pb, (0, 255, 255), 2)


def depth_to_pointcloud(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """depth(HxW, m) -> 点云 Nx3（相机坐标系）"""
    h, w = depth.shape
    i, j = np.indices((h, w))
    z = depth.reshape(-1)
    x = (j.reshape(-1) - K[0, 2]) * z / K[0, 0]
    y = (i.reshape(-1) - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1)
    mask = (z > 0)
    return pts[mask]


def make_o3d_geom(pc: np.ndarray, R: np.ndarray, t: np.ndarray, bbox3d: np.ndarray):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
    cloud.estimate_normals()
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # 物体坐标轴（将原点坐标轴变换到物体位姿）
    obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = t
    obj_frame.transform(T)
    # 3D bbox（连线）
    lines = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    bbox = o3d.geometry.LineSet()
    bbox.points = o3d.utility.Vector3dVector((R @ bbox3d.T).T + t)
    bbox.lines = o3d.utility.Vector2iVector(lines)
    return cloud, frame, obj_frame, bbox


# =============== 主循环 ===============
class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.estimator = PoseEstimator()
        self.cam: Optional[RGBDCamera] = None
        self.o3d_vis: Optional[o3d.visualization.Visualizer] = None
        self.use_o3d = bool(cfg.USE_O3D and (o3d is not None))

    def init_cam(self):
        if self.cfg.BACKEND == "realsense":
            self.cam = RealSenseCam(self.cfg.WIDTH, self.cfg.HEIGHT, self.cfg.FPS)
        else:
            self.cam = AstraCam(self.cfg.WIDTH, self.cfg.HEIGHT, self.cfg.FPS, self.cfg.DEPTH_SCALE, self.cfg.INTRINSICS)
        self.cam.start()

    def toggle_o3d(self):
        self.use_o3d = not self.use_o3d
        if not self.use_o3d and self.o3d_vis is not None:
            self.o3d_vis.destroy_window()
            self.o3d_vis = None

    def run(self):
        self.init_cam()
        print("按 q 退出；按 o 切换 Open3D 可视化开/关。")
        if self.use_o3d and self.o3d_vis is None:
            self.o3d_vis = o3d.visualization.Visualizer()
            self.o3d_vis.create_window("Open3D - 点云与位姿")
        while True:
            rgb, depth, depth_scale, K = self.cam.read()
            # 位姿推理（替换为你的模型）
            R, t, bbox3d = self.estimator.infer(rgb, depth, K)

            # 2D 叠加
            uv8 = project_points(bbox3d, K, R, t)
            vis_img = rgb.copy()
            draw_axes(vis_img, K, R, t)
            draw_bbox2d(vis_img, uv8)

            # 深度可视化
            depth_vis = (np.clip(depth / 2.0, 0, 1) * 255).astype(np.uint8)  # 0~2m 范围伪彩
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            both = np.hstack([vis_img, depth_vis])
            cv2.imshow("RGB (带轴与框) | Depth", both)

            # Open3D 3D 可视化
            if self.use_o3d and o3d is not None:
                pc = depth_to_pointcloud(depth, K)
                cloud, frame, obj_frame, bbox = make_o3d_geom(pc, R, t, bbox3d)
                if self.o3d_vis is None:
                    self.o3d_vis = o3d.visualization.Visualizer()
                    self.o3d_vis.create_window("Open3D - 点云与位姿")
                self.o3d_vis.clear_geometries()
                self.o3d_vis.add_geometry(cloud)
                self.o3d_vis.add_geometry(frame)
                self.o3d_vis.add_geometry(obj_frame)
                self.o3d_vis.add_geometry(bbox)
                self.o3d_vis.poll_events()
                self.o3d_vis.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                self.toggle_o3d()

        self.cam.stop()
        if self.o3d_vis is not None:
            self.o3d_vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        App(CONFIG).run()
    except KeyboardInterrupt:
        pass
