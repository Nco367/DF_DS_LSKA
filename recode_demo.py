import os
import cv2
import time
from typing import Optional, Tuple
import numpy as np


class USBCamera:
    def __init__(self,
                 camera_index: int = 0,
                 resolution: Tuple[int, int] = (640, 480),
                 fps: int = 30,
                 mirror: bool = True):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头设备")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self.mirror = mirror
        self.running = False
        self._latest_frame = None
        self._timestamps = []

        # 视频录制配置
        self.is_recording = False
        self.video_writer = None
        self.record_fps = fps

        # 帧序列保存配置
        self.is_saving_frames = False
        self.frame_counter = 0
        self.output_dir = "frame_sequence"
        self.image_format = "png"  # 支持jpg/png格式

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def start_frame_capture(self):
        """开始捕获帧序列"""
        if not self.is_saving_frames:
            self.frame_counter = 0
            self.is_saving_frames = True
            print(f"开始保存帧序列到 {self.output_dir}/")

    def stop_frame_capture(self):
        """停止捕获帧序列"""
        if self.is_saving_frames:
            self.is_saving_frames = False
            print(f"已保存 {self.frame_counter} 帧到 {self.output_dir}/")

    def _save_frame(self, frame: np.ndarray):
        """保存单个RGB帧"""
        filename = os.path.join(
            self.output_dir,
            f"frame_{self.frame_counter:04d}.{self.image_format}"
        )

        # 注意：OpenCV保存需要BGR格式，这里转换后保存
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr_frame, [
            int(cv2.IMWRITE_JPEG_QUALITY), 95  # 对于JPEG的质量参数
        ])

        self.frame_counter += 1

    def get_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            return None

        if self.mirror:
            frame = cv2.flip(frame, 1)

        self._latest_frame = frame
        self._timestamps.append(time.time())
        return frame

    def get_rgb_frame(self) -> Optional[np.ndarray]:
        frame = self.get_frame()
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 视频录制处理
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(frame)  # 直接使用原始BGR帧

            # 帧序列保存处理
            if self.is_saving_frames:
                self._save_frame(rgb_frame)

            return rgb_frame
        return None



    def show_preview(self, window_name: str = "Camera Preview") -> None:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        recording_indicator = False
        last_toggle_time = 0

        while self.running:
            rgb_frame = self.get_rgb_frame()
            if rgb_frame is None:
                continue

            display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # 显示状态信息
            # fps = self.calculate_fps()
            status_text = [
                #f"FPS: {fps:.1f}",
                "REC" if self.is_recording else "",
                "CAP" if self.is_saving_frames else ""
            ]

            # 绘制状态信息
            y_offset = 30
            for text in filter(None, status_text):
                color = (0, 255, 0) if text == "FPS" else (0, 0, 255)
                cv2.putText(display_frame, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                y_offset += 40

            cv2.imshow(window_name, display_frame)

            # 按键处理（带防抖）
            current_time = time.time()
            key = cv2.waitKey(1)
            if key == 27:  # ESC退出
                self.stop()
            elif key == ord('r') and (current_time - last_toggle_time) > 1:
                # R键控制视频录制
                last_toggle_time = current_time
                if self.is_recording:
                    self.stop_recording()
                else:
                    self.start_recording()
            elif key == ord('s') and (current_time - last_toggle_time) > 1:
                # S键控制帧捕获
                last_toggle_time = current_time
                if self.is_saving_frames:
                    self.stop_frame_capture()
                else:
                    self.start_frame_capture()

    def stop(self) -> None:
        self.running = False
        self.stop_recording()
        self.cap.release()
        cv2.destroyAllWindows()
        print("摄像头采集已停止")

    def stop_recording(self) -> None:
        """停止视频录制"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        print("视频录制已停止")

    def start(self) -> None:
        self.running = True
        print("摄像头采集已启动...")

    def start_recording(self, filename: str = "output.avi") -> None:
        """启动视频录制"""
        if self.is_recording:
            return

        # 获取摄像头参数
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 配置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            filename,
            fourcc,
            self.record_fps,
            (frame_width, frame_height)
        )

        if not self.video_writer.isOpened():
            raise RuntimeError("无法创建视频文件")

        self.is_recording = True
        print(f"开始录制视频到 {filename}")



    # 其他方法保持不变...


if __name__ == "__main__":
    # 初始化摄像头
    for idx in range(3):
        try:
            cam = USBCamera(
                camera_index=idx,
                resolution=(1280, 720),
                fps=30,
                mirror=True
            )
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("未找到可用摄像头设备")

    try:
        cam.start()
        cam.show_preview()  # 操作提示：
        # - R键：开始/停止视频录制
        # - S键：开始/停止帧捕获
        # - ESC键：退出程序
    finally:
        cam.stop()
