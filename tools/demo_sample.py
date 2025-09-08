import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBError

# 创建 Pipeline
pipeline = Pipeline()

# 配置流
config = Config()

# 获取设备支持的分辨率和帧率（可选）
# 你可以通过 OBSensor.get_stream_profile_list() 查询支持的格式

# 配置 RGB 流（1920x1080, 30fps, MJPG 或 RGB）
try:
    config.enable_stream(OBSensorType.COLOR_SENSOR, 1920, 1080, OBFormat.MJPG, 30)
except OBError as e:
    print("RGB stream config failed:", e)
    try:
        config.enable_stream(OBSensorType.COLOR_SENSOR, 1920, 1080, OBFormat.RGB, 30)
        print("Using RGB format")
    except:
        print("Falling back to lower resolution")
        config.enable_stream(OBSensorType.COLOR_SENSOR, 1280, 720, OBFormat.RGB, 30)

# 配置深度流（可根据相机支持设置，如 640x480, 30fps）
try:
    config.enable_stream(OBSensorType.DEPTH_SENSOR, 640, 480, OBFormat.Y16, 30)
except OBError as e:
    print("Depth stream config failed:", e)

# 启动流
pipeline.start(config)

print("Orbbec 相机已启动，按 'q' 退出...")

try:
    while True:
        # 等待一帧数据
        frames = pipeline.wait_for_frames(100)  # 超时 100ms
        if frames is None:
            continue

        # 获取 RGB 帧
        color_frame = frames.get_color_frame()
        if color_frame is None:
            continue

        # 获取深度帧
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            continue

        # 转换为 numpy 数组
        # 注意：RGB 数据是 HxWx3，通道顺序为 RGB
        color_data = np.asanyarray(color_frame.get_data())
        if color_data.size == 0:
            continue

        # 深度数据（单位：mm）
        depth_data = np.asanyarray(depth_frame.get_data())
        if depth_data.size == 0:
            continue

        # OpenCV 显示需要 BGR
        if len(color_data.shape) == 3:
            color_bgr = cv2.cvtColor(color_data, cv2.COLOR_RGB2BGR)
        else:
            color_bgr = color_data  # 灰度图

        # 深度图归一化以便可视化
        depth_colored = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_data, alpha=0.05), cv2.COLORMAP_JET
        )

        # 显示
        cv2.imshow("Orbbec RGB", color_bgr)
        cv2.imshow("Orbbec Depth", depth_colored)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()