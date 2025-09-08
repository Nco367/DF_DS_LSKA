import cv2
import numpy as np
from orbbecsdk import pipeline, config, OB_FORMAT_Y16, OB_FORMAT_RGB

# 创建管道
pipe = pipeline()
cfg = config()

# 开启深度流和彩色流
cfg.enable_stream('depth', 640, 480, OB_FORMAT_Y16, 30)
cfg.enable_stream('color', 640, 480, OB_FORMAT_RGB, 30)

pipe.start(cfg)

while True:
    frames = pipe.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if depth_frame and color_frame:
        depth = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())

        # 深度图可视化
        depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)

        cv2.imshow("Color", color)
        cv2.imshow("Depth", depth_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipe.stop()
cv2.destroyAllWindows()
