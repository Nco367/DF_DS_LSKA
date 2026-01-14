import cv2
import numpy as np
import matplotlib.pyplot as plt


# 读取深度图（假设是16位单通道PNG）
depth_image = cv2.imread('000000.png', cv2.IMREAD_ANYDEPTH)  # 保留原始位深

# 检查深度图数据范围
print("Min depth:", np.min(depth_image), "Max depth:", np.max(depth_image))

# 方法1：归一化到0-255并转换为8位
depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 方法2：手动缩放（假设已知有效范围，例如500-5000毫米）
depth_scaled = np.clip((depth_image - 500) / (5000 - 500) * 255, 0, 255).astype(np.uint8)
# 直接显示原始深度数据（自动缩放）
plt.imshow(depth_image, cmap='jet')  # 'jet'颜色映射更易区分深度
plt.colorbar()  # 显示颜色条
plt.title('Depth Map Visualization')
plt.show()