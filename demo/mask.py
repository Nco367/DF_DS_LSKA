import cv2
import numpy as np
# 读取图片
image_path = "/home/q/ku/DenseFusion-1/DenseFusion-1/demo//mask.png"  # 你的图片路径
img = cv2.imread(image_path)  # 以 BGR 格式加载

# 转换到 HSV 颜色空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义红色的 HSV 范围（适用于不同红色深度）
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# 生成红色的掩码
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# 形态学操作，填充内部黑点（闭运算）
kernel = np.ones((50, 50), np.uint8)
processed_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

# 生成黑白（纯二值）图像
binary_result = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)[1]

# 可选：保存处理后的图片
output_path = "/home/q/ku/DenseFusion-1/DenseFusion-1/demo/mask_gray.png"
cv2.imwrite(output_path, binary_result)
print(f"黑白图像已保存到: {output_path}")
