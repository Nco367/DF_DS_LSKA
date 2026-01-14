from PIL import Image
import numpy as np

# 图像尺寸
width, height = 1440, 1080
white_w, white_h = 1000, 1000

# 创建全黑图像
img_array = np.zeros((height, width), dtype=np.uint8)

# 计算中心白色区域的坐标
top = (height - white_h) // 2
left = (width - white_w) // 2
bottom = top + white_h
right = left + white_w

# 填充白色区域
img_array[top:bottom, left:right] = 255

# 转换为 PIL 图像
img = Image.fromarray(img_array, mode='L')

# 保存或显示（可选）
img.save("center_white_1440.png")
