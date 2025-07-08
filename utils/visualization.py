import cv2
import numpy as np
import numpy as np
import torch
def get_3d_obb_corners(points):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    points = np.squeeze(points)  # 防止多余维度
    print("points shape:", points.shape)
    if points.shape[0] == 3 and points.shape[1] != 3:
        print("检测到 (3, N) 点云，自动转置")
        points = points.T
    # 检查维度
    assert points.ndim == 2 and points.shape[1] == 3, f"点云shape异常: {points.shape}"
    assert points.shape[0] >= 3, f"点云数量过少: {points.shape[0]}"

    # 1. 去中心化
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # 2. PCA
    cov = np.cov(centered_points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 3. 投影到主方向坐标系
    projected_points = centered_points @ eigvecs
    print("projected_points shape:", projected_points.shape)

    # 4. 在新坐标系下计算 AABB
    min_vals = np.min(projected_points, axis=0)
    max_vals = np.max(projected_points, axis=0)
    print("min_vals shape:", min_vals.shape)

    assert min_vals.shape == (3,), f"min_vals shape 异常: {min_vals.shape}"

    x_min, y_min, z_min = min_vals
    x_max, y_max, z_max = max_vals

    # 5. 8个角点
    corners_local = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ])

    # 6. 变换回原坐标系
    obb_corners = corners_local @ eigvecs.T + centroid

    return obb_corners


# def get_3d_obb_corners(points):
#     """
#     计算点云的Oriented Bounding Box（OBB）8个角点
#     参数：
#         points: (N, 3) NumPy数组，点云坐标，必须是相机坐标系下
#     返回：
#         obb_corners: (8, 3) 数组，OBB的8个角点坐标
#     """
#     if isinstance(points, torch.Tensor):
#         points = points.cpu().numpy()  # 先转 NumPy
#
#     # 1. 去中心化
#     centroid = np.mean(points, axis=0)
#     centered_points = points - centroid
#
#     # 2. PCA：计算协方差矩阵和特征值特征向量
#     cov = np.cov(centered_points.T)
#     eigvals, eigvecs = np.linalg.eigh(cov)  # eigvecs的列是特征方向
#
#     # 3. 将点云投影到主方向坐标系
#     # 3. 投影到主方向坐标系
#     projected_points = centered_points @ eigvecs
#     projected_points = np.squeeze(projected_points)  # 防止多余维度
#
#     # 4. 在新坐标系下计算 AABB
#     min_vals = np.min(projected_points, axis=0)
#     max_vals = np.max(projected_points, axis=0)
#     assert min_vals.shape == (3,)
#     x_min, y_min, z_min = min_vals
#     x_max, y_max, z_max = max_vals
#     print("projected_points shape:", projected_points.shape)
#     print("min_vals shape:", min_vals.shape)
#     # 5. 得到新坐标系下8个角点
#     corners_local = np.array([
#         [x_min, y_min, z_min],
#         [x_max, y_min, z_min],
#         [x_max, y_max, z_min],
#         [x_min, y_max, z_min],
#         [x_min, y_min, z_max],
#         [x_max, y_min, z_max],
#         [x_max, y_max, z_max],
#         [x_min, y_max, z_max]
#     ])
#
#     # 6. 把角点从OBB局部坐标系变换回相机坐标系
#     obb_corners = corners_local @ eigvecs.T + centroid
#
#     return obb_corners
#

# def get_3d_bbox_corners(model_points):
#     """
#     计算点云的轴对齐包围框（AABB）的8个角点
#     参数：
#         model_points: (N, 3) 形状的NumPy数组，表示点云坐标
#     返回：
#         bbox_corners: (8, 3) 形状的数组，包含8个角点坐标
#     """
#     if model_points.ndim == 3:
#         model_points = model_points.squeeze(0)  # 去除批次维度
#         print("model_points_np shape:", model_points.shape)
#
#     # 1. 计算各轴的最小值和最大值
#     min_vals = np.min(model_points, axis=0)  # 形状 (3,)
#     max_vals = np.max(model_points, axis=0)  # 形状 (3,)
#
#     # 2. 生成所有可能的极值组合
#     x_min, y_min, z_min = min_vals
#     x_max, y_max, z_max = max_vals
#
#     # 3. 组合成8个角点
#     bbox_corners = np.array([
#         [x_min, y_min, z_min],  # 前左下
#         [x_max, y_min, z_min],  # 前右下
#         [x_max, y_max, z_min],  # 前右上
#         [x_min, y_max, z_min],  # 前左上
#         [x_min, y_min, z_max],  # 后左下
#         [x_max, y_min, z_max],  # 后右下
#         [x_max, y_max, z_max],  # 后右上
#         [x_min, y_max, z_max]  # 后左上
#     ])
#     print("model_points_np shape:", bbox_corners.shape)
#     return bbox_corners

def draw_coordinate_axis(img, pose, K, length=1, colors=None):
    """
    在图像上绘制物体坐标系
    参数：
        img: 原始RGB图像 (H,W,3)
        pose: 4x4位姿矩阵
        K: 3x3相机内参
        length: 轴长度（米）
        colors: XYZ轴颜色列表
    """
    # 坐标系端点（物体坐标系）
    points = np.float32([
        [0, 0, 0],
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length]
    ])

    # 转换到相机坐标系
    cam_points = np.dot(points, pose[:3, :3].T) + pose[:3, 3]

    # 投影到图像平面
    img_points, _ = cv2.projectPoints(
        cam_points,# 相机坐标系下的3D点
        np.eye(3), np.zeros(3),# 旋转向量（单位矩阵，表示无额外旋转）
        K, # 相机内参矩阵
        None # 畸变系数（假设无畸变）
    )
    img_points = img_points.reshape(-1, 2).astype(int)

    # 绘制轴线
    origin = tuple(img_points[0]) # 原点在图像上的坐标
    for i in range(1, 4):
        color = colors[i - 1] if colors else (0, 0, 255)  # 颜色：X-红，Y-绿，Z-蓝（默认）
        cv2.arrowedLine(img, origin, tuple(img_points[i]), color, 2)


def draw_3d_bbox(img, bbox_corners, pose, K, color=(0, 255, 0), thickness=2):
    """
    绘制3D包围框
    参数：
        bbox_corners: 8个角点的物体坐标系坐标 (N,3)
        pose: 4x4位姿矩阵
    """
    # 将包围框角点从物体坐标系转换到相机坐标系
    cam_corners = np.dot(bbox_corners, pose[:3, :3].T) + pose[:3, 3]

    # 投影到图像
    proj_points, _ = cv2.projectPoints(
        cam_corners,
        np.eye(3), np.zeros(3),
        K, None
    )
    proj_points = proj_points.reshape(-1, 2).astype(int)

    # 绘制立方体边
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for (s, e) in edges:
        cv2.line(img, tuple(proj_points[s]), tuple(proj_points[e]),
                 color, thickness)