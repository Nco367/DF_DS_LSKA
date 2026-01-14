import os
import numpy as np
import pickle

def check_point_cloud_shapes(folder_path):
    error_files = []
    total_files = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy') or file.endswith('.pkl'):
                total_files += 1
                file_path = os.path.join(root, file)

                try:
                    if file.endswith('.npy'):
                        points = np.load(file_path)
                    elif file.endswith('.pkl'):
                        with open(file_path, 'rb') as f:
                            points = pickle.load(f)

                    points = np.squeeze(points)
                    if isinstance(points, np.ndarray):
                        print(f"{file}: {points.shape}")
                        if points.ndim != 2 or (points.shape[1] != 3 and points.shape[0] != 3):
                            error_files.append((file, points.shape))
                    else:
                        error_files.append((file, "非NumPy数组"))

                except Exception as e:
                    error_files.append((file, str(e)))

    print("\n📊 总共检查了", total_files, "个文件")
    if error_files:
        print("❌ 存在异常的文件：")
        for ef in error_files:
            print("  ", ef)
    else:
        print("✅ 所有点云shape正常！")

# 调用方法：
# check_point_cloud_shapes("/home/q/ku/DenseFusion-1/DenseFusion-1/dataset/lm/targets")
