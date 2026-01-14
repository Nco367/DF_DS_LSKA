import os
import glob
import yaml
import numpy as np
import open3d as o3d

def compute_model_info(ply_path):
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()

    pts = np.asarray(mesh.vertices)   # (N,3)

    # AABB
    min_xyz = pts.min(axis=0)
    max_xyz = pts.max(axis=0)
    size_xyz = max_xyz - min_xyz

    # diameter = max distance between any two points
    # 用凸包点加速
    hull, _ = mesh.compute_convex_hull()
    hull_pts = np.asarray(hull.vertices)

    # 计算两两距离（O(N^2)，但 hull 点一般不多）
    dists = np.linalg.norm(
        hull_pts[None, :, :] - hull_pts[:, None, :],
        axis=-1
    )
    diameter = dists.max()

    return {
        "diameter": float(diameter),
        "min_x": float(min_xyz[0]),
        "min_y": float(min_xyz[1]),
        "min_z": float(min_xyz[2]),
        "size_x": float(size_xyz[0]),
        "size_y": float(size_xyz[1]),
        "size_z": float(size_xyz[2]),
    }


def generate_models_info(ply_dir, save_path):
    ply_files = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))

    models_info = {}

    for idx, ply_path in enumerate(ply_files, start=1):
        print(f"Processing {idx}: {os.path.basename(ply_path)}")
        info = compute_model_info(ply_path)
        models_info[idx] = info

    with open(save_path, "w") as f:
        yaml.dump(models_info, f, sort_keys=False)

    print("Saved:", save_path)


if __name__ == "__main__":
    ply_dir = r"/media/q/SSD2T/1linux/Linemod6_dataset/Linemod_preprocessed/model/2"          # 你的ply文件夹
    save_path = "/media/q/SSD2T/1linux/Linemod6_dataset/Linemod_preprocessed/model/21"       # 输出

    generate_models_info(ply_dir, save_path)
