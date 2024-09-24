import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
import open3d as o3d



def save_pcd(intrinsics, units, depth_image, box):
    print("saving to output.pcd ...")
    # 创建空的点云列表
    # points = []
    
    pad_y = 50
    pad_x = 50

    x_1 = box[0] - pad_x
    x_2 = box[2] + pad_x
    y_1 = box[1] - pad_y
    y_2 = box[3] + pad_y

    # 遍历图像的每个像素
    U = np.multiply(np.arange(x_1, x_2), np.ones((y_2 - y_1, 1))).astype(np.int16)
    V = np.multiply(np.arange(y_1, y_2).reshape(-1, 1), np.ones((1, x_2 - x_1))).astype(np.int16)

    Z = depth_image[y_1:y_2, x_1:x_2] *units
    X = (U - intrinsics.ppx) / intrinsics.fx * Z
    Y = (V - intrinsics.ppy) / intrinsics.fy * Z
    points = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)), axis=1)
    # for v in range(box[1] - pad_y, box[3] + pad_y):
    #     for u in range(box[0] - pad_x, box[2] + pad_x):
    #         z = depth_image[v, u] * units  # 深度值转换为米
    #         if z > 0:  # 排除无效点
    #             x = (u - intrinsics.ppx) / intrinsics.fx * z
    #             y = (v - intrinsics.ppy) / intrinsics.fy * z
    #             points.append([x, y, z])

    # 将点和颜色数据转换为Open3D点云格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 保存点云为PCD文件
    o3d.io.write_point_cloud("output.pcd", pcd)
    print("Done! output.pcd")