import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

# 配置RealSense相机
pipeline = rs.pipeline()
config = rs.config()

# 配置流（深度和颜色）
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
rs.config.enable_device_from_file(config, "/home/zju/Documents/elu_435_in.bag")

# 开始流
pipeline.start(config)

# 创建对齐对象，将深度图像与彩色图像对齐
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # 获取一帧数据
        frames = pipeline.wait_for_frames()

        # 对齐深度帧与颜色帧
        aligned_frames = align.process(frames)
        
        # 获取对齐后的深度帧与颜色帧
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # 转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # 显示颜色图像
        cv2.imshow('RealSense', color_image)
        
        # 按键检测
        key = cv2.waitKey(1)
        
        if key == ord('s'):  # 当按下 's' 键时保存当前帧为点云
            print("按下 's' 键，正在保存点云...")

            # 获取深度相机的内参
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

            # 创建空的点云列表
            points = []
            colors = []

            # 遍历图像的每个像素
            for v in range(depth_image.shape[0]):
                for u in range(depth_image.shape[1]):
                    z = depth_image[v, u] * depth_frame.get_units()  # 深度值转换为米
                    if z > 0:  # 排除无效点
                        x = (u - intrinsics.ppx) / intrinsics.fx * z
                        y = (v - intrinsics.ppy) / intrinsics.fy * z
                        points.append([x, y, z])
                        colors.append(color_image[v, u] / 255.0)  # 颜色值归一化

            # 将点和颜色数据转换为Open3D点云格式
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            # pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

            # 保存点云为PLY文件
            # o3d.io.write_point_cloud("output.ply", pcd, write_ascii=True)
            o3d.io.write_point_cloud("output.pcd", pcd)
            print("点云已保存为 output.pcd")

        elif key == ord('q'):  # 按下 'q' 键退出
            break

finally:
    # 停止流
    pipeline.stop()
    cv2.destroyAllWindows()
