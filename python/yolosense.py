import ctypes
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
import open3d as o3d
 
from ultralytics import YOLO 
 
''' 深度相机 '''
pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
config = rs.config()  # 定义配置config
 
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流
rs.config.enable_device_from_file(config, "/home/zju/Documents/elu_435_in.bag")
 
pipe_profile = pipeline.start(config)  # streaming流开始
align = rs.align(rs.stream.color)
 
 
def get_aligned_images(frame):
    aligned_frames = align.process(frame)  # 获取对齐帧，将深度框与颜色框对齐
 
    aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的的depth帧
    aligned_color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的的color帧
 
    #### 获取相机参数 ####
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
 
    img_color = np.asanyarray(aligned_color_frame.get_data())  # RGB图
    img_depth = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
 
    # depth_colormap = cv2.applyColorMap \
    #     (cv2.convertScaleAbs(img_depth, alpha=0.008)
    #      , cv2.COLORMAP_JET)
 
    return  depth_intrin, img_color, aligned_depth_frame, img_depth

def save_pcd(intrinsics, units, depth_image, box):
    print("saving to output.pcd ...")
    # 创建空的点云列表
    points = []
    
    pad_y = 20
    pad_x = 20

    # 遍历图像的每个像素
    for v in range(box[1] - pad_y, box[3] + pad_y):
        for u in range(box[0] - pad_x, box[2] + pad_x):
            z = depth_image[v, u] * units  # 深度值转换为米
            if z > 0:  # 排除无效点
                x = (u - intrinsics.ppx) / intrinsics.fx * z
                y = (v - intrinsics.ppy) / intrinsics.fy * z
                points.append([x, y, z])

    # 将点和颜色数据转换为Open3D点云格式
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    # 保存点云为PLY文件
    o3d.io.write_point_cloud("output.pcd", pcd)
    print("Done! output.pcd")


class Py2Cpp(ctypes.Structure):
    _fields_ = [
        ('template_list_path', ctypes.c_char_p),
    ]

class Var2Py(ctypes.Structure):
    _fields_ = [
        ('best_fitness_score', ctypes.c_float),

        ('rotation_00', ctypes.c_float),
        ('rotation_01', ctypes.c_float),
        ('rotation_02', ctypes.c_float),
        ('rotation_10', ctypes.c_float),
        ('rotation_11', ctypes.c_float),
        ('rotation_12', ctypes.c_float),
        ('rotation_20', ctypes.c_float),
        ('rotation_21', ctypes.c_float),
        ('rotation_22', ctypes.c_float),

        ('translation_x', ctypes.c_float),
        ('translation_y', ctypes.c_float),
        ('translation_z', ctypes.c_float)
    ]
 
 
if __name__ == '__main__':
    # load weights
    model = YOLO("best.pt") 

    # load shared library from cpp
    so = ctypes.CDLL("/home/zju/Yolov8/build/libtemplate_alignment.so")
    py2cpp = Py2Cpp(b"/home/zju/realsense_ws/template_pcd/template_list.txt")
    so.set_py2cpp.argtypes = [ctypes.POINTER(Py2Cpp)]
    so.set_py2cpp(ctypes.byref(py2cpp))
 
    try:
        while True:
            frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
            depth_intrin, img_color, aligned_depth_frame, img_depth = get_aligned_images(frames)  # 获取对齐图像与相机参数

            source = [img_color]

            # 调用YOLOv8中的推理，还是相当于把d435i中某一帧的图片进行detect推理
            results = model.predict(source, save=False, show_conf=False)

            boxes = results[0].boxes.data.numpy()
            index_array = np.argwhere(boxes[:,-1] == 0)
            if np.size(index_array) == 0:
                continue
            index = index_array[0, 0]

            charge_box = boxes[index, 0:4].astype(dtype=np.int16)  # [x, y, x, y]

            # plot a BGR numpy array of predictions
            im_array = results[0].plot()  

            # 计算像素坐标系
            ux, uy = int((charge_box[0] + charge_box[2]) / 2), int((charge_box[1] + charge_box[3]) / 2)
            dis = aligned_depth_frame.get_distance(ux, uy)
            
            # 计算相机坐标系的xyz
            camera_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis) 
            if camera_xyz[-1] > 0:
                camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
                camera_xyz = np.array(list(camera_xyz)) * 1000  # mm

                cv2.circle(im_array, (ux, uy), 4, (255, 255, 255), 5)  # 标出中心点
                cv2.putText(im_array, str(camera_xyz.tolist()), (ux + 20, uy + 10), 0, 0.5,
                            [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)  # 标出坐标
                
            # if camera_xyz[-1] < 250:
            if (charge_box[2] - charge_box[0]) > 90:
                save_pcd(depth_intrin, aligned_depth_frame.get_units(), img_depth, charge_box)
                cv2.destroyAllWindows()
                print("Aligning the target to template ...")
                so.TemplateAlign(b"/home/zju/Yolov8/python/output.pcd")
                break

            cv2.namedWindow('detection', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            cv2.resizeWindow('detection', 640, 480)
            cv2.imshow('detection', im_array)
            cv2.waitKey(200)
 
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                pipeline.stop()
                break


    finally:
        # Stop streaming
        pipeline.stop()