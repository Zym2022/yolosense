import ctypes
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
import open3d as o3d
 
from ultralytics import YOLO 

from time import sleep
import socket
import threading

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from utils import *
 
''' 深度相机 '''
pipeline = rs.pipeline()  # 定义流程pipeline，创建一个管道
config = rs.config()  # 定义配置config
 
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流
# rs.config.enable_device_from_file(config, "/home/zju/Documents/20240920_170453.bag")
 
pipe_profile = pipeline.start(config)  # streaming流开始
align = rs.align(rs.stream.color)

class Py2Cpp(ctypes.Structure):
    _fields_ = [
        ('template_list_path', ctypes.c_char_p),
        ('px', ctypes.c_float),
        ('py', ctypes.c_float),
        ('pz', ctypes.c_float),
        ('qx', ctypes.c_float),
        ('qy', ctypes.c_float),
        ('qz', ctypes.c_float),
        ('qw', ctypes.c_float),
    ]

class Var2Py(ctypes.Structure):
    _fields_ = [
        ('best_fitness_score', ctypes.c_float),

        ('px', ctypes.c_float),
        ('py', ctypes.c_float),
        ('pz', ctypes.c_float),
        ('qx', ctypes.c_float),
        ('qy', ctypes.c_float),
        ('qz', ctypes.c_float),
        ('qw', ctypes.c_float),
    ]

pub_list = [0, 0, 0, 0, 0, 0, 0, 0]
rec_list = []
state = -2
recieved = 0

def udp_server(s):
    global pub_list
    global state
    global recieved
    while True:
        if state == pub_list[0] and recieved != 0:
            sleep(0.5)
            continue

        if recieved == 0:
            recieved = 1 if len(rec_list) != 0 else 0

        s.sendto((str(pub_list[0]) + ' ' + str(pub_list[1])+ ' ' + str(pub_list[2])+ ' ' + str(pub_list[3])+ ' ' + str(pub_list[4])+ ' ' + str(pub_list[5])+ ' ' + str(pub_list[6])+ ' ' + str(pub_list[7])) .encode('utf-8'), udp_addr_server)
        print("sending: ", str(pub_list[0]) + ' ' + str(pub_list[1])+ ' ' + str(pub_list[2])+ ' ' + str(pub_list[3])+ ' ' + str(pub_list[4])+ ' ' + str(pub_list[5])+ ' ' + str(pub_list[6])+ ' ' + str(pub_list[7]))
        state = pub_list[0]
        sleep(1)

def udp_client(s):
    global recieved
    global rec_list
    while True:
        recv_data = s.recvfrom(1024)
        print("[From %s:%d]:%s" % (recv_data[1][0], recv_data[1][1], recv_data[0].decode("utf-8")))
        rec_list =  [float(x) for x in recv_data[0].split()]
        sleep(1)

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

def residuals(params, points):
    a, b, c, d = params
    x, y, z = points.T
    return (a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)

def project_point_to_plane(point, plane_coeffs):
                    """
                    计算点到平面的投影点坐标。

                    :param point: 三维点的坐标 (x0, y0, z0)
                    :param plane_coeffs: 平面方程的系数 (a, b, c, d)
                    :return: 投影点的坐标 (x, y, z)
                    """
                    x0, y0, z0 = point
                    a, b, c, d = plane_coeffs

                    # 计算点到平面的距离
                    numerator = a * x0 + b * y0 + c * z0 + d
                    denominator = a**2 + b**2 + c**2

                    # 计算投影点的坐标
                    x = x0 - a * numerator / denominator
                    y = y0 - b * numerator / denominator
                    z = z0 - c * numerator / denominator
                    return np.array([x, y, z])


if __name__ == '__main__':
    # load weights
    model = YOLO("./weights/best.pt") 

    # load shared library from cpp
    so = ctypes.CDLL("/home/zju/Yolov8/build/libtemplate_alignment.so")
    py2cpp = Py2Cpp(b"/home/zju/realsense_ws/template_pcd/template_list.txt", \
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    so.set_py2cpp.argtypes = [ctypes.POINTER(Py2Cpp)]
    so.set_py2cpp(ctypes.byref(py2cpp))

    # udp
    udp_addr_server = ('192.168.5.17', 9999)
    udp_socket_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    udp_addr_client = ('', 8888)
    udp_socket_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket_client.bind(udp_addr_client)
    
    
    thread_list = []
    t1 = threading.Thread(target=udp_client, args=(udp_socket_client, ))
    thread_list.append(t1)
    t2 = threading.Thread(target=udp_server, args=(udp_socket_server, ))
    thread_list.append(t2)
 
    for t in thread_list:
        t.setDaemon(True)
        t.start()
    
    # the pose of the camera in end
    trans_camera2end =np.array([[0.04999190841220458, 0.9987464691287659, -0.002509879704122885, -0.1233286407353617],
                                [-0.9986450328192172, 0.05002288314324066, 0.01434606522856827, 0.003561782838864993],
                                [0.01445363341206645, 0.001789291720513989, 0.9998939398337848, 0.0788226942731852],
                                [0, 0, 0, 1]])
    
    try:
        while True:
            frames = pipeline.wait_for_frames()  # 等待获取图像帧，获取颜色和深度的框架集
            depth_intrin, img_color, aligned_depth_frame, img_depth = get_aligned_images(frames)  # 获取对齐图像与相机参数

            source = [img_color]

            # 调用YOLOv8中的推理，还是相当于把d435i中某一帧的图片进行detect推理
            results = model.predict(source, save=False, show_conf=False) # the "Speed" info is closed in the /ultralytics/engin/predictor.py

            boxes = results[0].boxes.data.numpy()
            index_array = np.argwhere(boxes[:,-1] == 0)
            if np.size(index_array) == 0:
                pub_list = [0, 0, 0, 0, 0, 0, 0, 0]
                continue
            index = index_array[0, 0]
            charge_box = boxes[index, 0:4].astype(dtype=np.int16)  # [x, y, x, y]

            # plot a BGR numpy array of predictions
            im_array = results[0].plot()  

            # 计算像素坐标系
            corners = [[charge_box[0], charge_box[1]], [charge_box[0], charge_box[3]], \
                       [charge_box[2], charge_box[1]], [charge_box[2], charge_box[3]], \
                       [int((charge_box[0] + charge_box[2]) / 2), int((charge_box[1] + charge_box[3]) / 2)]]
            camera_xyz = []
            for ux, uy in corners:
                dis = aligned_depth_frame.get_distance(ux, uy)
                
                # 计算相机坐标系的xyz
                pc_xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, (ux, uy), dis)
                if pc_xyz[-1] > 0:
                    camera_xyz.append(pc_xyz)

            # plot the coordinate of the center
            if pc_xyz[-1] > 0:
                pc_xyz = np.round(np.array(pc_xyz), 3)  # 转成3位小数
                pc_xyz = np.array(list(pc_xyz)) * 1000  # mm

                cv2.circle(im_array, (ux, uy), 4, (255, 255, 255), 5)  # 标出中心点
                cv2.putText(im_array, str(pc_xyz.tolist()), (ux + 20, uy + 10), 0, 0.5,
                            [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)  # 标出坐标
                
            # ax + by + cz + d = 0
            if len(camera_xyz) > 2:
                camera_xyz = np.array(camera_xyz)
                init_guess = [1, 1, 1, 1]
                result = least_squares(residuals, init_guess, args=(camera_xyz,))
                a, b, c, d = result.x if result.x[2] > 0 else -result.x

                # where is the origin
                #####################
                origin = np.mean(camera_xyz, axis=0).tolist()
                origin = project_point_to_plane(origin, [a, b, c, d])

                # what is axis
                #####################
                z = np.array([a,b,c]) / np.linalg.norm(np.array([a,b,c]))
                down_middel = (camera_xyz[1, :] + camera_xyz[3, :]) / 2
                proj_dm = project_point_to_plane(list(down_middel), [a, b, c, d])
                y = (proj_dm - origin) / np.linalg.norm(proj_dm - origin)
                
                x = np.cross(z, y)

                # the pose of the hole in camera
                trans_hole2camera = np.identity(4)
                trans_hole2camera[0, 0:3] = x
                trans_hole2camera[1, 0:3] = y
                trans_hole2camera[2, 0:3] = z
                trans_hole2camera[0:3, 3] = origin

                print("recieved: ", rec_list)
                if len(rec_list) != 8:
                    continue
                # the pose of the end in base
                p_end2base = np.array([rec_list[i] for i in range(1, 4)])
                quat_end2base = np.array([rec_list[i] for i in range(4, 8)])
                rot_end2base = Rotation.from_quat(quat_end2base).as_matrix()
                trans_end2base = np.identity(4)
                trans_end2base[0:3, 0:3] = rot_end2base
                trans_end2base[0:3, 3] = p_end2base

                # the pose of the hole in base
                trans_hole2base = trans_end2base * trans_camera2end * trans_hole2camera

                quat_hole2base = Rotation.from_matrix(trans_camera2end[0:3, 0:3]).as_quat()
                
                pub_list = [1, trans_hole2base[0,3], trans_hole2base[1,3], trans_hole2base[2,3],\
                            quat_hole2base[0], quat_hole2base[1], quat_hole2base[2], quat_hole2base[3]]


            else:
                pub_list = [0, 0, 0, 0, 0, 0, 0, 0]
                continue

                
            
            # when the size of the hole in picture is large enough
            # if camera_xyz[-1] < 250:
            if (charge_box[2] - charge_box[0]) > 70:
                # send message to robot, ask it to stop
                pub_list[0] = -1

                # check if the robot has stopped
                if rec_list[0] != -1:
                    cv2.waitKey(200)
                    continue

                # send end pose to cpp
                py2cpp = Py2Cpp(b"/home/zju/realsense_ws/template_pcd/template_list.txt", \
                                rec_list[1], rec_list[2], rec_list[3], \
                                rec_list[4], rec_list[5], rec_list[6], rec_list[7])
                so.set_py2cpp(ctypes.byref(py2cpp))

                # save point cloud from depth image as pcd file
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
        
        # get results from cpp
        so.get_var2py.restype = ctypes.POINTER(Var2Py)
        var2py_ptr = so.get_var2py()
        var2py = var2py_ptr.contents
        
        pub_list = [2, round(var2py.px, 4), round(var2py.py, 4), round(var2py.pz, 4), \
                    round(var2py.qx, 4), round(var2py.qy, 4), round(var2py.qz, 4), round(var2py.qw, 4)]
        print("\n\n\n", pub_list[0])
        print("\n\n\n")

        # continue to send messages to robot
        while True:
            continue

    finally:
        # Stop streaming
        pipeline.stop()