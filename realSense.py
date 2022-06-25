import pyrealsense2 as rs
import numpy as np
import cv2
from rotationEstimatorFromAccAndGyro import rotation_estimator

#旋转任意度数,旋转之后图像的宽高并没有交换
def rotate_img(src,angle):
    # dividing height and width by 2 to get the center of the image
    height, width = src.shape[:2]
    # get the center coordinates of the image to create the 2D rotation matrix
    center = (width / 2, height / 2)
    # using cv2.getRotationMatrix2D() to get the rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=45, scale=1)
    # rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

    # rotate the image using cv2.warpAffine
    rotated_image = cv2.warpAffine(src=src, M=rotate_matrix, dsize=(width, height))
    return rotated_image

# 顺时针旋转90度，旋转之后图像宽高被交换
def Rotate_pos90(src):
    trans_img = cv2.transpose(src)
    new_img = cv2.flip(trans_img, 1)
    return new_img

# 逆时针旋转90度，旋转之后图像宽高被交换
def Rotate_neg90(src):
    trans_img = cv2.transpose(src)
    new_img = cv2.flip(trans_img, 0)
    return new_img

#获取深度图像和彩色图像，并将深度图像对齐待彩色图像，获得彩色图像中心点的深度信息
def get_color_and_depth_align():
    #开启管道
    pipeline = rs.pipeline()
    #配置参数
    cfg = rs.config()
    #配置深度数据流，分辨率-类型-帧率
    #不同分辨率的深度图像所能检测的最小距离不同，分辨率越小检测的最小距离越小
    cfg.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 30)
    # 配置彩色数据流，分辨率-类型-帧率
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(cfg)
    #获得相机句柄
    depth_sensor = profile.get_device().first_depth_sensor()
    #获得深度比例系数
    depth_scale = depth_sensor.get_depth_scale()
    print("深度比例系数为：", depth_scale)
    # 深度比例系数为： 0.0010000000474974513
    # 测试了数个摄像头，发现深度比例系数都相同，甚至D435i的也一样。

    #配置对齐参数
    align = rs.align(rs.stream.color)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            #设置对齐
            aligned_frames = align.process(frames)
            #获取对齐后的深度数据
            aligned_depth_frame = aligned_frames.get_depth_frame()
            #获得彩色图像
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            #转换为cv格式
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # 逆时针旋转90
            color_image_neg90=Rotate_neg90(color_image)
            # cv2.imshow('RealSense_color', color_image)
            cv2.imshow('RealSense_color_neg90', color_image_neg90)
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_neg90 = Rotate_neg90(depth_colormap)
            # cv2.imshow('RealSense_depth', depth_colormap)
            cv2.imshow('RealSense_depth_neg90', depth_colormap_neg90)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)

            #获得对齐后的深度数据的分辨率，与彩色数据的分辨率相同
            width = aligned_depth_frame.get_width()
            height = aligned_depth_frame.get_height()
            # print(width, '', height)
            # 640  480
            # print(int(width / 2))
            # 320

            #获得彩色图像中心点的深度数据
            dist_to_center = aligned_depth_frame.get_distance(int(width / 2), int(height / 2))
            # 不知为什么没有用到深度比例系数
            print(dist_to_center)
            #退出条件
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()

def get_all_data():
    # 开始准备IMU
    imu_pipeline = rs.pipeline()
    imu_config = rs.config()
    imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)  # acceleration 加速度
    imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # gyroscope    陀螺仪
    # 启用IMU
    imu_pipeline.start(imu_config)

    # # 开启管道
    # ir_pipeline = rs.pipeline()
    # # 配置参数
    # ir_cfg = rs.config()
    # # IR左图
    # ir_cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # 默认格式 rs.format.y8
    # # IR右图
    # ir_cfg.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)  # 默认格式 rs.format.y8
    # ir_pipeline.start(ir_cfg)

    # 开启管道
    pipeline = rs.pipeline()
    # 配置参数
    cfg = rs.config()
    # 配置深度数据流，分辨率-类型-帧率
    # 不同分辨率的深度图像所能检测的最小距离不同，分辨率越小检测的最小距离越小
    cfg.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 30)
    # # IR左图
    # cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # 默认格式 rs.format.y8
    # # IR右图
    # cfg.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)  # 默认格式 rs.format.y8
    # 配置彩色数据流，分辨率-类型-帧率
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)  # acceleration 加速度
    # cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # gyroscope    陀螺仪

    profile = pipeline.start(cfg)
    # 获得相机句柄
    depth_sensor = profile.get_device().first_depth_sensor()
    # 获得深度比例系数
    depth_scale = depth_sensor.get_depth_scale()
    print("深度比例系数为：", depth_scale)
    # 深度比例系数为： 0.0010000000474974513
    # 测试了数个摄像头，发现深度比例系数都相同，甚至D435i的也一样。

    # 配置对齐参数
    align = rs.align(rs.stream.color)
    algo = rotation_estimator()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            # ir_frame = ir_pipeline.wait_for_frames()
            # 设置对齐
            aligned_frames = align.process(frames)
            # 获取对齐后的深度数据
            aligned_depth_frame = aligned_frames.get_depth_frame()
            # 获得彩色图像
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # 转换为cv格式
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # 逆时针旋转90
            color_image_neg90 = Rotate_neg90(color_image)
            # cv2.imshow('RealSense_color', color_image)
            cv2.imshow('RealSense_color_neg90', color_image_neg90)
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_neg90 = Rotate_neg90(depth_colormap)
            # cv2.imshow('RealSense_depth', depth_colormap)
            cv2.imshow('RealSense_depth_neg90', depth_colormap_neg90)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)

            # 获得对齐后的深度数据的分辨率，与彩色数据的分辨率相同
            width = aligned_depth_frame.get_width()
            height = aligned_depth_frame.get_height()

            # 获得彩色图像中心点的深度数据
            dist_to_center = aligned_depth_frame.get_distance(int(width / 2), int(height / 2))
            # 不知为什么没有用到深度比例系数
            # print(dist_to_center)

            imu_frames = imu_pipeline.wait_for_frames(200)
            # 获取加速度信息
            accel_frame = imu_frames.first_or_default(rs.stream.accel, rs.format.motion_xyz32f)
            # 获取陀螺仪信息
            gyro_frame = imu_frames.first_or_default(rs.stream.gyro, rs.format.motion_xyz32f)
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            print("type:",type(gyro_data))
            #Get the timestamp of the current frame
            ts = accel_frame.as_motion_frame().get_timestamp()
            #Call function that computes the angle of motion based on the retrieved measures
            algo.process_gyro(gyro_data, ts)
            #Call function that computes the angle of motion based on the retrieved measures
            algo.process_accel(accel_data)
            print("algo--------",algo.get_theta())

            # 打印IMU信息
            print("\taccel = {}, \n\tgyro = {}".format(str(accel_frame.as_motion_frame().get_motion_data()),
                                                     str(gyro_frame.as_motion_frame().get_motion_data())))

            # # 等待IMU数据
            # imu_frames = imu_pipeline.wait_for_frames()
            # 获取加速度信息
            # accel_frame = frames.first_or_default(rs.stream.accel, rs.format.motion_xyz32f)
            # 获取陀螺仪信息
            # gyro_frame = frames.first_or_default(rs.stream.gyro, rs.format.motion_xyz32f)
            # 打印IMU信息
            # print("accel = {}, \n\tgyro = {}".format(str(accel_frame.as_motion_frame().get_motion_data()),
            #                                          str(gyro_frame.as_motion_frame().get_motion_data())))
            # 获取IR左右图像对
            # ir_frame = ir_pipeline.wait_for_frames()
            # frame_left = ir_frame.get_infrared_frame(1)
            # frame_right = ir_frame.get_infrared_frame(2)
            # left_image = np.asarray(frame_left.get_data())
            # right_image = np.asarray(frame_right.get_data())
            # cv2.imshow("left_image",left_image)
            # 退出条件
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


def get_color_and_depth():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # Start streaming
    pipeline.start(config)
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays

            depth_image = np.asanyarray(depth_frame.get_data())

            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
if __name__ == "__main__":
    # get_color_and_depth()
    # get_color_and_depth_align()
    get_all_data()

