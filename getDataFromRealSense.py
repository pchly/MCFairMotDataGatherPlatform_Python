import pyrealsense2 as rs
import numpy as np
import cv2
from rotationEstimatorFromAccAndGyro import rotation_estimator

class getDataFromRealSense(object):
    def __init__(self, color_width=640, color_height=480,color_frame_rate= 30,depth_frame_rate=30, depth_width=480,depth_height=270):
        self.color_width = color_width
        self.color_height = color_height
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.color_frame_rate = color_frame_rate
        self.depth_frame_rate = depth_frame_rate
    def config_sensor(self):

        # 开始准备IMU
        self.imu_pipeline = rs.pipeline()
        self.imu_config = rs.config()
        self.imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)  # acceleration 加速度
        self.imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)  # gyroscope    陀螺仪
        # 启用IMU
        self.imu_pipeline.start(self.imu_config)

        # 开启管道
        self.pipeline = rs.pipeline()
        # 配置参数
        cfg = rs.config()
        # 配置深度数据流，分辨率-类型-帧率
        # 不同分辨率的深度图像所能检测的最小距离不同，分辨率越小检测的最小距离越小
        cfg.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.depth_frame_rate)
        # 配置彩色数据流，分辨率-类型-帧率
        cfg.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.color_frame_rate)

        self.profile = self.pipeline.start(cfg)
        if not self.profile:
            print("no device!!")
            return 0
        # 获得相机句柄
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        # 获得深度比例系数
        depth_scale = self.depth_sensor.get_depth_scale()
        print("深度比例系数为：", depth_scale)
        # 深度比例系数为： 0.0010000000474974513
        # 测试了数个摄像头，发现深度比例系数都相同，甚至D435i的也一样。
        # 配置对齐参数
        self.align = rs.align(rs.stream.color)
        self.algo = rotation_estimator()
    # 获取深度图像和彩色图像，并将深度图像对齐待彩色图像，获得彩色图像中心点的深度信息
    def getFrames(self):
        frames = self.pipeline.wait_for_frames()
        # 设置对齐
        aligned_frames = self.align.process(frames)
        # 获取对齐后的深度数据
        aligned_depth_frame = aligned_frames.get_depth_frame()
        # 获得彩色图像
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            # continue
            print("aligned_depth_frame or color_frame empty!!!")
        # 转换为cv格式
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # 逆时针旋转90
        color_image_neg90 = self.Rotate_neg90(color_image)
        # cv2.imshow('RealSense_color', color_image)
        # cv2.imshow('RealSense_color_neg90', color_image_neg90)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap_neg90 = self.Rotate_neg90(depth_colormap)
        depth_nomap_neg90 = self.Rotate_neg90(depth_image)
        # cv2.imshow('RealSense_depth', depth_colormap)
        # cv2.imshow('RealSense_depth_neg90', depth_colormap_neg90)
        # Stack both images horizontally
        # images = np.hstack((color_image, depth_colormap))
        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', images)

        # 获得对齐后的深度数据的分辨率，与彩色数据的分辨率相同
        self.depth_width_after_align = aligned_depth_frame.get_width()
        self.depth_height_after_align = aligned_depth_frame.get_height()
        # print(width, '', height)
        # 640  480
        # print(int(width / 2))
        # 320

        # 获得彩色图像中心点的深度数据
        dist_to_center = aligned_depth_frame.get_distance(int(self.depth_width_after_align / 2), int(self.depth_height_after_align / 2))
        # 不知为什么没有用到深度比例系数
        # print(dist_to_center)

        imu_frames = self.imu_pipeline.wait_for_frames(200)
        # 获取加速度信息
        accel_frame = imu_frames.first_or_default(rs.stream.accel, rs.format.motion_xyz32f)
        # 获取陀螺仪信息
        gyro_frame = imu_frames.first_or_default(rs.stream.gyro, rs.format.motion_xyz32f)
        accel_data = accel_frame.as_motion_frame().get_motion_data()
        gyro_data = gyro_frame.as_motion_frame().get_motion_data()
        # Get the timestamp of the current frame
        # self.algo.process_accel(accel_data)
        # ts = accel_frame.as_motion_frame().get_timestamp()
        # self.algo.process_gyro(gyro_data, ts)

        # Call function that computes the angle of motion based on the retrieved measures
        # self.algo.process_gyro(gyro_data, ts)
        # Call function that computes the angle of motion based on the retrieved measures
        # self.algo.process_accel(accel_data)


        # 打印IMU信息
        # print("\taccel = {}, \n\tgyro = {}".format(str(accel_frame.as_motion_frame().get_motion_data()),
        #                                            str(gyro_frame.as_motion_frame().get_motion_data())))
        self.algo.Update_IMU(accel_data.x,accel_data.y,accel_data.z,gyro_data.x,gyro_data.y,gyro_data.z)

        # ts = gyro_frame.as_motion_frame().get_timestamp()
        # self.algo.imutest(ts,gyro_data,accel_data)

        angles = self.algo.get_theta()
        # print("algo--------", angles)
        return color_image_neg90 , depth_nomap_neg90, depth_colormap_neg90 ,gyro_data,angles

    def stop_sensor(self):
        self.pipeline.stop()
        self.imu_pipeline.stop()
    # 逆时针旋转90度，旋转之后图像宽高被交换
    def Rotate_neg90(self,src):
        trans_img = cv2.transpose(src)
        new_img = cv2.flip(trans_img, 0)
        return new_img

