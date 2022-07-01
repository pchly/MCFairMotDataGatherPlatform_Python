import sys
import os
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QApplication, QMainWindow,QFileDialog,QProgressBar,QMessageBox
from PyQt5.QtCore import QThread, QObject, QTimer, pyqtSignal, pyqtSlot ,QTimer,QDateTime,QUrl
import mainWindowGUI
import time
import pygame.camera
import cv2
import json
import glob
import shutil
import random
import threading
import queue
import math
from clipVideo import clipvideo
from dataSetClipsToTaV import dataSetClipsTaV
from ContoursPCA import getRotateDegreeUsingPCA

from getDataFromRealSense import getDataFromRealSense



class mainWindow(QMainWindow):
    def __init__(self):
        self.app = QApplication(sys.argv)
        super().__init__()
        #调用自动转换好的界面类的对象
        self.ui = mainWindowGUI.Ui_MainWindow()
        self.ui.setupUi(self)

        self.timer_camera = QtCore.QTimer()  # 定时器
        self.timer_processBar = QtCore.QTimer()  # 定时器
        self.startGatherTome_ms = 0.0
        self.stopGatherTime_ms = 0.0
        self.clipVideo_inst = None
        self.datasetclip_inst = None
        self.cap = cv2.VideoCapture()  # cv获取图像
        self.realSenseSensor = None #realSense获取图像
        self.CAM_NUM = 0
        self.frame_size=(0,0)
        self.FPS=0
        self.saveVideo_filepath = "video_0001.mp4"  # 保存视频的路径和名字
        self.isSaveVideoChecked = False
        self.isSaveVideoOutSeted = False
        self.trainRate=self.ui.spinBox_trainRate.value()
        self.valRate=self.ui.spinBox_valRate.value()
        self.saveVideo_dir = "./visdrone_mcmot/videos/"
        self.saveDepthVideo_dir = "./visdrone_mcmot/depth_videos/"
        self.out_images_seq_path = "./visdrone_mcmot/images/"
        self.delete_images_path = "./visdrone_mcmot/deleteImages/"
        self.pico_images_path = "./visdrone_mcmot/pico_labelme_imgs/images"
        self.pico_annotations_path = "./visdrone_mcmot/pico_labelme_imgs/annotations"
        self.pico_delete_images = "./visdrone_mcmot/pico_labelme_imgs/deleteImgs"
        self.imagesCountAll = 0
        self.imagesDeleteCount = 0
        self.imagesSurplusCount = 0
        self.PICOimagesCountAll = 0
        self.PICOimagesDeleteCount = 0
        self.PICOimagesSurplusCount = 0

        self.train_image_list = []
        self.val_image_list = []
        self.annotations_path = "./visdrone_mcmot/annotations/"
        self.haveFaultAnno = False
        self.labels_with_ids_path = "./visdrone_mcmot/labels_with_ids/"
        self.num_of_allcheck = 5
        self.num_of_check = 0
        self.checkImgIndex = 0
        self.FrameGapToClipVideo = 10
        self.label_list = []
        self.numFramePreVideo = 10


        self.testPeoples = ["YuntaoLi","ChunyuanShi"]
        self.testPeople = "YuntaoLi"
        self.testPose = "stand"
        self.testGroupAll = 2
        self.testGroupCurrent = 0
        self.testNumAll =13
        self.testNumCurrent = 0
        self.testObject = ['Bottled','Bagged','Boxed','Canned','Mug','Vials']
        self.testObjectPart = ['Bottled_top','Bottled_side','Bagged_top','Bagged_side','Boxed_top','Boxed_side','Canned_top','Canned_side','Mug_top','Mug_handle','Mug_side','Vials_top','Vials_side']
        self.sitGroupTestOrder = []
        self.standGroupTestOrder = []
        self.currentGroup = []
        self.testResults = ['unknow','succeed','failure']
        self.startTestTime_ms = 0.0
        self.stopTestTime_ms = 0.0
        self.testTime_ms = 0.0
        self.timer_testRecord = QtCore.QTimer()  # 定时器

        self.usbQueueImage = queue.Queue(maxsize=4)
        self.realSenseColorImages = queue.Queue(maxsize=4)
        self.realSenseDepthImages = queue.Queue(maxsize=4)
        # self.label_list=['Bottled','Bagged','Boxed','Canned','Mug','Vials']
        # 初始化
        self.init_ui()

    # ui初始化
    def init_ui(self):

        #读取label_list.txt文件
        with open("./visdrone_mcmot/label_list.txt", "r", encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line= line.split("\n")[0]
            self.label_list.append(line)
            print(self.label_list)
        # 初始化方法，这里可以写按钮绑定等的一些初始函数
        self.pbar = QProgressBar(self)


        # 界面初始值设定
        self.ui.spinBox_allcheck.setValue(self.num_of_allcheck)
        self.ui.label_num_of_allcheck.setText(str(self.num_of_allcheck))
        self.ui.label_checkImgIndex.setText(str(self.checkImgIndex))
        self.ui.spinBox_frameGap.setValue(self.FrameGapToClipVideo)
        #按钮槽函数绑定
        self.ui.toolButton_scanCamera.clicked.connect(self.click_toolButton_scanCamera)
        self.ui.comboBox_cameraDeviceList.currentTextChanged.connect(self.comboBox_activated_cameraDeviceList)
        self.ui.toolButton_openCamera.clicked.connect(self.click_button_openCamera)
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_processBar.timeout.connect(self.processBarFlesh)
        self.ui.toolButton_startGather.clicked.connect(self.click_toolButton_startGather)
        self.ui.toolButton_stopGather.clicked.connect(self.click_toolButton_stopGather)
        self.ui.toolButton_closeCamera.clicked.connect(self.click_button_closeCamera)
        self.ui.checkBox_saveVideo.stateChanged.connect(self.checkBox_saveVideo_stateChanged)
        # self.ui.toolButton_changeSaveDirAndName.clicked.connect(self.click_toolButton_changeSaveDirAndName)
        self.ui.toolButton_changeDir.clicked.connect(self.click_toolButton_changeDir)
        self.ui.toolButton_datasetClip.clicked.connect(self.click_toolButton_datasetClip)
        self.ui.spinBox_trainRate.valueChanged.connect(self.valueChanged_spinBox_trainRate)
        self.ui.spinBox_valRate.valueChanged.connect(self.valueChanged_spinBox_valRate)
        self.ui.spinBox_frameGap.valueChanged.connect(self.valueCahnged_spinBox_frameGap)
        self.ui.toolButton_clipVideo.clicked.connect(self.click_toolButton_clipVideo)
        self.ui.toolButton_VideoAnalyse.clicked.connect(self.click_toolButton_VideoAnalyse)
        self.ui.toolButton_filterData.clicked.connect(self.click_toolButton_filterData)
        self.ui.toolButton_generateAnnotation.clicked.connect(self.click_toolButton_generateAnnotation)
        self.ui.toolButton_checkFromJson.clicked.connect(self.click_toolButton_checkFromJson)
        self.ui.toolButton_jsonToTxt.clicked.connect(self.click_toolButton_jsonToTxt)
        self.ui.toolButton_checkNextImg.clicked.connect(self.click_toolButton_checkNextImg)
        self.ui.toolButton_checkLastImg.clicked.connect(self.click_toolButton_checkLastImg)
        self.ui.spinBox_allcheck.valueChanged.connect(self.valueChanged_num_of_allcheck)
        self.ui.toolButton_checkFromTxt.clicked.connect(self.click_toolButton_checkFromTxt)
        self.ui.toolButton_generateImgList.clicked.connect(self.click_toolButton_generateImgList)
        self.ui.toolButton_picodatafilter.clicked.connect(self.click_tooloButton_picodatafilter)
        self.ui.toolButton_generatePico.clicked.connect(self.click_toolButton_generatePico)
        self.ui.toolButton_startTest.clicked.connect(self.click_toolButton_startTest)
        self.ui.spinBox_picodata_num.valueChanged.connect(self.valueChangeed_spinBox_picodata_num)
        self.ui.toolButton_transToCOCO.clicked.connect(self.click_toolButton_transToCOCO)

        self.ui.comboBox_testPeople.currentTextChanged.connect(self.textChanged_comboBox_testPeople)
        self.ui.toolButton_nextTest.clicked.connect(self.click_toolButton_nextTest)
        self.ui.toolButton_lastTest.clicked.connect(self.click_toolButton_lastTest)
        #菜单项的槽函数绑定
        self.ui.actionOpenFile.triggered.connect(self.menu_click_OpenFile)
        self.ui.actionOpenFile.setShortcut('Ctrl+O')#设置菜单项的快捷键

        self.ui.comboBox_testPeople.addItems(self.testPeoples)
        self.ui.spinBox_testGroupAll.setValue(self.testGroupAll)
        self.ui.spinBox_testNumAll.setValue(self.testNumAll)
        self.ui.comboBox_testResult.addItems(self.testResults)
        self.ui.toolButton_scanCamera_test.clicked.connect(self.click_toolButton_scanCamera_test)
        self.ui.comboBox_cameraDeviceList_wrist.currentTextChanged.connect(self.comboBox_activated_cameraDeviceList_wrist)
        self.ui.comboBox_cameraDeviceList_record.currentTextChanged.connect(self.comboBox_activated_cameraDeviceList_record)
        self.timer_testRecord.timeout.connect(self.timer_testRecord_timeout)
        # #设置label显示图像
        # icon = QPixmap(":/toolButton/static/endJaw.png")
        # self.ui.label_show.setPixmap(icon)
        # self.ui.label_show.setScaledContents(True) #图像缩放至label大小

        self.show()
    def comboBox_activated_cameraDeviceList(self,text):
        print("text:",text)
        # if text == "Integrated Camera":
        #     self.CAM_NUM = self.camera_id_lis.index("Integrated Camera")
        #     print("Cam_num :",self.CAM_NUM)
        # elif text == "LRCP USB2.0" :
        #     self.CAM_NUM = self.camera_id_lis.index("LRCP USB2.0")
        #     print("Cam_num :", self.CAM_NUM)
        # elif text == "Intel(R) RealSense(TM) Depth Camera 435i Depth" or "Intel(R) RealSense(TM) Depth Camera 435i RGB" :
        #     self.CAM_NUM = 10
        #     print("Cam_num :", self.CAM_NUM)

        if text == "Intel(R) RealSense(TM) Depth Camera 435i Depth" or text == "Intel(R) RealSense(TM) Depth Camera 435i RGB" :
            self.CAM_NUM = 10
            print("Cam_num :", self.CAM_NUM)
        else:
            self.CAM_NUM = self.camera_id_lis.index(text)
            print("Cam_num :", self.CAM_NUM)
    def click_toolButton_scanCamera(self):
        pygame.camera.init()
        self.camera_id_lis = pygame.camera.list_cameras()
        print(self.camera_id_lis)
        self.ui.comboBox_cameraDeviceList.addItems(self.camera_id_lis)
    def click_button_openCamera(self):

        if self.timer_camera.isActive() == False:
            if self.CAM_NUM == 10:
                self.realSenseSensor = getDataFromRealSense()
                self.realSenseSensor.config_sensor()
                self.frame_size = (int(self.realSenseSensor.color_height), int(self.realSenseSensor.color_width))  # 获取摄像头分辨率
                self.FPS = self.realSenseSensor.color_frame_rate  # 获取摄像头帧率
                self.timer_camera.start(30)
            else:
                flag = self.cap.open(self.CAM_NUM)
                print("open camera by cv!")
                if flag == False:
                    msg = QtWidgets.QMessageBox.warning(
                        self, u"Warning", u"请检测相机与电脑是否连接正确",
                        buttons=QtWidgets.QMessageBox.Ok,
                        defaultButton=QtWidgets.QMessageBox.Ok)
                else:
                    self.frame_size = (int(self.cap.get(3)), int(self.cap.get(4)))  # 获取摄像头分辨率
                    self.FPS = self.cap.get(5)  # 获取摄像头帧率
                    self.timer_camera.start(30)

    def show_camera(self):

        if self.CAM_NUM == 10:
            self.image,self.depth_img_nomap,self.depth_img_colormap,self.gyro,self.sensor_angles = self.realSenseSensor.getFrames()
            show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        else:
            flag, self.image = self.cap.read()
            self.image = cv2.flip(self.image, 1) # 左右翻转
            show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        if self.isSaveVideoChecked:
            if not(self.isSaveVideoOutSeted):
                # 保存视频
                code = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 编码格式
                save_fps = 30  # 保存视频的帧率
                time = QDateTime.currentDateTime()  # 获取现在的时间
                timeplay = time.toString('yyyy_MM_dd_hh_mm_ss')  # 设置显示时间的格式
                self.saveVideo_filepath = self.saveVideo_dir + "video_" + timeplay + ".mp4"
                self.saveDepthVideo_filepath = self.saveDepthVideo_dir + "video_" + timeplay + ".mp4"
                # print(self.saveVideo_filepath)
                self.out = cv2.VideoWriter(self.saveVideo_filepath, code, save_fps, self.frame_size,isColor=True)  # 保存视频的视频流
                if self.CAM_NUM == 10:
                    print(self.saveDepthVideo_filepath)
                    self.depth_out = cv2.VideoWriter(self.saveDepthVideo_filepath, code, save_fps, self.frame_size,
                                                     isColor=False)  # 保存深度视频的视频流

                self.isSaveVideoOutSeted = True
            if self.out.isOpened():  # 判断视频流是否创建成功
                # 设置记录时间
                currentTime = QDateTime.currentDateTime()  # 获取现在的时间
                self.stopGatherTime_ms = currentTime.toMSecsSinceEpoch()
                self.gatherTime = (self.stopGatherTime_ms - self.startGatherTome_ms) / 1000
                self.ui.label_gatherTime.setText(str(self.gatherTime))
                print('out is  opened')
                self.out.write(self.image)
                # self.depth_out.write(self.depth_img_colormap)
                print('out is  writing...')
            if self.CAM_NUM == 10:
                if self.depth_out.isOpened():  # 判断视频流是否创建成功
                    print("shendu视频流创建成功")
                    self.depth_out.write(self.depth_img_nomap)

        if self.CAM_NUM == 10:
            color_and_depth_images = np.hstack((show, self.depth_img_colormap))
            showImage = QtGui.QImage(color_and_depth_images.data, color_and_depth_images.shape[1], color_and_depth_images.shape[0], QtGui.QImage.Format_RGB888)

            #设置角速度和欧拉角
            self.ui.doubleSpinBox_gyrox.setValue(self.gyro.x)
            self.ui.doubleSpinBox_gyroy.setValue(self.gyro.y)
            self.ui.doubleSpinBox_gyroz.setValue(self.gyro.z)
            self.ui.doubleSpinBo_x.setValue(self.sensor_angles[1])
            self.ui.doubleSpinBox_y.setValue(self.sensor_angles[0])
            self.ui.doubleSpinBox_z.setValue(self.sensor_angles[2])
        else:
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.ui.label_imgShow.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.ui.label_imgShow.setScaledContents(True)

    def click_toolButton_startGather(self):
        currentTime = QDateTime.currentDateTime()  # 获取现在的时间
        self.startGatherTome_ms = currentTime.toMSecsSinceEpoch()
        self.stopGatherTime_ms = self.startGatherTome_ms
        print("ms",self.startGatherTome_ms)
        self.ui.checkBox_saveVideo.setCheckState(QtCore.Qt.Checked)
    def click_toolButton_stopGather(self):
        self.ui.checkBox_saveVideo.setCheckState(QtCore.Qt.Unchecked)
        currentTime = QDateTime.currentDateTime()  # 获取现在的时间
        self.stopGatherTime_ms = currentTime.toMSecsSinceEpoch()
        self.gatherTime = (self.stopGatherTime_ms - self.startGatherTome_ms)/1000
        self.ui.label_gatherTime.setText(str(self.gatherTime))
    def checkBox_saveVideo_stateChanged(self):

        if self.ui.checkBox_saveVideo.checkState() == QtCore.Qt.Checked:
            print("checked")
            self.isSaveVideoChecked = True
        else:
            if self.out.isOpened():
                self.out.release()
            if self.CAM_NUM == 10:
                if self.depth_out.isOpened():
                    self.depth_out.release()
            self.isSaveVideoChecked = False
            self.isSaveVideoOutSeted = False
            print("unchecked")

    # def toolButton_changeSaveDirAndName(self):
    #     saveVideo_file_path = QFileDialog.getSaveFileName(self, "保存视频文件", os.getcwd() + "/未命名", "mp4 files (*.mp4)")
    #     print(saveVideo_file_path)
    #     self.saveVideo_filepath=saveVideo_file_path[0]

    def click_toolButton_changeDir(self):
        saveVideodir=QFileDialog.getExistingDirectory(self, "choose saveVideo Directory","./");
        self.saveVideo_dir=saveVideodir + "/"
        print(self.saveVideo_dir)


    def click_button_closeCamera(self):
        if self.timer_camera.isActive() != False:
            ok = QtWidgets.QPushButton()
            cacel = QtWidgets.QPushButton()

            msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

            msg.addButton(ok,QtWidgets.QMessageBox.ActionRole)
            msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
            ok.setText(u'确定')
            cacel.setText(u'取消')

            if msg.exec_() != QtWidgets.QMessageBox.RejectRole:
                if self.realSenseSensor:
                    self.realSenseSensor.stop_sensor()
                    self.realSenseSensor = None
                    print("stop realSense!")
                if self.cap.isOpened():
                    self.cap.release()
                if self.timer_camera.isActive():
                    self.timer_camera.stop()
                if self.isSaveVideoChecked == True :
                    self.out.release()
                    self.depth_out.release()
                    self.isSaveVideoOutSeted = False
                # self.label_face.setText("<html><head/><body><p align=\"center\"><img src=\":/newPrefix/pic/Hint.png\"/><span style=\" font-size:28pt;\">点击打开摄像头</span><br/></p></body></html>")


    def menu_click_OpenFile(self):
        file_name = QFileDialog.getOpenFileName(self, "选择图像文件", os.getcwd(), "image files(*.png *.xpm *.jpg);;jpg files(*.jpg);;png files(*.png)")
        print(file_name)

    def click_toolButton_datasetClip(self):
        if self.datasetclip_inst == None:
            self.datasetclip_inst = dataSetClipsTaV(self.saveVideo_dir, self.trainRate,self.valRate)
        self.datasetclip_inst.run()

    def valueChanged_spinBox_trainRate(self,trainRate_value):
        self.trainRate = trainRate_value
    def valueChanged_spinBox_valRate(self,valRate_value):
        self.valRate = valRate_value
    def valueCahnged_spinBox_frameGap(self,FGapValue):
        self.FrameGapToClipVideo = FGapValue
    def click_toolButton_clipVideo(self):
        if self.timer_processBar.isActive() == False:
            self.timer_processBar.start(100)
        self.ui.statusbar.addWidget(self.pbar)
        # self.out_images_seq_path = QFileDialog.getExistingDirectory(self, "choose images Directory", "./");
        if self.clipVideo_inst == None:
            self.clipVideo_inst = clipvideo(self.saveVideo_dir, self.out_images_seq_path, self.FrameGapToClipVideo)
        # self.pbar.setValue(self.clip.haveClipRate)
        self.clipVideo_inst.run()
        self.timer_processBar.stop()
        self.pbar.setValue(self.clipVideo_inst.haveClipRate)
        # cv2.putText(img, str, (123,456)), font, 2, (0,255,0), 3)
        # 各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    def click_toolButton_VideoAnalyse(self):
        print("123")
        dataSetPart = ["train/", "val/","test/"]
        first_video = True
        self.haveAnalyseNum = 0
        self.averange_frame_count = 0
        self.max_frame_count = 0
        self.min_frame_count = 0
        self.averange_duration = 0
        self.max_duration = 0
        self.min_duration = 0

        all_frame = 0
        all_duration = 0
        self.train_viseos_num = len(glob.glob(self.saveVideo_dir + "train/" + '*.mp4'))
        self.val_viseos_num = len(glob.glob(self.saveVideo_dir + "val/" + '*.mp4'))
        self.test_viseos_num = len(glob.glob(self.saveVideo_dir + "test/" + '*.mp4'))
        self.allVideosNum = self.train_viseos_num + self.val_viseos_num + self.test_viseos_num
        print("all videos num :", self.allVideosNum)
        # 依次读取文件名
        for videos_part in dataSetPart:
            videos_files = os.listdir(self.saveVideo_dir + videos_part)
            len_of_videos = len(videos_files)
            for video_name in videos_files:
                # print("video_name:", video_name)
                # # 对文件名进行拆分处理
                # file_name = video_name.split('.')[0]
                # folder_name = self.save_path + "/" + videos_part + file_name  # 构成新目录存放视频帧
                # os.makedirs(folder_name, exist_ok=True)  # 查看是否存在目录，否则创建
                print("video_path:", self.saveVideo_dir + videos_part + video_name)
                vc = cv2.VideoCapture(self.saveVideo_dir + videos_part + video_name)
                # 获取视频帧率
                fps = vc.get(cv2.CAP_PROP_FPS)
                frame_count = vc.get(cv2.CAP_PROP_FRAME_COUNT)

                duration = frame_count / fps
                if first_video:
                    self.max_frame_count = frame_count
                    self.min_frame_count = frame_count
                    self.max_duration = duration
                    self.min_duration = duration
                    first_video = False
                if frame_count > self.max_frame_count:
                    self.max_frame_count = frame_count
                if frame_count < self.min_frame_count:
                    self.min_frame_count = frame_count
                if duration > self.max_duration:
                    self.max_duration = duration
                if duration < self.min_duration:
                    self.min_duration = duration

                all_frame = all_frame + frame_count
                all_duration = all_duration + duration

                print("fps:", fps)
                # 判断视频是否可以打开
                rval = vc.isOpened()
                vc.release()
                self.haveAnalyseNum = self.haveAnalyseNum + 1
                self.haveAnalyseRate = 100 * (self.haveAnalyseNum) / self.allVideosNum
        self.averange_frame_count = (all_frame - self.max_frame_count - self.min_frame_count) / (self.allVideosNum - 2)
        self.averange_duration = (all_duration -self.max_duration - self.min_duration) / (self.allVideosNum - 2)

        self.ui.label_maxseq.setText(str(self.max_frame_count / self.FrameGapToClipVideo))
        self.ui.label_minseq.setText(str(self.min_frame_count / self.FrameGapToClipVideo))
        self.ui.label_averangeseq.setText(str(self.averange_frame_count / self.FrameGapToClipVideo))
        print("22222")
        self.ui.label_allVideos.setText(str(self.allVideosNum))
        self.ui.label_trainVideos.setText(str(self.train_viseos_num))
        self.ui.label_valVideos.setText(str(self.val_viseos_num))
        self.ui.label_testVideos.setText(str(self.test_viseos_num))
        print("22222")
        self.ui.label_maxframeCount.setText(str(self.max_frame_count))
        self.ui.label_minframeCount.setText(str(self.min_frame_count))
        self.ui.label_averangeframeCount.setText(str(self.averange_frame_count))
        print("22222")
        self.ui.label_maxDuration.setText(str(self.max_duration))
        self.ui.label_minDuration.setText(str(self.min_duration))
        self.ui.label_averangeDuration.setText(str(self.averange_duration))
        print("22222")
        print("self.averang_frame_count:", self.averange_frame_count)
        print("self.max_frame_count:", self.max_frame_count)
        print("self.min_frame_count:", self.min_frame_count)
        print("self.averange_duration:",self.averange_duration)
        print("self.max_duration:", self.max_duration)
        print("self.min_duration:", self.min_duration)

    def processBarFlesh(self):
        self.pbar.setValue(self.clipVideo_inst.haveClipRate)
    def show_label_from_json(self,img_path, json_d):
        src_img = cv2.imread(img_path)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for item in json_d["shapes"]:
            # print(item['points'])
            point = item['points']
            id = item['group_id']
            label = item['label']
            if label not in self.label_list:
                self.ui.comboBox_labelList.addItem(label)
                self.label_list.append(label)
                print(self.label_list)
                self.ui.label_numofClasses.setText(str(len(self.label_list)))

            putTextStr = "%s:%s" % (label, id)
            p1 = (int(point[0][0]), int(point[0][1]))
            p2 = (int(point[1][0]), int(point[1][1]))
            cv2.rectangle(src_img, p1, p2, (0, 255, 0), 1)
            cv2.putText(src_img, putTextStr, p1, font, 0.5, (0, 255, 255), 1)

        showImage = QtGui.QImage(src_img.data, src_img.shape[1], src_img.shape[0], QtGui.QImage.Format_RGB888)
        self.checkedJsonImgs.append(showImage)
        with open("./visdrone_mcmot/label_list.txt", 'w') as f:
            for item in self.label_list:
                f.write(item)
                # f.write(" \n")
                f.write("\n")
        if self.num_of_check == self.num_of_allcheck:
            self.ui.label_checkImgIndex.setText(str(self.checkImgIndex + 1))
            self.ui.label_imgShow.setPixmap(QtGui.QPixmap.fromImage(self.checkedJsonImgs[self.checkImgIndex]))
            self.ui.label_imgShow.setScaledContents(True)


    def show_label_from_txt(self, img_path, txt_path):
        src_img = cv2.imread(img_path)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = src_img.shape[:2]
        with open(txt_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            data = line.split(' ')
            label = self.label_list[int(data[0])]
            id = data[1]
            putTextStr = "%s:%s" % (label, id)
            x1 = int((float(data[2]) - float(data[4]) / 2) * w)
            y1 = int((float(data[3]) - float(data[5]) / 2) * h)
            x2 = int((float(data[2]) + float(data[4]) / 2) * w)
            y2 = int((float(data[3]) + float(data[5]) / 2) * h)
            p1 = (x1, y1)
            p2 = (x2, y2)
            print(p1,p2)
            cv2.rectangle(src_img, p1, p2, (0, 250, 0), 1)
            cv2.putText(src_img, putTextStr, p1, font, 0.5, (0, 255, 255), 1)

        showImage = QtGui.QImage(src_img.data, src_img.shape[1], src_img.shape[0], QtGui.QImage.Format_RGB888)
        self.checkedJsonImgs.append(showImage)

        if self.num_of_check == self.num_of_allcheck:
            self.ui.label_checkImgIndex.setText(str(self.checkImgIndex + 1))
            self.ui.label_imgShow.setPixmap(QtGui.QPixmap.fromImage(self.checkedJsonImgs[self.checkImgIndex]))
            self.ui.label_imgShow.setScaledContents(True)

        # cv2.imshow(window_name, src_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # return
    def cut_roi_from_txt(self, img_path, txt_path):
        src_img = cv2.imread(img_path)
        # src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w = src_img.shape[:2]
        with open(txt_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
        txt_file_name = txt_path.split("\\")[-1].split(".txt")[0]
        print("txt_file_name:",txt_file_name)
        pico_img_path = txt_path.replace(".txt",".jpg").replace("images","cut_roi_pico_img").replace("labels_with_ids","pico_labelme_imgs")
        pico_img_path = pico_img_path.split(".jpg")[0]
        separate_path_to_build = pico_img_path.split("\\")[0] + "/" + pico_img_path.split("\\")[1]

        print("dir :",separate_path_to_build)
        if not os.path.isdir(separate_path_to_build):
            os.makedirs(separate_path_to_build)

        for line in lines:
            data = line.split(' ')
            label = self.label_list[int(data[0])]
            id = data[1]
            putTextStr = "%s:%s" % (label, id)
            x1 = int((float(data[2]) - float(data[4]) / 2) * w)
            y1 = int((float(data[3]) - float(data[5]) / 2) * h)
            x2 = int((float(data[2]) + float(data[4]) / 2) * w)
            y2 = int((float(data[3]) + float(data[5]) / 2) * h)
            p1 = (x1, y1)
            p2 = (x2, y2)
            rect=(p1,p2)
            print(rect)
            pico_img_path = pico_img_path + "_" + label + id + ".png"
            uni_pico_img_path = self.uni_img_path_to_build + txt_file_name + "_" + label + id + ".png"
            print("pico_img_path:", pico_img_path)
            pico_img = src_img[y1:y2, x1:x2]
            print("999")
            pico_img = cv2.resize(pico_img, (300, 400), interpolation=cv2.INTER_LINEAR)
            print("888")
            cv2.imwrite(pico_img_path, pico_img)
            cv2.imwrite(uni_pico_img_path, pico_img)
            print("77")
            # cv2.rectangle(src_img, p1, p2, (0, 250, 0), 1)
            # cv2.putText(src_img, putTextStr, p1, font, 0.5, (0, 255, 255), 1)

    def valueChangeed_spinBox_picodata_num(self,picodata_num):
        self.numFramePreVideo = picodata_num
    def click_toolButton_generatePico(self):
        self.uni_img_path_to_build = "./visdrone_mcmot/pico_labelme_imgs/images/"
        self.uni_anno_path_to_build = "./visdrone_mcmot/pico_labelme_imgs/annotations/"
        if not os.path.isdir(self.uni_img_path_to_build):
            os.makedirs(self.uni_img_path_to_build)
        if not os.path.isdir(self.uni_anno_path_to_build):
            os.makedirs(self.uni_anno_path_to_build)
        for train_val in os.listdir(self.labels_with_ids_path):
            train_val_dir = os.path.join(self.labels_with_ids_path, train_val)
            # 如果是一个子目录就继续for循环，跳过下面的步骤，继续下一个循环
            if os.path.isdir(train_val_dir):
                for videos in os.listdir(train_val_dir):
                    videos_dir = os.path.join(train_val_dir, videos)
                    num_of_roi = 0
                    for video_images in os.listdir(videos_dir):
                        video_images_dir = os.path.join(videos_dir, video_images)
                        self.num_of_check = self.num_of_check + 1
                        print("txt_path:\t", video_images_dir)
                        img_name = video_images_dir.split("\\")[-1].split(".")[0] + ".jpg"
                        img_train_val_path = os.path.join(self.out_images_seq_path, train_val)
                        img_video_path = os.path.join(img_train_val_path, videos)
                        img_path = os.path.join(img_video_path, img_name)
                        print("img_path:\t", img_path)
                        num_of_roi = num_of_roi + 1
                        if num_of_roi <= self.numFramePreVideo:
                            self.cut_roi_from_txt(img_path, video_images_dir)
    def click_toolButton_transToCOCO(self):
        os.system("python x2coco.py --dataset_type=labelme --json_input_dir=./visdrone_mcmot/pico_labelme_imgs/annotations --image_input_dir=./visdrone_mcmot/pico_labelme_imgs/images --output_dir=./visdrone_mcmot/pico_labelme_imgs/coco --train_proportion=0.8 --val_proportion=0.1 --test_proportion=0.1")  # 运行python文件

    def click_toolButton_generateImgList(self):
        # for seq in sorted(glob.glob('./visdrone_mcmot/images/train' + "/*")):
        for train_val in os.listdir("visdrone_mcmot/images/"):
            for seq in os.listdir('visdrone_mcmot/images/' + train_val):
                print("seq---:", seq)
                # for image in glob.glob(seq + "/video" + '*.jpg'):
                for image in os.listdir("visdrone_mcmot/images/" + train_val + "/" + seq):
                    print("image", image)
                    # image = image.replace('PaddleDetection/dataset/mot/','')
                    if train_val == "train":
                        self.train_image_list.append("visdrone_mcmot/images/" + train_val + "/" + seq + "/" + image)
                    if train_val == "val":
                        self.val_image_list.append("visdrone_mcmot/images/" + train_val + "/" + seq + "/" + image)
        with open('visdrone_mcmot/visdrone_mcmot.train', 'w') as image_list_file:
            image_list_file.write(str.join('\n', self.train_image_list))
        with open('visdrone_mcmot/visdrone_mcmot.val', 'w') as image_list_file:
            image_list_file.write(str.join('\n', self.val_image_list))

    def valueChanged_num_of_allcheck(self,value):
        self.num_of_allcheck = value
        self.ui.label_num_of_allcheck.setText(str(value))
    def click_toolButton_checkLastImg(self):
        if self.checkImgIndex != 0:
            self.checkImgIndex = self.checkImgIndex - 1
            self.ui.label_checkImgIndex.setText(str(self.checkImgIndex + 1))
        # else:
        #     self.checkImgIndex = (self.num_of_allcheck - 1)
        self.ui.label_imgShow.setPixmap(QtGui.QPixmap.fromImage(self.checkedJsonImgs[self.checkImgIndex]))
        self.ui.label_imgShow.setScaledContents(True)
    def click_toolButton_checkNextImg(self):
        if self.checkImgIndex != (self.num_of_allcheck - 1):
            self.checkImgIndex = self.checkImgIndex + 1
            self.ui.label_checkImgIndex.setText(str(self.checkImgIndex + 1))
        # else:
        #     self.checkImgIndex = (self.num_of_allcheck - 1)
        self.ui.label_imgShow.setPixmap(QtGui.QPixmap.fromImage(self.checkedJsonImgs[self.checkImgIndex]))
        self.ui.label_imgShow.setScaledContents(True)

    # 保存为相对坐标形式 :label id x_center y_center w h
    def relative_coordinate_txt(self,img_name, json_d, img_path, label_with_id_path,json_path):
        src_img = cv2.imread(img_path)
        h, w = src_img.shape[:2]
        txt_name = img_name.split(".")[0] + ".txt"
        txt_path = os.path.join(label_with_id_path, txt_name)
        print("txt_path: ", txt_path)

        if not os.path.exists(label_with_id_path):  # 判断文件夹是否存在
            os.makedirs(label_with_id_path)  # 新建文件夹
        else:
            print('文件夹已存在...')
        with open(txt_path, 'w') as f:
            for item in json_d["shapes"]:
                # print(item['points'])
                # print(item['label'])
                point = item['points']
                x_center = (point[0][0] + point[1][0]) / 2
                y_center = (point[0][1] + point[1][1]) / 2
                width = point[1][0] - point[0][0]
                hight = point[1][1] - point[0][1]
                # print(x_center)
                label_index = self.label_list.index(item['label'])
                f.write("{} ".format(label_index))
                if item["group_id"] == None:
                    choice = QMessageBox.question(self, u"出现错误的标注null！", "faultAnno at " + json_path,
                                                  QMessageBox.Ok)  # 1
                    if choice == QMessageBox.Ok:  # 2
                       QDesktopServices.openUrl(QUrl.fromLocalFile(json_path));
                       break
                else:
                    f.write("{} ".format(item['group_id']))
                # if self.haveFaultAnno == True:
                #     print("666666")
                #     self.haveFaultAnno = False
                #     return
                f.write("{:.6f} ".format(x_center / w))
                f.write("{:.6f} ".format(y_center / h))
                f.write("{:.6f} ".format(width / w))
                # f.write(" {:.6f} ".format(hight / h))
                f.write("{:.6f}".format(hight / h))
                # f.write(" \n")
                f.write("\n")

    # 保存为绝对坐标形式 :label id x1 y1 x2 y2
    def absolute_coordinate_txt(self,img_name, json_d, img_path):
        src_img = cv2.imread(img_path)
        h, w = src_img.shape[:2]
        txt_name = img_name.split(".")[0] + ".txt"
        txt_path = os.path.join(self.labels_with_ids_path, txt_name)
        print("txt_path:\t", txt_path)
        with open(txt_path, 'w') as f:
            for item in json_d["shapes"]:
                # print(item['points'])
                # print(item['label'])
                point = item['points']
                x1 = point[0][0]
                y1 = point[0][1]
                x2 = point[1][0]
                y2 = point[1][1]
                f.write(" {} ".format(item['label']))
                f.write(" {} ".format(item['group_id']))
                f.write(" {} ".format(x1))
                f.write(" {} ".format(y1))
                f.write(" {} ".format(x2))
                f.write(" {} ".format(y2))
                f.write(" \n")

    def click_toolButton_filterData(self):
        self.imagesCountAll = 0
        self.imagesDeleteCount = 0
        self.imagesSurplusCount = 0
        for train_val in os.listdir(self.out_images_seq_path):
            train_val_dir = os.path.join(self.out_images_seq_path, train_val)
            # 如果是一个子目录就继续for循环，跳过下面的步骤，继续下一个循环
            if os.path.isdir(train_val_dir):
                for jsons in os.listdir(train_val_dir):
                    jsons_dir = os.path.join(train_val_dir, jsons)
                    # print(jsons_dir)
                    imageFiles_MoveTo_dir = self.delete_images_path + "/" + train_val + "/" + jsons
                    # print("jsonFiles_MoveTo_dir:", jsonFiles_MoveTo_dir)
                    # if not os.path.exists(jsonFiles_MoveTo_dir):  # 判断文件夹是否存在
                    #     os.makedirs(jsonFiles_MoveTo_dir)  # 新建文件夹
                    for images_files in glob.glob(jsons_dir + "/*.jpg"):
                        print("images_files---+=:",images_files)
                        self.imagesCountAll = self.imagesCountAll + 1
                        print(" self.imagesCountAll:",self.imagesCountAll)
                        json_files = images_files.split(".jpg")[0] + ".json"
                        print(" json_files:", json_files)
                        if not os.path.exists(json_files):
                            print("delete images------------------")
                            self.imagesDeleteCount = self.imagesDeleteCount+ 1
                            if not os.path.exists(imageFiles_MoveTo_dir):  # 判断文件夹是否存在
                                os.makedirs(imageFiles_MoveTo_dir)  # 新建文件夹
                            if self.imagesCountAll != self.imagesDeleteCount:
                                shutil.move(images_files, imageFiles_MoveTo_dir)  # 移动文件
        if self.imagesCountAll == self.imagesDeleteCount:
            self.ui.label_imageCountAll.setText(str(self.imagesCountAll))
            self.ui.label_imageDeleteCount.setText(str(0))
            self.ui.label_imageSurplusCount.setText(str(self.imagesCountAll))
        else:
            self.ui.label_imageCountAll.setText(str(self.imagesCountAll))
            self.ui.label_imageDeleteCount.setText(str(self.imagesDeleteCount))
            self.ui.label_imageSurplusCount.setText(str(self.imagesCountAll-self.imagesDeleteCount))

    def click_tooloButton_picodatafilter(self):
        self.PICOimagesCountAll = 0
        self.PICOimagesDeleteCount = 0
        self.PICOimagesSurplusCount = 0
        for images_files in glob.glob(self.pico_images_path + "/*.png"):
            print("images_files---+=:",images_files)
            self.PICOimagesCountAll = self.PICOimagesCountAll + 1
            # print(" self.imagesCountAll:",self.imagesCountAll)
            json_files = images_files.split("/images")[0] + "/annotations" + images_files.split("/images")[1].replace(".png",".json")
            print("json_files:", json_files)
            if not os.path.exists(json_files):
                print("delete images------------------")
                self.PICOimagesDeleteCount = self.PICOimagesDeleteCount+ 1
                if not os.path.exists(self.pico_delete_images):  # 判断文件夹是否存在
                    os.makedirs(self.pico_delete_images)  # 新建文件夹
                if self.PICOimagesCountAll != self.PICOimagesDeleteCount:
                    shutil.move(images_files, self.pico_delete_images)  # 移动文件
        if self.PICOimagesCountAll == self.PICOimagesDeleteCount:
            self.ui.label_PICOimageCountAll.setText(str(self.PICOimagesCountAll))
            self.ui.label_PICOimageDeleteCount.setText(str(0))
            self.ui.label_PICOimageSurplusCount.setText(str(self.PICOimagesCountAll))
        else:
            self.ui.label_PICOimageCountAll.setText(str(self.PICOimagesCountAll))
            self.ui.label_PICOimageDeleteCount.setText(str(self.PICOimagesDeleteCount))
            self.ui.label_PICOimageSurplusCount.setText(str(self.PICOimagesCountAll-self.PICOimagesDeleteCount))
    def click_toolButton_generateAnnotation(self):
        print("123")
        for train_val in os.listdir(self.out_images_seq_path):
            train_val_dir = os.path.join(self.out_images_seq_path, train_val)
            # 如果是一个子目录就继续for循环，跳过下面的步骤，继续下一个循环
            if os.path.isdir(train_val_dir):
                for jsons in os.listdir(train_val_dir):
                    jsons_dir = os.path.join(train_val_dir, jsons)
                    print(jsons_dir)
                    jsonFiles_MoveTo_dir = self.annotations_path + "/" + train_val + "/" + jsons
                    print("jsonFiles_MoveTo_dir:", jsonFiles_MoveTo_dir)
                    if not os.path.exists(jsonFiles_MoveTo_dir):  # 判断文件夹是否存在
                        os.makedirs(jsonFiles_MoveTo_dir)  # 新建文件夹
                    for jsons_files in glob.glob(jsons_dir + "/*.json"):
                        shutil.move(jsons_files, jsonFiles_MoveTo_dir)  # 移动文件
    def click_toolButton_jsonToTxt(self):
        for train_val in os.listdir(self.annotations_path):
            train_val_dir = os.path.join(self.annotations_path, train_val)
            # 如果是一个子目录就继续for循环，跳过下面的步骤，继续下一个循环
            if os.path.isdir(train_val_dir):
                for videos in os.listdir(train_val_dir):
                    videos_dir = os.path.join(train_val_dir, videos)
                    for video_images in os.listdir(videos_dir):
                        video_images_dir = os.path.join(videos_dir, video_images)
                        print("json_path:\t", video_images_dir)
                        with open(video_images_dir, "r", encoding='utf-8') as f:
                            json_d = json.load(f)
                            img_name = json_d['imagePath'].split("\\")[-1].split(".")[0] + ".jpg"

                            img_train_val_path = os.path.join(self.out_images_seq_path, train_val)
                            img_video_path = os.path.join(img_train_val_path, videos)
                            img_path = os.path.join(img_video_path, img_name)

                            txt_train_val_path = os.path.join(self.labels_with_ids_path, train_val)
                            txt_video_path = os.path.join(txt_train_val_path, videos)
                            # txt_path = os.path.join(txt_video_path, img_name)

                            print("img_path:\t", img_path)
                            self.relative_coordinate_txt(img_name, json_d, img_path, txt_video_path,video_images_dir)
                            # absolute_coordinate_txt(img_name, json_d, img_path)
    def click_toolButton_checkFromTxt(self):
        self.checkedJsonImgs = []
        self.checkImgIndex = 0
        if self.num_of_check != self.num_of_allcheck:
            # self.annotations_path = QFileDialog.getExistingDirectory(self, "choose annotations Directory", "./");
            for train_val in os.listdir(self.labels_with_ids_path):
                train_val_dir = os.path.join(self.labels_with_ids_path, train_val)
                # 如果是一个子目录就继续for循环，跳过下面的步骤，继续下一个循环
                if os.path.isdir(train_val_dir):
                    for videos in os.listdir(train_val_dir):
                        videos_dir = os.path.join(train_val_dir, videos)
                        self.all_images_to_check_inThisDir = len(glob.glob(videos_dir + "/" + '*.txt'))
                        self.have_check_inThisDir = 0
                        for video_images in os.listdir(videos_dir):
                            video_images_dir = os.path.join(videos_dir, video_images)
                            self.num_of_check = self.num_of_check + 1
                            print("txt_path:\t", video_images_dir)
                            img_name = video_images_dir.split("\\")[-1].split(".")[0] + ".jpg"
                            img_train_val_path = os.path.join(self.out_images_seq_path, train_val)
                            img_video_path = os.path.join(img_train_val_path, videos)
                            img_path = os.path.join(img_video_path, img_name)
                            print("img_path:\t", img_path)
                            self.have_check_inThisDir = self.have_check_inThisDir + 1
                            self.ui.statusbar.addWidget(self.pbar)
                            self.pbar.setValue(100 * (self.have_check_inThisDir) / self.all_images_to_check_inThisDir)
                            self.show_label_from_txt(img_path, video_images_dir)
            else:
                self.num_of_check = 0

    def click_toolButton_checkFromJson(self):
        self.checkedJsonImgs = []
        self.checkImgIndex = 0
        if self.num_of_check != self.num_of_allcheck:
            # self.annotations_path = QFileDialog.getExistingDirectory(self, "choose annotations Directory", "./");
            for train_val in os.listdir(self.annotations_path):
                train_val_dir = os.path.join(self.annotations_path, train_val)
                # 如果是一个子目录就继续for循环，跳过下面的步骤，继续下一个循环
                if os.path.isdir(train_val_dir):
                    for videos in os.listdir(train_val_dir):
                        videos_dir = os.path.join(train_val_dir, videos)
                        self.all_images_to_check_inThisDir = len(glob.glob(videos_dir + "/" + '*.json'))
                        self.have_check_inThisDir = 0
                        for video_images in os.listdir(videos_dir):
                            video_images_dir = os.path.join(videos_dir, video_images)
                            print("json_path:\t", video_images_dir)
                            self.num_of_check = self.num_of_check + 1
                            with open(video_images_dir, "r", encoding='utf-8') as f:
                                json_d = json.load(f)
                                img_name = json_d['imagePath'].split("\\")[-1].split(".")[0] + ".jpg"
                                img_train_val_path = os.path.join(self.out_images_seq_path, train_val)
                                img_video_path = os.path.join(img_train_val_path, videos)
                                img_path = os.path.join(img_video_path, img_name)
                                print("img_path:\t", img_path)
                                self.have_check_inThisDir = self.have_check_inThisDir + 1
                                self.ui.statusbar.addWidget(self.pbar)
                                self.pbar.setValue(100 * (self.have_check_inThisDir ) / self.all_images_to_check_inThisDir)
                                self.show_label_from_json(img_path, json_d)
            else:
                self.num_of_check = 0
    def generateTwoGroupTestOrder(self):
        self.sitGroupTestOrder.clear()
        self.standGroupTestOrder.clear()
        for i in range(2):
            self.testObjectPart = ['Bottled_top', 'Bottled_side', 'Bagged_top', 'Bagged_side', 'Boxed_top',
                                   'Boxed_side', 'Canned_top', 'Canned_side', 'Mug_top', 'Mug_handle', 'Mug_side',
                                   'Vials_top', 'Vials_side']
            for j in range(13):
                if i == 0:
                    testSample = random.sample(self.testObjectPart, 1)
                    self.testObjectPart.remove(testSample[0])
                    self.standGroupTestOrder.append(testSample[0])
                if i == 1:
                    testSample = random.sample(self.testObjectPart, 1)
                    self.testObjectPart.remove(testSample[0])
                    self.sitGroupTestOrder.append(testSample[0])
        print("self.sitGroupTestOrder",self.sitGroupTestOrder)
        print("self.standGroupTestOrder",self.standGroupTestOrder)

    def setObjectStatus(self,object,object_part):
        self.ui.label_Object.setText(object)
        self.ui.label_Object.setStyleSheet("background-color: rgb(0, 255, 255);")
        if object_part == "top":
            self.ui.label_ObjectTop.setStyleSheet("background-color: rgb(0, 255, 0);")
            self.ui.label_objectSideR.setStyleSheet("background-color: rgb(222, 222, 222);")
            self.ui.label_objectSideL.setStyleSheet("background-color: rgb(222, 222, 222);")
        if object_part == "side":
            self.ui.label_ObjectTop.setStyleSheet("background-color: rgb(222, 222, 222);")
            self.ui.label_objectSideR.setStyleSheet("background-color: rgb(0, 255, 0);")
            self.ui.label_objectSideL.setStyleSheet("background-color: rgb(222, 222, 222);")
        if object_part == "handle":
            self.ui.label_ObjectTop.setStyleSheet("background-color: rgb(222, 222, 222);")
            self.ui.label_objectSideR.setStyleSheet("background-color: rgb(222, 222, 222);")
            self.ui.label_objectSideL.setStyleSheet("background-color: rgb(0, 255, 0);")
    def textChanged_comboBox_testPeople(self,text):
        print("text:", text)
        self.testPeople = text
        self.testGroupCurrent = 0
        self.testNumCurrent = 0
        self.ui.spinBox_testNumCurrent.setValue(self.testNumCurrent + 1)
        self.ui.spinBox_testGroupCurrent.setValue(self.testGroupCurrent + 1)
        self.generateTwoGroupTestOrder()
        self.currentGroup = self.standGroupTestOrder
        self.testPose = "stand"
        self.ui.label_sitOrStand.setText(self.testPose)

        currnet_Object = self.currentGroup[self.testNumCurrent].split("_")[0]
        currnet_Object_part = self.currentGroup[self.testNumCurrent].split("_")[1]
        self.testInfo = self.testPeople + "_" + self.testPose + "_" + currnet_Object + "_" + currnet_Object_part
        self.setObjectStatus(currnet_Object,currnet_Object_part)
        self.ui.lineEdit_testObject.setText(str(currnet_Object))
        self.ui.lineEdit_testObjectPart.setText(str(currnet_Object_part))
    def click_toolButton_startTest(self):
        if self.ui.toolButton_startTest.text() == "开始实验":
            icon19 = QtGui.QIcon()
            icon19.addPixmap(QtGui.QPixmap(":/toolButton/static/stopTest.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.ui.toolButton_startTest.setIcon(icon19)
            self.ui.toolButton_startTest.setText("结束实验")

            currentTime = QDateTime.currentDateTime()  # 获取现在的时间
            self.startTestTime_ms = currentTime.toMSecsSinceEpoch()
            self.stopTestTime_ms = self.startTestTime_ms
            print("ms", self.startGatherTome_ms)

            self.usbCameraThread = threadCameraUSB(self.record_camera, self.usbQueueImage,self.testInfo,self.startTestTime_ms)
            self.usbCameraThread.start()
            print("ms")

            self.realSenseThread = threadRealSenseSensor(self.realSenseColorImages,self.realSenseDepthImages,self.testInfo)
            self.realSenseThread.start()
            print("ms")


            self.timer_testRecord.start(30)
        else:
            icon19 = QtGui.QIcon()
            icon19.addPixmap(QtGui.QPixmap(":/toolButton/static/startTest.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.ui.toolButton_startTest.setIcon(icon19)
            self.ui.toolButton_startTest.setText("开始实验")
            self.usbCameraThread.runUsbCamera = False
            self.realSenseThread.runUsbCamera = False
            self.timer_testRecord.stop()
    def timer_testRecord_timeout(self):
        # 可以考虑在realSense采集的while循环里面更新时间
        self.ui.lineEdit_tesTime.setText(str(self.usbCameraThread.testTime))
        if self.realSenseColorImages.qsize() > 0 and self.realSenseDepthImages.qsize() > 0:
            # color_and_depth_images = self.realSenseColorImages.get()
            print("9999")
            color_and_depth_images = np.hstack((self.realSenseColorImages.get(), self.realSenseDepthImages.get()))
            print("9999")
            showImage = QtGui.QImage(color_and_depth_images.data, color_and_depth_images.shape[1],
                                     color_and_depth_images.shape[0], QtGui.QImage.Format_RGB888)
            self.ui.label_imgShow.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.ui.label_imgShow.setScaledContents(True)
    def click_toolButton_nextTest(self):
        if self.testNumCurrent == 12 :
            if self.testGroupCurrent == 0:
                self.testNumCurrent = 0
                self.testGroupCurrent = 1
                # print(self.standGroupTestOrder)
                # self.currentGroup.clear()
                # print(self.standGroupTestOrder)
                self.currentGroup = self.sitGroupTestOrder
                self.testPose = "sit"
                self.ui.label_sitOrStand.setText(self.testPose)
                self.ui.spinBox_testNumCurrent.setValue(self.testNumCurrent+1)
                self.ui.spinBox_testGroupCurrent.setValue(self.testGroupCurrent+1)
        else:
            self.testNumCurrent = self.testNumCurrent + 1
            self.ui.spinBox_testNumCurrent.setValue(self.testNumCurrent + 1)
            self.ui.spinBox_testGroupCurrent.setValue(self.testGroupCurrent + 1)

        currnet_Object = self.currentGroup[self.testNumCurrent].split("_")[0]
        currnet_Object_part = self.currentGroup[self.testNumCurrent].split("_")[1]
        self.testInfo = self.testPeople + "_" + self.testPose + "_" + currnet_Object + "_" + currnet_Object_part
        self.setObjectStatus(currnet_Object, currnet_Object_part)
        self.ui.lineEdit_testObject.setText(str(currnet_Object))
        self.ui.lineEdit_testObjectPart.setText(str(currnet_Object_part))

    def click_toolButton_lastTest(self):
        if self.testNumCurrent == 0:
            if self.testGroupCurrent == 1:
                self.testNumCurrent = 12
                self.testGroupCurrent = 0
                # print(self.standGroupTestOrder)
                # self.currentGroup.clear()
                # print(self.standGroupTestOrder)
                self.currentGroup = self.standGroupTestOrder
                self.testPose = "stand"
                self.ui.label_sitOrStand.setText(self.testPose)
                self.ui.spinBox_testNumCurrent.setValue(self.testNumCurrent + 1)
                self.ui.spinBox_testGroupCurrent.setValue(self.testGroupCurrent + 1)
        else:
            self.testNumCurrent = self.testNumCurrent - 1
            self.ui.spinBox_testNumCurrent.setValue(self.testNumCurrent + 1)
            self.ui.spinBox_testGroupCurrent.setValue(self.testGroupCurrent + 1)

        currnet_Object = self.currentGroup[self.testNumCurrent].split("_")[0]
        currnet_Object_part = self.currentGroup[self.testNumCurrent].split("_")[1]
        self.testInfo = self.testPeople + "_" + self.testPose + "_" + currnet_Object + "_" + currnet_Object_part
        self.setObjectStatus(currnet_Object, currnet_Object_part)
        self.ui.lineEdit_testObject.setText(str(currnet_Object))
        self.ui.lineEdit_testObjectPart.setText(str(currnet_Object_part))

    def click_toolButton_scanCamera_test(self):
        pygame.camera.init()
        self.camera_id_lis = pygame.camera.list_cameras()
        print(self.camera_id_lis)
        self.ui.comboBox_cameraDeviceList.addItems(self.camera_id_lis)
        self.ui.comboBox_cameraDeviceList_wrist.addItems(self.camera_id_lis)
        self.ui.comboBox_cameraDeviceList_record.addItems(self.camera_id_lis)

    def comboBox_activated_cameraDeviceList_wrist(self,text):
        if text == "Intel(R) RealSense(TM) Depth Camera 435i Depth" or text == "Intel(R) RealSense(TM) Depth Camera 435i RGB" :
            self.wrist_camera = 10

    def comboBox_activated_cameraDeviceList_record(self,text):
        if text == "Integrated Camera":
            self.record_camera  = self.camera_id_lis.index("Integrated Camera")
            print("self.record_camera :",self.record_camera)
        elif text == "LRCP USB2.0" :
            self.record_camera = self.camera_id_lis.index("LRCP USB2.0")
            print("self.record_camera :", self.record_camera)

class threadCameraUSB(threading.Thread):
    """usb camera
    @access   Usb descriptor.
    @queue    Output queue.
    """
    def __init__(self, access, queue,testInfo,startTestTime):
        threading.Thread.__init__(self)
        self.access = access
        self.q = queue
        self.runUsbCamera = True
        self.image = None
        self.out = None
        self.cap = None
        self.testInfo = testInfo
        self.startTestTime_ms = startTestTime
        self.stopTestTime_ms = startTestTime
        self.testTime = 0.0
    def setVideoOut(self):
        # 保存视频
        code = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 编码格式
        save_fps = 30  # 保存视频的帧率
        time = QDateTime.currentDateTime()  # 获取现在的时间
        timeplay = time.toString('yyyy_MM_dd_hh_mm_ss')  # 设置显示时间的格式
        self.saveVideo_filepath = "./onlineTestData/testRecordVideos/" + self.testInfo + "_" + "video_" + timeplay + ".mp4"
        # print(self.saveVideo_filepath)
        self.frame_size = (int(self.cap.get(3)), int(self.cap.get(4)))  # 获取摄像头分辨率
        self.out = cv2.VideoWriter(self.saveVideo_filepath, code, save_fps, self.frame_size, isColor=True)  # 保存视频的视频流
    def save_video(self):
        if self.out.isOpened():  # 判断视频流是否创建成功
            print('out is  opened')
            self.out.write(self.image)
            print('out is  writing...')

    def run(self):
        self.cap = cv2.VideoCapture(self.access)
        if self.cap.isOpened():
            print('camera usb ' + str(self.access) + " connected.")
            self.setVideoOut()

        while self.runUsbCamera:
            ret, self.image = self.cap.read()
            # 设置记录时间
            currentTime = QDateTime.currentDateTime()  # 获取现在的时间
            self.stopTestTime_ms = currentTime.toMSecsSinceEpoch()
            self.testTime = (self.stopTestTime_ms - self.startTestTime_ms) / 1000
            if ret:
                print("usbCamera")
                self.q.put(self.image)
                self.save_video()
                self.q.get() if self.q.qsize() > 2 else time.sleep(0.01)
        self.out.release()
class threadRealSenseSensor(threading.Thread):
    """usb camera
    @access   Usb descriptor.
    @queue    Output queue.
    """
    def __init__(self, colorQueue, depthQueue, testInfo):
        threading.Thread.__init__(self)
        print('opened1')
        self.realSenseSensor = getDataFromRealSense()
        self.realSenseSensor.config_sensor()
        self.frame_size = (int(self.realSenseSensor.color_height), int(self.realSenseSensor.color_width))  # 获取摄像头分辨率
        self.FPS = self.realSenseSensor.color_frame_rate  # 获取摄像头帧率
        print('opened')

        # self.access = access
        self.color_q = colorQueue
        self.depth_q = depthQueue
        self.runUsbCamera = True
        self.testInfo = testInfo
        self.image = None
        self.out = None
        self.depth_out = None
        # self.testInfo = testInfo
        # self.startTestTime_ms = startTestTime
        # self.stopTestTime_ms = startTestTime
        # self.testTime = 0.0
    def setVideoOut(self):
        # 保存视频
        code = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 编码格式
        save_fps = 30  # 保存视频的帧率
        time = QDateTime.currentDateTime()  # 获取现在的时间
        timeplay = time.toString('yyyy_MM_dd_hh_mm_ss')  # 设置显示时间的格式
        self.saveVideo_filepath = "./onlineTestData/testWristCamVideos/color/" + self.testInfo + "_" + "color_video_" + timeplay + ".mp4"
        # print(self.saveVideo_filepath)

        self.saveDepthVideo_filepath = "./onlineTestData/testWristCamVideos/depth/" + self.testInfo + "_" + "depth_video_" + timeplay + ".mp4"
        # print(self.saveVideo_filepath)
        print(self.saveDepthVideo_filepath)
        self.out = cv2.VideoWriter(self.saveVideo_filepath, code, save_fps, self.frame_size, isColor=True)  # 保存视频的视频流
        self.depth_out = cv2.VideoWriter(self.saveDepthVideo_filepath, code, save_fps, self.frame_size,
                                         isColor=False)  # 保存深度视频的视频流
    def save_video(self):
        if self.out.isOpened():  # 判断视频流是否创建成功
            print('out is  opened')
            self.out.write(self.image)
            print('out is  writing...')
        if self.depth_out.isOpened():  # 判断视频流是否创建成功
            print("shendu视频流创建成功")
            self.depth_out.write(self.depth_img_nomap)

    def run(self):
        self.setVideoOut()
        while self.runUsbCamera:
            self.image, self.depth_img_nomap, self.depth_img_colormap, self.gyro, self.sensor_angles = self.realSenseSensor.getFrames()
            show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.color_q.put(show)
            self.depth_q.put(self.depth_img_colormap)
            print('real sense opened')
            self.save_video()
            self.color_q.get() if self.color_q.qsize() > 2 else time.sleep(0.01)

        self.out.release()
        self.depth_out.release()

# 程序入口
if __name__ == '__main__':

    e = mainWindow()
    sys.exit(e.app.exec())