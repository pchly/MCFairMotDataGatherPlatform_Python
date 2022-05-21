"""Script read video from local ,and clip pic rename save in local"""
import os
import cv2
import glob
# source_video_path = r'D:\Paddle\visdroneDataSetSelfLabel\videos\\'  # 图片存放文件夹
# out_inages_seq_path = r"D:\Paddle\visdroneDataSetSelfLabel\imgs\\"
class clipvideo(object):
    def __init__(self, video_path, save_path, num):
        """
        :param video_path: 视频文件路径
        :param save_path: 图像存储路径
        :param num:  每隔多少张图像被存储
        """
        self.video_path = video_path
        self.save_path = save_path
        self.num = num
        self.haveClipNum = 0
        self.allVideosNum = 0
        self.haveClipRate = 0

    def run(self):
        dataSetPart = ["train/","val/"]
        self.allVideosNum = len(glob.glob(self.video_path + "train/" + '*.mp4')) + len(glob.glob(self.video_path + "val/" + '*.mp4'))
        print("train videos num :", self.allVideosNum)
        # 依次读取文件名
        for videos_part in dataSetPart:
            videos_files = os.listdir(self.video_path + videos_part)
            len_of_videos = len(videos_files)
            for video_name in videos_files:
                print("video_name:",video_name)
                # 对文件名进行拆分处理
                file_name = video_name.split('.')[0]
                folder_name = self.save_path + "/" + videos_part + file_name  # 构成新目录存放视频帧
                os.makedirs(folder_name, exist_ok=True)  # 查看是否存在目录，否则创建
                print("video_path:", self.video_path + videos_part + video_name)
                vc = cv2.VideoCapture(self.video_path + videos_part + video_name)
                # 获取视频帧率
                fps = vc.get(cv2.CAP_PROP_FPS)
                print("fps:", fps)
                # 判断视频是否可以打开
                rval = vc.isOpened()
                c = 1
                while rval:
                    raval, frame = vc.read()
                    frame_name = folder_name + "/" + file_name
                    print("frame_name",frame_name)
                    if raval:
                        if (c % self.num == 0):
                            if c < 10:
                                frame_num_toWrite = "00" + str(c)
                            elif 10 <= c  and c <= 99:
                                frame_num_toWrite = "0" + str(c)
                            elif  c > 100:
                                frame_num_toWrite = str(c)
                            cv2.imwrite(frame_name + "_" + frame_num_toWrite + ".jpg", frame)
                        cv2.waitKeyEx(1)
                    else:
                        break
                    c = c + 1
                vc.release()
                self.haveClipNum = self.haveClipNum + 1
                self.haveClipRate = 100 * (self.haveClipNum ) / self.allVideosNum
                print("One video have clip finshed! file is %s " % (self.video_path + video_name))
        self.haveClipNum = 0
        self.allVideosNum = 0


# if __name__ == '__main__':
#     clip = clipvideo(source_video_path,out_inages_seq_path,10)



