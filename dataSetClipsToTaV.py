"""Script read video from local ,and clip pic rename save in local"""
import os
import shutil
import random
from glob import glob


class dataSetClipsTaV(object):
    def __init__(self, video_path, rate):
        """
        :param video_path: 视频文件路径
        :param rate:  训练集的比例
        """
        self.video_path=video_path
        self.rate=rate


    def mymovefile(self,srcfile,dstpath):                       # 移动函数
        #srcfile 需要复制、移动的文件
        # dstpath 目的地址
        if not os.path.isfile(srcfile):
            print ("%s not exist!"%(srcfile))
        else:
            fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
            if not os.path.exists(dstpath):
                os.makedirs(dstpath)                       # 创建路径
            shutil.move(srcfile, dstpath + fname)          # 移动文件
            print ("move %s -> %s"%(srcfile, dstpath + fname))

    def mycopyfile(self,srcfile,dstpath):                       # 移动函数
        #srcfile 需要复制、移动的文件
        # dstpath 目的地址
        if not os.path.isfile(srcfile):
            print ("%s not exist!"%(srcfile))
        else:
            fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
            if not os.path.exists(dstpath):
                os.makedirs(dstpath)                       # 创建路径
            shutil.copy(srcfile, dstpath + fname)          # 移动文件
            print ("copy %s -> %s"%(srcfile, dstpath + fname))

    def run(self):
        print("self.video_path:",self.video_path)
        all_videos = os.listdir(self.video_path)  # 返回指定文件夹下的视频文件
        # video = glob(self.video_path + '*')
        print(all_videos)
        num_of_train= int(len(all_videos)*self.rate/100)
        print(num_of_train)
        train_videos = random.sample(all_videos,num_of_train)
        print(train_videos)
        for video_file in train_videos:
            # self.mymovefile(self.video_path + video_file,self.video_path + "/train/")
            self.mycopyfile(self.video_path + video_file, self.video_path + "/train/")
        val_videos = os.listdir(self.video_path)
        val_videos = [item for item in all_videos if item not in train_videos]
        for video_file in val_videos:
            # self.mymovefile(self.video_path + video_file,self.video_path + "/val/")
            self.mycopyfile(self.video_path + video_file, self.video_path + "/val/")







