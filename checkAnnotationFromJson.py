import json
import os
import cv2

img_folder_path = r'D:\Paddle\visdroneDataSetSelfLabel\images'  # 原图片存放地址
annotation_folder_path = r"D:\Paddle\visdroneDataSetSelfLabel\annotations"  # 标注数据的文件地址

class checkAnnotationFromJson(object):
    def __init__(self, img_folder_path, annotation_folder_path):
        self.img_folder_path = img_folder_path
        self.annotation_folder_path = annotation_folder_path
    def check(self):
        for train_val in os.listdir(self.annotation_folder_path):
            train_val_dir = os.path.join(self.annotation_folder_path, train_val)
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
                            img_train_val_path = os.path.join(img_folder_path, train_val)
                            img_video_path = os.path.join(img_train_val_path, videos)
                            img_path = os.path.join(img_video_path, img_name)
                            print("img_path:\t", img_path)
                            self.show_label_from_json(img_path, json_d)
                    # temp_path
                # continue
    # cv2.putText(img, str, (123,456)), font, 2, (0,255,0), 3)
    # 各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
    def show_label_from_json(img_path, json_d):
        window_name = ('src')
        cv2.namedWindow(window_name, cv2.WINDOW_FREERATIO)
        src_img = cv2.imread(img_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for item in json_d["shapes"]:
            # print(item['points'])
            point = item['points']
            id = item['group_id']
            label = item['label']
            putTextStr= "%s:%s"%(label,id)
            p1 = (int(point[0][0]), int(point[0][1]))
            p2 = (int(point[1][0]), int(point[1][1]))
            cv2.rectangle(src_img, p1, p2, (0, 255, 0), 1)
            cv2.putText(src_img, putTextStr, p1, font, 0.5, (0, 255, 255), 1)

        cv2.imshow(window_name, src_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return



# for jsonfile in os.listdir(folder_path):
#     temp_path = os.path.join(folder_path, jsonfile)
#
#     # 如果是一个子目录就继续for循环，跳过下面的步骤，继续下一个循环
#     if os.path.isdir(temp_path):
#
#         continue
#     print("json_path:\t", temp_path)
#     temp_path
#     with open(temp_path, "r", encoding='utf-8') as f:
#         json_d = json.load(f)
#         img_name = json_d['imagePath'].split("\\")[-1].split(".")[0] + ".jpg"
#         img_path = os.path.join(img_folder_path, img_name)
#         print("img_path:\t", img_path)
#         show_label_from_json(img_path, json_d)




