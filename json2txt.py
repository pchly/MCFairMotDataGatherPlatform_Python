import json
import os
import cv2

img_folder_path = r'D:\Paddle\visdroneDataSetSelfLabel\images'  # 图片存放文件夹
folder_path = r"D:\Paddle\visdroneDataSetSelfLabel\annotations"  # 标注数据的文件地址
txt_folder_path = r"D:\Paddle\visdroneDataSetSelfLabel\labels_with_ids"  # 转换后的txt标签文件存放的文件夹
label_list=['cat','dog']
# filename='D:\Paddle\data_annotated\label_list.txt'
# with open(filename) as f:
#     for line in f:
#         print(line)
#         label_list.push(line)

# 保存为相对坐标形式 :label id x_center y_center w h
def relative_coordinate_txt(img_name, json_d, img_path,label_with_id_path):
    src_img = cv2.imread(img_path)
    h, w = src_img.shape[:2]
    txt_name = img_name.split(".")[0] + ".txt"
    txt_path = os.path.join(label_with_id_path, txt_name)
    print("txt_path: ",txt_path)

    if not os.path.exists(label_with_id_path):  # 判断文件夹是否存在
        os.makedirs(label_with_id_path)  # 新建文件夹
    else:
        print('文件夹已存在....')

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
            label_index = label_list.index(item['label'])
            f.write("{} ".format(label_index))
            f.write(" {} ".format(item['group_id']))
            f.write(" {:.6f} ".format(x_center / w))
            f.write(" {:.6f} ".format(y_center / h))
            f.write(" {:.6f} ".format(width / w))
            f.write(" {:.6f} ".format(hight / h))
            f.write(" \n")


# 保存为绝对坐标形式 :label id x1 y1 x2 y2
def absolute_coordinate_txt(img_name, json_d, img_path):
    src_img = cv2.imread(img_path)
    h, w = src_img.shape[:2]
    txt_name = img_name.split(".")[0] + ".txt"
    txt_path = os.path.join(txt_folder_path, txt_name)
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


#
# for jsonfile in os.listdir(folder_path):
#     temp_path = os.path.join(folder_path, jsonfile)
#
#     # 如果是一个子目录就继续
#     if os.path.isdir(temp_path):
#         continue
#     print("json_path:\t", temp_path)
#     jsonfile_path = temp_path
#     with open(jsonfile_path, "r", encoding='utf-8') as f:
#         json_d = json.load(f)
#         img_name = json_d['imagePath'].split("\\")[-1].split(".")[0] + ".jpg"
#         img_path = os.path.join(img_folder_path, img_name)
#         print("img_path:\t", img_path)
#         relative_coordinate_txt(img_name, json_d, img_path)
#         #absolute_coordinate_txt(img_name, json_d, img_path)


for train_val in os.listdir(folder_path):
    train_val_dir = os.path.join(folder_path, train_val)
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


                    txt_train_val_path = os.path.join(txt_folder_path, train_val)
                    txt_video_path = os.path.join(txt_train_val_path, videos)
                    # txt_path = os.path.join(txt_video_path, img_name)

                    print("img_path:\t", img_path)
                    relative_coordinate_txt(img_name, json_d, img_path,txt_video_path)
                    # absolute_coordinate_txt(img_name, json_d, img_path)
            # temp_path
        # continue