import json
import os
import cv2

img_folder_path = r'D:\Paddle\visdroneDataSetSelfLabel\images'  # 图片存放文件夹
folder_path = r"D:\Paddle\visdroneDataSetSelfLabel\annotations"  # 标注数据的文件地址
txt_folder_path = r"D:\Paddle\visdroneDataSetSelfLabel\labels_with_ids"  # 转换后的txt标签文件存放的文件夹

label_list=['cat','dog']
# 相对坐标格式
def show_label_from_txt(img_path, txt_path):
    window_name = ('src')
    cv2.namedWindow(window_name, cv2.WINDOW_FREERATIO)
    src_img = cv2.imread(img_path)
    h, w = src_img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    with open(txt_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        data = line.split(' ')
        label = label_list[int(data[0])]
        id =data[2]
        putTextStr = "%s:%s" % (label, id)
        x1 = int((float(data[4]) - float(data[8]) / 2) * w)
        y1 = int((float(data[6]) - float(data[10]) / 2) * h)
        x2 = int((float(data[4]) + float(data[8]) / 2) * w)
        y2 = int((float(data[6]) + float(data[10]) / 2) * h)
        p1 = (x1, y1)
        p2 = (x2, y2)
        cv2.rectangle(src_img, p1, p2, (0, 250, 0), 1)
        cv2.putText(src_img, putTextStr, p1, font, 0.5, (0, 255, 255), 1)

    cv2.imshow(window_name, src_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


#
# for txtfile in os.listdir(txt_folder_path):
#     temp_path = os.path.join(txt_folder_path, txtfile)
#
#     # 如果是一个子目录就继续
#     if os.path.isdir(temp_path):
#
#         continue
#     print("txt_path:\t", temp_path)
#     img_name = txtfile.split("\\")[-1].split(".")[0] + ".jpg"
#     img_path = os.path.join(img_folder_path, img_name)
#     show_label_from_txt(img_path, temp_path)


for train_val in os.listdir(txt_folder_path):
    train_val_dir = os.path.join(txt_folder_path, train_val)
    # 如果是一个子目录就继续for循环，跳过下面的步骤，继续下一个循环
    if os.path.isdir(train_val_dir):
        for videos in os.listdir(train_val_dir):
            videos_dir = os.path.join(train_val_dir, videos)
            for video_images in os.listdir(videos_dir):
                video_images_dir = os.path.join(videos_dir, video_images)

                print("txt_path:\t", video_images_dir)
                img_name = video_images_dir.split("\\")[-1].split(".")[0] + ".jpg"


                img_train_val_path = os.path.join(img_folder_path, train_val)
                img_video_path = os.path.join(img_train_val_path, videos)
                img_path = os.path.join(img_video_path, img_name)
                show_label_from_txt(img_path, video_images_dir)
