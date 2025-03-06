# camera-ready
import os
import sys
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import re

sys.path.append("py/walk/deeplabv3-master")
from datasets import DatasetSeq # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

sys.path.append("py/walk/deeplabv3-master/model")
from model.deeplabv3 import DeepLabV3

sys.path.append("py/walk/deeplabv3-master/utils")
from utils.utils import label_img_to_color

# 检查点文件
checkpoint_file = 'walk/deeplabv3-master/training_logs/model_eval_seq/checkpoints/checkpoint1.txt'

def load_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return int(f.read().strip())
    return 0

def save_checkpoint(index):
    with open(checkpoint_file, 'w') as f:
        f.write(str(index))

def clean_filename(filename):
    # 使用正则表达式去除最后一个 .png
    return re.sub(r'\.png$', '', filename)
if __name__ == "__main__":
    # 打印检查点文件的位置
    print(f"Checkpoint file location: {os.path.abspath(checkpoint_file)}")

    # 检查检查点文件是否存在
    if os.path.exists(checkpoint_file):
        print("Checkpoint file exists.")
    else:
        print("Checkpoint file does not exist.")

    batch_size = 2
    new_img_h = 512
    new_img_w = 1024

    network = DeepLabV3("eval_seq", project_dir="D:/py/walk/deeplabv3-master").cuda()
    network.load_state_dict(torch.load("walk/deeplabv3-master/pretrained_models/model_13_2_2_2_epoch_580.pth"))
    # network.load_state_dict(torch.load("walk/deeplabv3-master/training_logs/model_2/checkpoints/model_2_epoch_2.pth"))
    # network.load_state_dict(torch.load("walk/deeplabv3-master/training_logs/model_1/checkpoints/model_1_epoch_256.pth"))
    
    # 加载检查点
    current_image_index = load_checkpoint()
    image_index = 0
    n = 0
    
    # for sequence in ["00", "01", "02"]:
    for sequence in ["01"]:  # 只处理序列 "01"
        print(f"Processing sequence: {sequence}")
        

        val_dataset = DatasetSeq(cityscapes_data_path="walk",
                                cityscapes_meta_path="walk/meta",
                                sequence=sequence
                                )#要在指定位置放上要测试的图片
        # print(val_dataset)
        num_val_batches = int(len(val_dataset) / batch_size)
        # print ("num_val_batches:", num_val_batches)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size, shuffle=False,
                                                 num_workers=1)

        network.eval()  # (set in evaluation mode, this affects BatchNorm and dropout)设置为评估模式
        unsorted_img_ids = []
        
        n += len(val_dataset)
        print(len(val_dataset))
        print(n)
        
        for step, (imgs, img_ids, padding_infos) in enumerate(val_loader):
            # 如果当前索引小于等于检查点索引，则跳过
            if current_image_index > image_index:
                if current_image_index - image_index != 1:
                    image_index += batch_size          
                    if image_index - n == 1 :# 不然有可能计数会多一张
                        image_index -= 1
                        current_image_index -= 1
                    continue

            with torch.no_grad():  # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)不需要计算梯度
                imgs = Variable(imgs).cuda()  # (shape: (batch_size, 3, img_h, img_w))

                outputs = network(imgs)  # (shape: (batch_size, num_classes, img_h, img_w))

                ####################################################################
                # save data for visualization:
                ####################################################################
                outputs = outputs.data.cpu().numpy()  # (shape: (batch_size, num_classes, img_h, img_w))
                pred_label_imgs = np.argmax(outputs, axis=1)  # (shape: (batch_size, img_h, img_w))
                pred_label_imgs = pred_label_imgs.astype(np.uint8)

                for i in range(pred_label_imgs.shape[0]):
                    if current_image_index > image_index:
                        image_index += 1
                        continue
                    
                    pred_label_img = pred_label_imgs[i]  # (shape: (img_h, img_w))
                    img_id = img_ids[i]
                    img = imgs[i]  # (shape: (3, img_h, img_w))

                    img = img.data.cpu().numpy()
                    img = np.transpose(img, (1, 2, 0))  # (shape: (img_h, img_w, 3))
                    img = img * np.array([0.229, 0.224, 0.225])
                    img = img + np.array([0.485, 0.456, 0.406])
                    img = img * 255.0
                    img = img.astype(np.uint8)

                    # 获取当前样本的填充信息
                    pad_left = padding_infos["pad_left"][i].item()
                    pad_right = padding_infos["pad_right"][i].item()
                    pad_top = padding_infos["pad_top"][i].item()
                    pad_bottom = padding_infos["pad_bottom"][i].item()
                    original_width = padding_infos["original_width"][i].item()
                    original_height = padding_infos["original_height"][i].item()

                    # 对预测结果进行裁剪
                    pred_label_img_cropped = pred_label_img[pad_top:new_img_h-pad_bottom, pad_left:new_img_w-pad_right]

                    # 将裁剪后的预测结果转换为颜色图
                    pred_label_img_color = label_img_to_color(pred_label_img_cropped)

                    # 裁剪原始图像和叠加图像
                    img_cropped = img[pad_top:new_img_h-pad_bottom, pad_left:new_img_w-pad_right]
                    # overlayed_img_cropped = overlayed_img[pad_top:new_img_h-pad_bottom, pad_left:new_img_w-pad_right]
                    overlayed_img = 0.35 * img_cropped + 0.65 * pred_label_img_color
                    overlayed_img = overlayed_img.astype(np.uint8)

                    # 保存裁剪后的图像和预测结果
                    img_id_cleaned = clean_filename(img_id)
                    cv2.imwrite(network.model_dir + "/" + img_id_cleaned + ".png", img_cropped)
                    cv2.imwrite(network.model_dir + "/" + img_id_cleaned + "_pred.png", pred_label_img_color)
                    cv2.imwrite(network.model_dir + "/" + img_id_cleaned + "_overlayed.png", overlayed_img)

                    unsorted_img_ids.append(img_id)
                    image_index += 1
                    current_image_index += 1
                    save_checkpoint(current_image_index)  # 更新检查点

        ############################################################################
        # video没什么用，不要了
        # create visualization video:
        ############################################################################
        # out = cv2.VideoWriter("%s/stuttgart_%s_combined.avi" % (network.model_dir, sequence), cv2.VideoWriter_fourcc(*"MJPG"), 20, (2 * original_width, 2 * original_height))
        # sorted_img_ids = sorted(unsorted_img_ids)
        # for img_id in sorted_img_ids:
        #     img = cv2.imread(network.model_dir + "/" + img_id + ".png", -1)
        #     pred_img = cv2.imread(network.model_dir + "/" + img_id + "_pred.png", -1)
        #     overlayed_img = cv2.imread(network.model_dir + "/" + img_id + "_overlayed.png", -1)

        #     # combined_img = np.zeros((2*img_h, 2*img_w, 3), dtype=np.uint8)

        #     # combined_img[0:img_h, 0:img_w] = img
        #     # combined_img[0:img_h, img_w:(2*img_w)] = pred_img
        #     # combined_img[img_h:(2*img_h), (int(img_w/2)):(img_w + int(img_w/2))] = overlayed_img


        #     combined_img = np.zeros((2 * original_height, 2 * original_width, 3), dtype=np.uint8)

        #     combined_img[0:original_height, 0:original_width] = img
        #     combined_img[0:original_height, original_width:(2 * original_width)] = pred_img
        #     combined_img[original_height:(2 * original_height), (int(original_width / 2)):(original_width + int(original_width / 2))] = overlayed_img

        #     out.write(combined_img)

        # out.release()