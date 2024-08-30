import os
import cv2
import numpy as np

# 设置Cityscapes数据集路径
CITYSCAPES_PATH = 'walk/gtFine'

# 加载标签数据并统计灰度值
def load_and_count_grayscale_labels(image_set='try_train'):
    label_path = os.path.join(CITYSCAPES_PATH, image_set)
    label_files = os.listdir(label_path)
    
    grayscale_values = set()
    
    for label_file in label_files:
        if label_file.lower().endswith('.png'):
            label = cv2.imread(os.path.join(label_path, label_file), cv2.IMREAD_GRAYSCALE)
            if label is not None:
                unique_values = np.unique(label)
                grayscale_values.update(unique_values)
    
    num_classes = len(grayscale_values)
    
    return num_classes

# 统计训练集中的类别数量
num_classes_train = load_and_count_grayscale_labels('try_train')
print(f"Number of classes in training set: {num_classes_train}")

# 统计验证集中的类别数量
num_classes_val = load_and_count_grayscale_labels('try_val')
print(f"Number of classes in validation set: {num_classes_val}")

# 统计验证集中的类别数量
num_classes_test = load_and_count_grayscale_labels('try_test')
print(f"Number of classes in validation set: {num_classes_test}")