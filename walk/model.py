import tensorflow as tf
import tensorflow_addons as tfa  # 如果需要额外的操作
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Reshape, Concatenate
from tensorflow.keras.models import Model

def build_deeplabv3_plus(num_classes, input_shape=(512, 512, 3)):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False)

    # 获取特征层
    x = base_model.output
    x = Conv2D(256, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # ASPP
    b0 = Conv2D(256, 1, padding='same')(x)
    b1 = tfa.layers.SpatialPyramidPooling2D([1, 6, 12, 18])(x)
    b2 = Conv2D(256, 3, dilation_rate=6, padding='same')(x)
    b3 = Conv2D(256, 3, dilation_rate=12, padding='same')(x)

    # Concatenate and upsample
    x = Concatenate()([b0, b1, b2, b3])
    x = Conv2D(256, 1, padding='same')(x)
    x = UpSampling2D(size=(4, 4))(x)

    # Low-level features
    low_level_features = base_model.get_layer('block_1_expand_relu').output
    low_level_features = Conv2D(48, 1, padding='same')(low_level_features)
    x = Concatenate()([x, low_level_features])

    # Final layers
    x = Conv2D(256, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_classes, 1)(x)
    x = UpSampling2D(size=(4, 4))(x)
    x = Reshape((input_shape[0] * input_shape[1], num_classes))(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model

# 构建模型
num_classes = 19  # Cityscapes 的类别数
input_shape = (512, 512, 3)
model = build_deeplabv3_plus(num_classes, input_shape)

# 加载预训练权重
checkpoint_path = 'walk/deeplabv3_cityscapes_train/model.ckpt.data-00000-of-00001'  # 通常是 model.ckpt 的路径
model.load_weights(checkpoint_path)

# import os
# import cv2
# import numpy as np
# from sklearn.model_selection import train_test_split

# def load_cityscapes_data(data_dir, split_ratio=0.2):
#     images_dir = os.path.join(data_dir, 'leftImg8bit/train')
#     labels_dir = os.path.join(data_dir, 'gtFine/train')

#     # 获取图像和标签文件名
#     image_files = []
#     label_files = []

#     for city in os.listdir(images_dir):
#         city_images_dir = os.path.join(images_dir, city)
#         city_labels_dir = os.path.join(labels_dir, city)

#         for filename in os.listdir(city_images_dir):
#             if filename.endswith('_leftImg8bit.png'):
#                 image_files.append(os.path.join(city_images_dir, filename))
#                 label_files.append(os.path.join(city_labels_dir, filename.replace('_leftImg8bit.png', '_gtFine_labelIds.png')))

#     # 划分数据集
#     X_train, X_val, y_train, y_val = train_test_split(image_files, label_files, test_size=split_ratio, random_state=42)

#     return X_train, X_val, y_train, y_val

# def preprocess_image(image_path, label_path):
#     # 加载图像
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

#     # 调整大小
#     image = cv2.resize(image, (512, 512))
#     label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)

#     # 归一化
#     image = image / 255.0

#     return image, label

# # 加载数据
# data_dir = 'path/to/cityscapes_dataset'
# X_train, X_val, y_train, y_val = load_cityscapes_data(data_dir)

# # 预处理数据
# train_data = [(preprocess_image(img, lbl)) for img, lbl in zip(X_train, y_train)]
# val_data = [(preprocess_image(img, lbl)) for img, lbl in zip(X_val, y_val)]

# # 分离图像和标签
# train_images, train_labels = zip(*train_data)
# val_images, val_labels = zip(*val_data)

# # 转换为 NumPy 数组
# train_images = np.array(train_images)
# train_labels = np.array(train_labels)
# val_images = np.array(val_images)
# val_labels = np.array(val_labels)