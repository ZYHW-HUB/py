import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# 设置Cityscapes数据集路径
CITYSCAPES_PATH = '语义分割\\archive'

# 加载图像和标签
def load_cityscapes_data(image_set='train'):
    images = []
    labels = []
    img_path = os.path.join(CITYSCAPES_PATH, image_set, 'img')
    label_path = os.path.join(CITYSCAPES_PATH, image_set, 'label')
    
    img_files = sorted(os.listdir(img_path))
    label_files = sorted(os.listdir(label_path))
    
    for img_file, label_file in zip(img_files, label_files):
        if not img_file.lower().endswith('.png') or not label_file.lower().endswith('.png'):
            continue
        
        img = cv2.imread(os.path.join(img_path, img_file))
        label = cv2.imread(os.path.join(label_path, label_file), cv2.IMREAD_GRAYSCALE)
        
        if img is None or label is None:
            print(f"Error reading image {img_file} or label {label_file}")
            continue
        
        images.append(img)
        labels.append(label)
    
    return np.array(images), np.array(labels)

# 加载训练数据
train_images, train_labels = load_cityscapes_data('train')
# 加载验证数据
val_images, val_labels = load_cityscapes_data('val')

# 正则化图像数据
train_images = train_images / 255.0
val_images = val_images / 255.0

# 将标签转换为整数类型
train_labels = train_labels.astype(np.int32)
val_labels = val_labels.astype(np.int32)

# 构建DeepLabV3+模型
def DeepLabV3Plus(input_shape, num_classes):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Atrous Spatial Pyramid Pooling
    b4 = layers.GlobalAveragePooling2D()(base_model.output)
    b4 = layers.Reshape((1, 1, b4.shape[-1]))(b4)
    b4 = layers.Conv2D(256, (1, 1), padding='same')(b4)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.ReLU()(b4)
    b4 = layers.UpSampling2D(size=(input_shape[0] // 32, input_shape[1] // 32), interpolation='bilinear')(b4)

    b0 = layers.Conv2D(256, (1, 1), padding='same')(base_model.output)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.ReLU()(b0)

    b1 = layers.Conv2D(256, (3, 3), padding='same', dilation_rate=6)(base_model.output)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.ReLU()(b1)

    b2 = layers.Conv2D(256, (3, 3), padding='same', dilation_rate=12)(base_model.output)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.ReLU()(b2)

    b3 = layers.Conv2D(256, (3, 3), padding='same', dilation_rate=18)(base_model.output)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.ReLU()(b3)

    x = layers.Concatenate()([b4, b0, b1, b2, b3])
    x = layers.Conv2D(256, (1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D(size=(input_shape[0] // 4, input_shape[1] // 4), interpolation='bilinear')(x)

    low_level_feature = base_model.get_layer('conv2_block3_1_relu').output
    low_level_feature = layers.Conv2D(48, (1, 1), padding='same')(low_level_feature)
    low_level_feature = layers.BatchNormalization()(low_level_feature)
    low_level_feature = layers.ReLU()(low_level_feature)
    low_level_feature = layers.UpSampling2D(size=(input_shape[0] // 4, input_shape[1] // 4), interpolation='bilinear')(low_level_feature)

    x = layers.Concatenate()([x, low_level_feature])
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(256, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(num_classes, (1, 1), padding='same')(x)
    x = layers.UpSampling2D(size=(input_shape[0] // 2, input_shape[1] // 2), interpolation='bilinear')(x)

    return models.Model(inputs=base_model.input, outputs=x)

# 确保 `input_shape` 是正确的
input_shape = train_images.shape[1:]  # 形如 (height, width, channels)

# 打印 `input_shape` 以确认
print("Input shape:", input_shape)

# 创建DeepLabV3+模型
model = DeepLabV3Plus(input_shape=input_shape, num_classes=3)  # 假设有3个类别

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10, batch_size=16)

# 绘制训练和验证的损失曲线
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# 绘制训练和验证的准确率曲线
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# 预测一个示例
predicted_label = model.predict(np.expand_dims(val_images[0], axis=0))
predicted_label = np.argmax(predicted_label, axis=-1)[0]

# 显示原始图像、标签和预测结果
def display_prediction(original, label, prediction):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.title('Label Image')
    plt.imshow(label)
    plt.subplot(1, 3, 3)
    plt.title('Prediction Image')
    plt.imshow(prediction)
    plt.show()

display_prediction(val_images[0], val_labels[0], predicted_label)
