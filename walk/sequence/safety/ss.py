import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 路径设置：语义分割结果图像所在目录和CSV标签文件
image_dir = "data/seg_results"  # 存放分割结果图像的目录
label_csv = "data/labels.csv"   # CSV 文件，包含 "image" 和 "score" 两列

# 读取 CSV 标签
df = pd.read_csv(label_csv)
print("样本数量:", len(df))

# 图像参数
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# 定义函数：根据文件名读取图像并预处理
def load_and_preprocess(image_filename, score):
    # 构造完整路径
    img_path = tf.strings.join([image_dir, image_filename], separator=os.sep)
    # 读取图像文件
    image = tf.io.read_file(img_path)
    # 假设图像为PNG/JPEG格式
    image = tf.image.decode_image(image, channels=3)
    # 转换为float32，并归一化到 [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    # 调整图像大小
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    return image, score

# 构建tf.data.Dataset
filenames = df["image"].tolist()
scores = df["score"].tolist()

# 创建 Dataset 对象
dataset = tf.data.Dataset.from_tensor_slices((filenames, scores))
dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 分割数据集：例如 80% 训练，20% 验证
total_samples = len(df)
train_size = int(0.8 * total_samples)
val_size = total_samples - train_size

dataset = dataset.cache()
train_dataset = dataset.take(train_size // BATCH_SIZE)
val_dataset = dataset.skip(train_size // BATCH_SIZE)

# 构建简单的 CNN 回归模型
def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1)  # 输出评分（回归输出）
    ])
    return model

model = create_model((IMG_HEIGHT, IMG_WIDTH, 3))
model.compile(optimizer="adam", loss="mse")

# 显示模型结构
model.summary()

# 训练模型，同时显示训练进度（verbose=1 默认会显示每个 epoch 进度）
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=50)

# 预测并评估模型
# 先将验证集转换为 NumPy 数组（这里只取全部验证集）
y_true = []
y_pred = []
for batch_images, batch_scores in val_dataset:
    preds = model.predict(batch_images)
    y_pred.extend(preds.flatten())
    y_true.extend(batch_scores.numpy())

mse_value = mean_squared_error(y_true, y_pred)
r2_value = r2_score(y_true, y_pred)
print(f"Mean Squared Error: {mse_value}")
print(f"R-squared: {r2_value}")

# 可视化预测结果
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.title("True vs Predicted Scores")
plt.show()
