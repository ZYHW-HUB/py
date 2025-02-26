# 只选择 category 为 "safety" 的数据
df_safety = df[df['category'] == 'safety']
# 特征列 (X)
X_safety = df_safety[['Road_left', 'Sidewalk_left', 'Building_left', 'Wall_left', 'Fence_left', 'Pole_left', 
                      'Traffic_Light_left', 'Traffic_Sign_left', 'Vegetation_left', 'Terrain_left', 'Sky_left', 
                      'Person_left', 'Rider_left', 'Car_left', 'Truck_left', 'Bus_left', 'Train_left', 
                      'Motorcycle_left', 'Bicycle_left', 'Other_left', 'Road_right', 'Sidewalk_right', 
                      'Building_right', 'Wall_right', 'Fence_right', 'Pole_right', 'Traffic_Light_right', 
                      'Traffic_Sign_right', 'Vegetation_right', 'Terrain_right', 'Sky_right', 'Person_right', 
                      'Rider_right', 'Car_right', 'Truck_right', 'Bus_right', 'Train_right', 'Motorcycle_right', 
                      'Bicycle_right', 'Other_right']]

# 标签列 (y)
y_safety = df_safety[['winner_left', 'winner_right', 'winner_equal']]  # 目标变量
from sklearn.model_selection import train_test_split

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_safety, y_safety, test_size=0.2, random_state=42)
import tensorflow as tf

# 建立神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_dim=X_train.shape[1]),  # 输入层
    tf.keras.layers.Dense(64, activation='relu'),  # 隐藏层
    tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid')  # 输出层，对应三个目标：winner_left, winner_right, winner_equal
])

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")

import matplotlib.pyplot as plt

# 绘制训练损失和准确率
plt.figure(figsize=(12, 6))

# 绘制训练和验证的损失
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制训练和验证的准确率
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
# 预测新的数据
predictions = model.predict(X_test)

# 打印预测结果
print(predictions)
