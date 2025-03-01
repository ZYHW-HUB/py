import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pymysql
import pandas as pd

# 连接到 MySQL 数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'  # 使用 utf8mb4，适应更广泛的字符编码
)

# 创建游标
cursor = conn.cursor()

# 查询需要的字段
query = """
SELECT 
    Road_left, Sidewalk_left, Building_left, Wall_left, Fence_left, 
    Pole_left, Traffic_Light_left, Traffic_Sign_left, Vegetation_left, 
    Terrain_left, Sky_left, Person_left, Rider_left, Car_left, 
    Truck_left, Bus_left, Train_left, Motorcycle_left, Bicycle_left, Other_left,
    Road_right, Sidewalk_right, Building_right, Wall_right, Fence_right, 
    Pole_right, Traffic_Light_right, Traffic_Sign_right, Vegetation_right, 
    Terrain_right, Sky_right, Person_right, Rider_right, Car_right, 
    Truck_right, Bus_right, Train_right, Motorcycle_right, Bicycle_right, Other_right,
    winner_label_encoded
FROM pp2_combined
WHERE category = 'safety'
;
"""

# 执行查询并加载数据到 DataFrame
cursor.execute(query)
data = cursor.fetchall()

# 获取列名
columns = [
    "Road_left", "Sidewalk_left", "Building_left", "Wall_left", "Fence_left", 
    "Pole_left", "Traffic_Light_left", "Traffic_Sign_left", "Vegetation_left", 
    "Terrain_left", "Sky_left", "Person_left", "Rider_left", "Car_left", 
    "Truck_left", "Bus_left", "Train_left", "Motorcycle_left", "Bicycle_left", "Other_left",
    "Road_right", "Sidewalk_right", "Building_right", "Wall_right", "Fence_right", 
    "Pole_right", "Traffic_Light_right", "Traffic_Sign_right", "Vegetation_right", 
    "Terrain_right", "Sky_right", "Person_right", "Rider_right", "Car_right", 
    "Truck_right", "Bus_right", "Train_right", "Motorcycle_right", "Bicycle_right", "Other_right",
    "winner_label_encoded"
]

# 将查询结果转换为 DataFrame
df = pd.DataFrame(data, columns=columns)

# 关闭游标和连接
cursor.close()
conn.close()

# 特征和目标变量
X = df.drop(columns=["winner_label_encoded"])
y = df["winner_label_encoded"]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建神经网络模型
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型并显示训练进度
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
