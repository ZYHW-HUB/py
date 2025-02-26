import pymysql
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 连接到MySQL数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'  # 使用 utf8mb4，适应更广泛的字符编码
)

cursor = conn.cursor()

# 查询 'category' 为 'safety' 的数据
query = """
SELECT * FROM pp2_combined WHERE category = 'safety';
"""
cursor.execute(query)
results = cursor.fetchall()

# 将查询结果转换为DataFrame
columns = [desc[0] for desc in cursor.description]
df = pd.DataFrame(results, columns=columns)

# 关闭数据库连接
cursor.close()
conn.close()

# 数据预处理
# 假设我们需要对 'winner' 列进行编码并准备特征和标签
label_encoder = LabelEncoder()

# 对 'winner' 进行标签编码
df['winner_label_encoded'] = label_encoder.fit_transform(df['winner'])

# 特征选择：假设我们选择所有与道路、建筑相关的列作为特征
features = ['Road_left', 'Sidewalk_left', 'Building_left', 'Wall_left', 'Fence_left',
            'Pole_left', 'Traffic_Light_left', 'Traffic_Sign_left', 'Vegetation_left', 'Terrain_left',
            'Sky_left', 'Person_left', 'Rider_left', 'Car_left', 'Truck_left', 'Bus_left', 
            'Train_left', 'Motorcycle_left', 'Bicycle_left', 'Other_left', 
            'Road_right', 'Sidewalk_right', 'Building_right', 'Wall_right', 'Fence_right', 
            'Pole_right', 'Traffic_Light_right', 'Traffic_Sign_right', 'Vegetation_right', 'Terrain_right',
            'Sky_right', 'Person_right', 'Rider_right', 'Car_right', 'Truck_right', 'Bus_right', 
            'Train_right', 'Motorcycle_right', 'Bicycle_right', 'Other_right']

# 提取特征数据和目标标签
X = df[features]
y = df['winner_label_encoded']

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 可以在这里创建并训练你的模型，例如使用 TensorFlow 或其他机器学习库
# 示例：训练一个简单的模型（以深度学习为例）



# 构建一个简单的神经网络模型
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # 输出层，数量与类别数量一致

# 编译模型
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"模型测试集准确率: {accuracy * 100:.2f}%")
