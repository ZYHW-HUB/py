import pandas as pd
import pymysql

# 连接数据库
connection = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'
)

# 查询数据
query = """
SELECT
    Road_left, Sidewalk_left, Building_left, Wall_left, Fence_left, Pole_left,
    Traffic_Light_left, Traffic_Sign_left, Vegetation_left, Terrain_left, Sky_left,
    Person_left, Rider_left, Car_left, Truck_left, Bus_left, Train_left, Motorcycle_left,
    Bicycle_left, Other_left,
    Road_right, Sidewalk_right, Building_right, Wall_right, Fence_right, Pole_right,
    Traffic_Light_right, Traffic_Sign_right, Vegetation_right, Terrain_right, Sky_right,
    Person_right, Rider_right, Car_right, Truck_right, Bus_right, Train_right, Motorcycle_right,
    Bicycle_right, Other_right,
    winner_label_encoded
FROM pp2_combined
WHERE category = 'safety';
"""

# 执行查询并加载数据
df = pd.read_sql(query, connection)

# 关闭连接
connection.close()
# 定义映射关系
label_mapping = {1: 1, 0: 0, 2: -1}

# 替换标签值
df['score'] = df['winner_label_encoded'].replace(label_mapping)
features = [
    'Road_left', 'Sidewalk_left', 'Building_left', 'Wall_left', 'Fence_left', 'Pole_left',
    'Traffic_Light_left', 'Traffic_Sign_left', 'Vegetation_left', 'Terrain_left', 'Sky_left',
    'Person_left', 'Rider_left', 'Car_left', 'Truck_left', 'Bus_left', 'Train_left', 'Motorcycle_left',
    'Bicycle_left', 'Other_left',
    'Road_right', 'Sidewalk_right', 'Building_right', 'Wall_right', 'Fence_right', 'Pole_right',
    'Traffic_Light_right', 'Traffic_Sign_right', 'Vegetation_right', 'Terrain_right', 'Sky_right',
    'Person_right', 'Rider_right', 'Car_right', 'Truck_right', 'Bus_right', 'Train_right', 'Motorcycle_right',
    'Bicycle_right', 'Other_right'
]

X = df[features].values
y = df['score'].values

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建模型
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # 输出评分，只有一个节点

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
from tensorflow.keras.callbacks import ModelCheckpoint

# 定义保存路径
checkpoint_path = 'best_model.keras'

# 创建 ModelCheckpoint 回调函数
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',  # 监控验证集损失
    verbose=1,
    save_best_only=True,  # 仅保存验证集损失最小的模型
    mode='min',  # 'min' 表示监控指标越小越好
    save_weights_only=False  # 保存整个模型，包括结构和权重
)

# 训练模型时添加回调函数
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

from sklearn.metrics import mean_squared_error

# 预测评分
y_pred = model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
