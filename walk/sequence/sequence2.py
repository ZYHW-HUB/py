import pymysql
import pandas as pd
import numpy as np
# 连接到MySQL数据库
conn = pymysql.Connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='123',
    db='scene',
    charset='utf8mb4'  # 使用 utf8mb4，适应更广泛的字符编码
)

# 创建游标对象，用于执行SQL查询
cursor = conn.cursor()

# 查询 `pp2` 表中的比较数据
query_comparisons = """
SELECT left_id, right_id, winner, left_lat, left_long, right_lat, right_long
FROM pp2
WHERE category='safety';
"""
cursor.execute(query_comparisons)

# 获取所有查询结果
comparisons_data = cursor.fetchall()

# 关闭游标和连接
cursor.close()
conn.close()

# 将比较数据转换为DataFrame
df_comparisons = pd.DataFrame(comparisons_data, columns=["left_id", "right_id", "winner", "left_lat", "left_long", 
                                                         "right_lat", "right_long"])
# 查看数据
print("原始数据：")
print(df_comparisons.head())

# 对 'winner' 列进行独热编码
df_comparisons_encoded = pd.get_dummies(df_comparisons, columns=['winner'], drop_first=False)

# 查看独热编码后的数据
print("\n独热编码后的数据：")
print(df_comparisons_encoded.head())

# 查看数据
print(df_comparisons.head())
# 连接到MySQL数据库以获取景物比例数据
conn = pymysql.Connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='123',
    db='scene',
    charset='utf8mb4'
)

# 创建游标对象，用于执行SQL查询
cursor = conn.cursor()

# 查询 `pp2_ss` 表中的景物比例数据
query_features = """
SELECT image, Road, Sidewalk, Building, Wall, Fence, Pole, Traffic_Light, Traffic_Sign, 
       Vegetation, Terrain, Sky, Person, Rider, Car, Truck, Bus, Train, Motorcycle, Bicycle, Other
FROM pp2_ss;
"""
cursor.execute(query_features)

# 获取所有查询结果
features_data = cursor.fetchall()

# 关闭游标和连接
cursor.close()
conn.close()

# 将景物比例数据转换为DataFrame
df_features = pd.DataFrame(features_data, columns=["image", "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", 
                                                   "Traffic_Light", "Traffic_Sign", "Vegetation", "Terrain", "Sky", 
                                                   "Person", "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle", 
                                                   "Bicycle", "Other"])

# 查看数据
print(df_features.head())
# 合并景物比例数据到比较数据中
def get_image_features(image_id):
    # 查找对应图片的特征
    return df_features[df_features["image"] == image_id].drop(columns=["image"]).values[0]

# 创建训练数据
X = []
y = []

for idx, row in df_comparisons.iterrows():
    features_A = get_image_features(row["left_id"])
    features_B = get_image_features(row["right_id"])
    
    # 组合特征：将左边和右边的图片特征拼接
    X.append(np.concatenate([features_A, features_B]))
    
    # 标签：'left' -> 1, 'right' -> 0, 'equal' -> 0.5
    if row["winner"] == "left":
        y.append(1)
    elif row["winner"] == "right":
        y.append(0)
    else:  # 'equal'
        y.append(0.5)

X = np.array(X)
y = np.array(y)

# 查看样本数据
print(X[:5], y[:5])
