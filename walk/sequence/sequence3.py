import numpy as np
import pymysql
from sklearn.preprocessing import LabelEncoder

def load_data_from_db(batch_size=1000, offset=0):
    conn = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='123',
        database='scene',
        charset='utf8mb4'
    )
    cursor = conn.cursor()
    
    # 查询数据库中的数据
    query = f"""
    SELECT left_id, right_id, winner_label_encoded, 
           Road_left, Sidewalk_left, Building_left, Wall_left, Fence_left, 
           Pole_left, Traffic_Light_left, Traffic_Sign_left, Vegetation_left, 
           Terrain_left, Sky_left, Person_left, Rider_left, Car_left, Truck_left, 
           Bus_left, Train_left, Motorcycle_left, Bicycle_left, Other_left,
           Road_right, Sidewalk_right, Building_right, Wall_right, Fence_right, 
           Pole_right, Traffic_Light_right, Traffic_Sign_right, Vegetation_right, 
           Terrain_right, Sky_right, Person_right, Rider_right, Car_right, Truck_right, 
           Bus_right, Train_right, Motorcycle_right, Bicycle_right, Other_right
    FROM pp2_combined
    LIMIT {batch_size} OFFSET {offset};
    """
    cursor.execute(query)
    results = cursor.fetchall()
    
    conn.close()
    
    return results

def preprocess_data(data):
    left_features = []
    right_features = []
    labels = []
    
    for row in data:
        # 左侧特征：22个景物比例数据
        left_feature = np.array(row[3:25])
        # 右侧特征：22个景物比例数据
        right_feature = np.array(row[25:47])
        # 标签：winner_label_encoded（0 或 1）
        label = row[2]
        
        left_features.append(left_feature)
        right_features.append(right_feature)
        labels.append(label)
    
    # 转换为 NumPy 数组
    left_features = np.array(left_features)
    right_features = np.array(right_features)
    labels = np.array(labels)
    
    return left_features, right_features, labels
