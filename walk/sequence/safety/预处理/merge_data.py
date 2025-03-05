import pymysql
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
# 创建表（如果不存在）
create_table_query = """
CREATE TABLE IF NOT EXISTS pp2_combined (
    left_id VARCHAR(255) NOT NULL,
    right_id VARCHAR(255) NOT NULL,
    left_lat FLOAT NOT NULL,
    left_long FLOAT NOT NULL,
    right_lat FLOAT NOT NULL,
    right_long FLOAT NOT NULL,
    winner_label_encoded INT,
    winner_left INT,
    winner_right INT,
    winner_equal INT,
    category VARCHAR(255),
    Road_left FLOAT NOT NULL,
    Sidewalk_left FLOAT NOT NULL,
    Building_left FLOAT NOT NULL,
    Wall_left FLOAT NOT NULL,
    Fence_left FLOAT NOT NULL,
    Pole_left FLOAT NOT NULL,
    Traffic_Light_left FLOAT NOT NULL,
    Traffic_Sign_left FLOAT NOT NULL,
    Vegetation_left FLOAT NOT NULL,
    Terrain_left FLOAT NOT NULL,
    Sky_left FLOAT NOT NULL,
    Person_left FLOAT NOT NULL,
    Rider_left FLOAT NOT NULL,
    Car_left FLOAT NOT NULL,
    Truck_left FLOAT NOT NULL,
    Bus_left FLOAT NOT NULL,
    Train_left FLOAT NOT NULL,
    Motorcycle_left FLOAT NOT NULL,
    Bicycle_left FLOAT NOT NULL,
    Other_left FLOAT NOT NULL,
    Road_right FLOAT NOT NULL,
    Sidewalk_right FLOAT NOT NULL,
    Building_right FLOAT NOT NULL,
    Wall_right FLOAT NOT NULL,
    Fence_right FLOAT NOT NULL,
    Pole_right FLOAT NOT NULL,
    Traffic_Light_right FLOAT NOT NULL,
    Traffic_Sign_right FLOAT NOT NULL,
    Vegetation_right FLOAT NOT NULL,
    Terrain_right FLOAT NOT NULL,
    Sky_right FLOAT NOT NULL,
    Person_right FLOAT NOT NULL,
    Rider_right FLOAT NOT NULL,
    Car_right FLOAT NOT NULL,
    Truck_right FLOAT NOT NULL,
    Bus_right FLOAT NOT NULL,
    Train_right FLOAT NOT NULL,
    Motorcycle_right FLOAT NOT NULL,
    Bicycle_right FLOAT NOT NULL,
    Other_right FLOAT NOT NULL
);
"""
cursor.execute(create_table_query)

# 提交事务
conn.commit()

# 定义每批次处理的记录数
batch_size = 1000
offset = 0

while True:
    # 查询当前批次的数据
    query = f"""
    SELECT p.left_id, p.right_id, p.left_lat, p.left_long, p.right_lat, p.right_long, p.winner, p.category,
           s.Road AS Road_left, s.Sidewalk AS Sidewalk_left, s.Building AS Building_left, s.Wall AS Wall_left, 
           s.Fence AS Fence_left, s.Pole AS Pole_left, s.Traffic_Light AS Traffic_Light_left, s.Traffic_Sign AS Traffic_Sign_left, 
           s.Vegetation AS Vegetation_left, s.Terrain AS Terrain_left, s.Sky AS Sky_left, s.Person AS Person_left, 
           s.Rider AS Rider_left, s.Car AS Car_left, s.Truck AS Truck_left, s.Bus AS Bus_left, s.Train AS Train_left, 
           s.Motorcycle AS Motorcycle_left, s.Bicycle AS Bicycle_left, s.Other AS Other_left,
           s2.Road AS Road_right, s2.Sidewalk AS Sidewalk_right, s2.Building AS Building_right, s2.Wall AS Wall_right, 
           s2.Fence AS Fence_right, s2.Pole AS Pole_right, s2.Traffic_Light AS Traffic_Light_right, s2.Traffic_Sign AS Traffic_Sign_right, 
           s2.Vegetation AS Vegetation_right, s2.Terrain AS Terrain_right, s2.Sky AS Sky_right, s2.Person AS Person_right, 
           s2.Rider AS Rider_right, s2.Car AS Car_right, s2.Truck AS Truck_right, s2.Bus AS Bus_right, s2.Train AS Train_right, 
           s2.Motorcycle AS Motorcycle_right, s2.Bicycle AS Bicycle_right, s2.Other AS Other_right
    FROM pp2 p
    JOIN pp2_ss s ON p.left_id = s.image
    JOIN pp2_ss s2 ON p.right_id = s2.image
    LIMIT {batch_size} OFFSET {offset};
    """
    cursor.execute(query)
    results = cursor.fetchall()

    if not results:
        break  # 如果当前批次没有数据，退出循环

    # 开始事务
    conn.begin()
    # 将查询结果转换为DataFrame
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(results, columns=columns)

    # 使用 LabelEncoder 对 'winner' 列进行标签编码
    label_encoder = LabelEncoder()
    df['winner_label_encoded'] = label_encoder.fit_transform(df['winner'])

    # 使用 OneHotEncoder 对编码后的结果进行独热编码
    onehot_encoder = OneHotEncoder(sparse_output=False)
    winner_encoded = onehot_encoder.fit_transform(df[['winner_label_encoded']])

    # 将独热编码结果转换为 DataFrame，并添加列名
    winner_columns = [f'winner_{i}' for i in range(winner_encoded.shape[1])]
    df_winner = pd.DataFrame(winner_encoded, columns=winner_columns)

    # 合并原始 DataFrame 和独热编码结果
    df = pd.concat([df, df_winner], axis=1)

    # 插入数据
    insert_query = """
    INSERT INTO pp2_combined (
        left_id, right_id, left_lat, left_long, right_lat, right_long,
        winner_label_encoded, winner_left, winner_right, winner_equal, category,
        Road_left, Sidewalk_left, Building_left, Wall_left, Fence_left, Pole_left,
        Traffic_Light_left, Traffic_Sign_left, Vegetation_left, Terrain_left, Sky_left,
        Person_left, Rider_left, Car_left, Truck_left, Bus_left, Train_left,
        Motorcycle_left, Bicycle_left, Other_left,
        Road_right, Sidewalk_right, Building_right, Wall_right, Fence_right, Pole_right,
        Traffic_Light_right, Traffic_Sign_right, Vegetation_right, Terrain_right, Sky_right,
        Person_right, Rider_right, Car_right, Truck_right, Bus_right, Train_right,
        Motorcycle_right, Bicycle_right, Other_right
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """

    # 循环处理每一条记录
    for _, row in df.iterrows():
        # 提取每一列数据并转换为元组形式
        data = (
            row['left_id'], row['right_id'], row['left_lat'], row['left_long'], row['right_lat'], row['right_long'],
            row['winner_label_encoded'], row['winner_0'], row['winner_1'], row['winner_2'], row['category'],
            row['Road_left'], row['Sidewalk_left'], row['Building_left'], row['Wall_left'], row['Fence_left'], row['Pole_left'],
            row['Traffic_Light_left'], row['Traffic_Sign_left'], row['Vegetation_left'], row['Terrain_left'], row['Sky_left'],
            row['Person_left'], row['Rider_left'], row['Car_left'], row['Truck_left'], row['Bus_left'], row['Train_left'], row['Motorcycle_left'], row['Bicycle_left'], row['Other_left'],
            row['Road_right'], row['Sidewalk_right'], row['Building_right'], row['Wall_right'], row['Fence_right'], row['Pole_right'],
            row['Traffic_Light_right'], row['Traffic_Sign_right'], row['Vegetation_right'], row['Terrain_right'], row['Sky_right'],
            row['Person_right'], row['Rider_right'], row['Car_right'], row['Truck_right'], row['Bus_right'], row['Train_right'], row['Motorcycle_right'], row['Bicycle_right'], row['Other_right']
        )

        # 执行插入操作
        cursor.execute(insert_query, data)

    # 提交事务
    conn.commit()

    # 更新偏移量
    offset += batch_size

    # 打印进度
    print(f"已处理 {offset} 条数据")

# 关闭游标和连接
cursor.close()
conn.close()

