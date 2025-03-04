import pandas as pd
import pymysql
import numpy as np
from tqdm import tqdm

# 连接到MySQL数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'  # 使用 utf8mb4，适应更广泛的字符编码
)

# 创建一个游标对象
cursor = conn.cursor()

# 创建 elo_scores_safety 表（如果表不存在）
create_table_sql = """
CREATE TABLE IF NOT EXISTS elo_scores_safety (
    image_id VARCHAR(255) NOT NULL,
    elo_score FLOAT NOT NULL,
    win_count INT NOT NULL,
    lose_count INT NOT NULL,
    PRIMARY KEY (image_id)
);
"""
# 执行SQL语句，创建表
cursor.execute(create_table_sql)

# 提交更改并关闭游标
conn.commit()
cursor.close()

# 1. 从数据库中获取 `image_stats_safety` 数据表中的图片统计信息
sql = "SELECT * FROM image_stats_safety"
image_stats_df = pd.read_sql(sql, conn)

# 2. 从数据库中获取 `pp2` 数据表中的比赛结果，并分批读取
sql = "SELECT * FROM pp2 WHERE category='safety'"
chunk_size = 10000  # 增加每次读取的数据量

# 初始化一个空的 DataFrame 来存储所有比赛结果
comparison_df = pd.DataFrame()

# 分批读取数据并显示进度
for chunk in tqdm(pd.read_sql(sql, conn, chunksize=chunk_size), desc="读取比赛结果"):
    comparison_df = pd.concat([comparison_df, chunk], ignore_index=True)

# 关闭数据库连接
conn.close()

# 测试插入数据库
test_data = pd.DataFrame({
    'image_id': ['test_image_1', 'test_image_2'],
    'elo_score': [1000.0, 1500.0],
    'win_count': [10, 15],
    'lose_count': [5, 5]
})

# 连接到MySQL数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'
)

# 定义批次大小
batch_size = 1000  # 增加每次插入的数据量

# 分批插入数据并显示进度
for i in tqdm(range(0, len(test_data), batch_size), desc="测试插入数据库中"):
    batch = test_data.iloc[i:i + batch_size]
    cursor = conn.cursor()
    for _, row in batch.iterrows():
        sql = """
        INSERT INTO elo_scores_safety (image_id, elo_score, win_count, lose_count)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(sql, (row['image_id'], row['elo_score'], row['win_count'], row['lose_count']))
    conn.commit()

# 关闭连接
cursor.close()
conn.close()

print("测试数据插入完成！")

# 删除测试数据
# 连接到MySQL数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'
)

# 删除测试数据的SQL语句
delete_test_data_sql = """
DELETE FROM elo_scores_safety WHERE image_id IN ('test_image_1', 'test_image_2');
"""

# 执行删除操作
cursor = conn.cursor()
cursor.execute(delete_test_data_sql)
conn.commit()

# 关闭连接
cursor.close()
conn.close()

print("测试数据已删除！")

# 3. 定义计算加权评分的函数
def calculate_weighted_scores(comparison_df, image_stats_df):
    # 创建一个字典存储每张图片的加权评分
    scores = {}

    # 使用 tqdm 显示进度条
    for _, row in tqdm(comparison_df.iterrows(), desc="计算加权评分"):
        left_id = row['left_id']
        right_id = row['right_id']
        winner = row['winner']
        
        # 获取每张图片的胜率和胜场数
        left_stats = image_stats_df.loc[image_stats_df['image_id'] == left_id].iloc[0]
        right_stats = image_stats_df.loc[image_stats_df['image_id'] == right_id].iloc[0]
        
        left_win_rate = left_stats['win_rate']
        right_win_rate = right_stats['win_rate']
        left_win_count = left_stats['win_count']
        right_win_count = right_stats['win_count']
        
        # 计算胜场和胜率的加权分数
        left_score = left_win_rate * left_win_count
        right_score = right_win_rate * right_win_count

        # 根据比赛结果调整评分
        if winner == 'left':
            scores[left_id] = scores.get(left_id, 0) + left_score
            scores[right_id] = scores.get(right_id, 0) - right_score
        elif winner == 'right':
            scores[left_id] = scores.get(left_id, 0) - left_score
            scores[right_id] = scores.get(right_id, 0) + right_score
        elif winner == 'equal':
            # 如果是平局，给两者加权评分
            scores[left_id] = scores.get(left_id, 0) + 0.5 * left_score
            scores[right_id] = scores.get(right_id, 0) + 0.5 * right_score
    
    # 将结果存储到 DataFrame 中
    weighted_scores_df = pd.DataFrame(list(scores.items()), columns=['image_id', 'elo_score'])
    
    # 调整评分范围到0-100
    weighted_scores_df['elo_score'] = (weighted_scores_df['elo_score'] - weighted_scores_df['elo_score'].min()) / (weighted_scores_df['elo_score'].max() - weighted_scores_df['elo_score'].min()) * 100
    
    return weighted_scores_df

# 4. 计算加权评分
weighted_scores_df = calculate_weighted_scores(comparison_df, image_stats_df)

# 5. 将加权评分插入到数据库的 `elo_scores_safety` 表中
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'
)

# 定义批次大小
batch_size = 1000  # 增加每次插入的数据量

# 分批插入数据并显示进度
for i in tqdm(range(0, len(weighted_scores_df), batch_size), desc="插入数据库中"):
    batch = weighted_scores_df.iloc[i:i + batch_size]
    cursor = conn.cursor()
    for _, row in batch.iterrows():
        sql = """
        INSERT INTO elo_scores_safety (image_id, elo_score, win_count, lose_count)
        VALUES (%s, %s, 0, 0)
        ON DUPLICATE KEY UPDATE elo_score = VALUES(elo_score)
        """
        cursor.execute(sql, (row['image_id'], row['elo_score']))
    conn.commit()

# 关闭连接
cursor.close()
conn.close()

print("加权评分计算完成，并已成功插入数据库！")