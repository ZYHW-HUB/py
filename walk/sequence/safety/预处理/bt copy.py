import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pymysql
from tqdm import tqdm

# 连接到 MySQL 数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'
)

# 读取 `pp2` 表中的数据，过滤 category = 'safety'
query = """
SELECT left_id, right_id, winner
FROM pp2
WHERE category = 'safety';
"""
df = pd.read_sql(query, conn)
conn.close()

# 处理 winner 字段，将其转换为 left_win, right_win, draw
df['left_win'] = (df['winner'] == 'left').astype(int)
df['right_win'] = (df['winner'] == 'right').astype(int)
df['draw'] = (df['winner'] == 'equal').astype(int)

# 删除原来的 winner 列
df = df.drop(columns=['winner'])

# 获取所有唯一的图片 ID
all_images = set(df['left_id']).union(set(df['right_id']))
image_to_index = {img: idx for idx, img in enumerate(all_images)}
index_to_image = {idx: img for img, idx in image_to_index.items()}
n_images = len(all_images)

# 初始化评分为零
initial_scores = np.zeros(n_images)

# 定义一个处理单个批次的对数似然函数
def bt_likelihood_batch(scores, batch):
    log_likelihood = 0
    for _, row in batch.iterrows():
        i = image_to_index[row['left_id']]
        j = image_to_index[row['right_id']]
        s_i, s_j = scores[i], scores[j]
        
        # 计算 P(i > j) 和 P(j > i)
        p_i_j = np.exp(s_i) / (np.exp(s_i) + np.exp(s_j))
        p_j_i = 1 - p_i_j  # P(j > i)

        # 累加对数似然
        log_likelihood += row['left_win'] * np.log(p_i_j) + row['right_win'] * np.log(p_j_i)

    return -log_likelihood  # 返回负的对数似然

# 定义批次大小
batch_size = 100  # 每次处理 100 条数据

# 定义一个批次优化函数
def bt_likelihood(scores):
    log_likelihood = 0
    # 使用 tqdm 显示进度条
    for i in tqdm(range(0, len(df), batch_size), desc="优化中", unit="批次"):
        batch = df.iloc[i:i + batch_size]
        log_likelihood += bt_likelihood_batch(scores, batch)
    return log_likelihood

# 使用 L-BFGS-B 算法进行最大似然估计优化
result = minimize(bt_likelihood, initial_scores, method='L-BFGS-B', options={'disp': True, 'maxiter': 1000})

# 获取最终的评分
bt_scores = result.x
bt_scores_dict = {index_to_image[i]: bt_scores[i] for i in range(n_images)}

# 将结果转为 DataFrame
bt_scores_df = pd.DataFrame(bt_scores_dict.items(), columns=['image_id', 'bt_score'])

# 将评分结果存入数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'
)
bt_scores_df.to_sql('bt_scores_safety', con=conn, if_exists='replace', index=False)
conn.close()

print("Bradley-Terry 评分计算完成并存入数据库！")
