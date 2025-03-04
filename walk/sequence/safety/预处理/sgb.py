import numpy as np
import pandas as pd
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

# 定义批次大小
batch_size = 100  # 每次处理 100 条数据

# 随机梯度下降 (SGD) 更新步骤
def sgd_update(scores, batch, learning_rate):
    gradients = np.zeros_like(scores)  # 初始化梯度
    for _, row in batch.iterrows():
        i = image_to_index[row['left_id']]
        j = image_to_index[row['right_id']]
        s_i, s_j = scores[i], scores[j]
        
        # 计算 P(i > j) 和 P(j > i)
        p_i_j = np.exp(s_i) / (np.exp(s_i) + np.exp(s_j))
        p_j_i = 1 - p_i_j  # P(j > i)

        # 计算梯度
        grad_i = row['left_win'] - p_i_j
        grad_j = row['right_win'] - p_j_i
        gradients[i] += grad_i
        gradients[j] += grad_j

    # 使用学习率更新评分
    scores -= learning_rate * gradients
    return scores

# 定义SGD的训练过程
def sgd_train(df, initial_scores, learning_rate=0.01, n_epochs=1000, batch_size=100):
    scores = initial_scores.copy()
    for epoch in tqdm(range(n_epochs), desc="训练中"):
        # 每次迭代时打乱数据
        df = df.sample(frac=1).reset_index(drop=True)
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            scores = sgd_update(scores, batch, learning_rate)
    return scores

# 训练模型
final_scores = sgd_train(df, initial_scores, learning_rate=0.01, n_epochs=1000, batch_size=100)

# 将最终的评分保存到数据库
final_scores_dict = {index_to_image[i]: final_scores[i] for i in range(n_images)}

# 转换为 DataFrame 并保存
final_scores_df = pd.DataFrame(final_scores_dict.items(), columns=['image_id', 'sgd_score'])

# 将评分结果存入数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'
)
final_scores_df.to_sql('sgd_scores_safety', con=conn, if_exists='replace', index=False)
conn.close()

print("随机梯度下降（SGD）评分计算完成并存入数据库！")
