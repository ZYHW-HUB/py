import pandas as pd
import pymysql

# 连接到MySQL数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'
)

# 从数据库中读取景物比例数据
scene_ratio_query = "SELECT * FROM pp2_ss"
scene_ratio_df = pd.read_sql(scene_ratio_query, con=conn)

# 从数据库中读取评分数据
scores_query = "SELECT * FROM image_scores_safety"
scores_df = pd.read_sql(scores_query, con=conn)

# 关闭数据库连接
conn.close()

# 合并数据，确保只保留同时存在于两张表中的 image_id
merged_df = pd.merge(scene_ratio_df, scores_df, left_on='image', right_on='image_id')

# 删除非数值列，保留数值列进行相关性计算
merged_df = merged_df.drop(columns=['image_id', 'image'])  # 去除 image_id 和 image 字符串列

# 计算相关矩阵
correlation_matrix = merged_df.corr()

# 提取评分与各景物比例之间的相关系数
score_correlations = correlation_matrix['score'].drop('score')  # 去除与自身的相关性

print("各景物比例与评分之间的皮尔逊相关系数：")
print(score_correlations)
