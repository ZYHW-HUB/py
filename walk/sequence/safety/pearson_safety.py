import pymysql
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 连接到MySQL数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'  # 使用 utf8mb4，适应更广泛的字符编码
)

# 从数据库中读取景物比例数据
scene_ratio_query = "SELECT * FROM pp2_ss"
scene_ratio_df = pd.read_sql(scene_ratio_query, con=conn)

# 从数据库中读取评分数据
scores_query = "SELECT image_id, score FROM image_scores_safety_S"
scores_df = pd.read_sql(scores_query, con=conn)

# 关闭数据库连接
conn.close()

# 合并数据，确保只保留同时存在于两张表中的 image_id
merged_df = pd.merge(scene_ratio_df, scores_df, left_on='image', right_on='image_id')

# 删除非数值列，保留数值列进行相关性计算
merged_df = merged_df.drop(columns=['image_id', 'image'])  # 去除 image_id 和 image 字符串列

# 计算相关矩阵和显著性矩阵
correlation_matrix = merged_df.corr()
p_values_matrix = pd.DataFrame(np.zeros_like(correlation_matrix), columns=correlation_matrix.columns, index=correlation_matrix.index)

for col in merged_df.columns:
    for row in merged_df.columns:
        if col != row:
            corr, p_value = pearsonr(merged_df[col], merged_df[row])
            correlation_matrix.loc[row, col] = corr
            p_values_matrix.loc[row, col] = p_value

# 提取评分与各景物比例之间的相关系数和显著性
score_correlations = correlation_matrix['score'].drop('score')  # 去除与自身的相关性
score_p_values = p_values_matrix['score'].drop('score')  # 去除与自身的显著性

print("各景物比例与评分之间的皮尔逊相关系数：")
print(score_correlations)
print("各景物比例与评分之间的显著性（p值）：")
print(score_p_values)

# 保存相关系数和显著性到CSV文件
score_correlations.to_csv('walk/sequence/safety/correlation_matrix_safety.csv', index=True)
score_p_values.to_csv('walk/sequence/safety/p_values_matrix_safety.csv', index=True)

# 生成热力图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.savefig('walk/sequence/safety/correlation_matrix_heatmap_safety.png')
plt.show()

