import pymysql
import pandas as pd

# 连接到MySQL数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'  # 使用 utf8mb4，适应更广泛的字符编码
)

# 假设数据已加载到DataFrame中，包含景物比例和评分
data = pd.read_sql("SELECT * FROM pp2_ss", con=conn)  # 假设你的景物比例数据表名是 pp2_ss
scores = pd.read_sql("SELECT image_id, score FROM image_scores WHERE split = 'train'", con=conn)  # 假设评分表为 image_scores_safety

# 假设 'image_id' 作为主键，合并景物比例数据与评分数据
merged_df = pd.merge(data, scores, left_on='image', right_on='image_id')

# 计算平方项和三次方项
for column in data.columns[1:]:  # 遍历所有景物比例列
    merged_df[f'{column}_squared'] = merged_df[column] ** 2
    merged_df[f'{column}_cubed'] = merged_df[column] ** 3

# 选择数值型列进行相关性计算
numerical_columns = merged_df.select_dtypes(include=['float64']).columns

# 计算相关系数
correlation_matrix = merged_df[numerical_columns].corr()

# 只选择与评分相关的项
score_columns = [col for col in merged_df.columns if 'score' in col]
correlation_with_scores = correlation_matrix[score_columns]

# 打印结果
print(correlation_with_scores)

# 关闭连接
conn.close()
import pandas as pd

# 假设 correlation_with_scores 是您之前计算得到的相关系数矩阵
# # 将相关系数矩阵保存到 Excel 文件
# correlation_with_scores.to_excel('correlation_matrix.xlsx', sheet_name='Correlation')
correlation_with_scores.to_csv('correlation_matrix.csv', index=True)
