import pymysql
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 连接到MySQL数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'
)

# 从数据库读取景物比例数据
ss_query = "SELECT * FROM pp2_ss"
ss_df = pd.read_sql(ss_query, con=conn)

# 从数据库读取图片评分数据
scores_query = "SELECT image_id, score FROM image_scores_wealthy"  # 假设评分数据在image_scores_wealthy表中
scores_df = pd.read_sql(scores_query, con=conn)

# 合并景物比例数据和评分数据
merged_df = pd.merge(ss_df, scores_df, left_on='image', right_on='image_id')

# 删除不必要的列（如image_id等）
merged_df = merged_df.drop(columns=['image_id'])

# 特征列：景物比例数据（除去评分列）
feature_columns = [
    'Road', 'Sidewalk', 'Building', 'Wall', 'Fence', 'Pole', 
    'Traffic_Light', 'Traffic_Sign', 'Vegetation', 'Terrain', 'Sky', 
    'Person', 'Rider', 'Car', 'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle', 'Other'
]

# 特征和目标变量
X = merged_df[feature_columns]  # 景物比例数据
y = merged_df['score']  # 评分

# 将数据分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出结果
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 关闭数据库连接
conn.close()

