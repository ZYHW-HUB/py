import pymysql
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 连接到MySQL数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'  # 使用 utf8mb4，适应更广泛的字符编码
)

# 从数据库中获取数据
query = """
SELECT ss.*, scores.score
FROM pp2_ss ss
JOIN image_scores_safety scores ON ss.image = scores.image_id
"""
data = pd.read_sql(query, conn)

# 假设数据已加载到DataFrame中
X = data.drop(columns=['score', 'image'])  # 除去评分列和图像ID列
y = data['score']  # 评分列

# 数据分割：将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用支持向量回归 (SVR) 进行训练
model = SVR(kernel='rbf', C=1.0, epsilon=0.2)
model.fit(X_train_scaled, y_train)

# 预测
y_pred = model.predict(X_test_scaled)

# 评估模型
mse = mean_squared_error(y_test, y_pred)  # 均方误差
r2 = r2_score(y_test, y_pred)  # R^2得分

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# 关闭数据库连接
conn.close()
