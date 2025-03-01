import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

# 连接到MySQL数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',  # 这里使用你的数据库名称
    charset='utf8mb4'
)

# 从数据库中提取数据
query = """
    SELECT ss.*, scores.score 
    FROM pp2_ss ss
    JOIN image_scores_safety scores ON ss.image = scores.image_id
"""
merged_df = pd.read_sql(query, conn)

# 关闭数据库连接
conn.close()

# 数据准备
X = merged_df.drop(columns=['score','image'])  # 所有景物比例特征
y = merged_df['score']  # 评分

# 数据分割：将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建多项式特征的转换器，选择多项式的度数
poly_degree = 3  # 可以调整为不同的度数
poly = PolynomialFeatures(degree=poly_degree)

# 转换训练集和测试集的特征为多项式特征
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 使用线性回归模型进行拟合
model = LinearRegression()

# 在训练中添加 tqdm 显示进度条
for _ in tqdm(range(1), desc="Training Progress"):
    model.fit(X_train_poly, y_train)

# 预测并评估模型
y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# 可视化预测结果
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(f"Polynomial Regression (Degree {poly_degree})")
plt.show()
