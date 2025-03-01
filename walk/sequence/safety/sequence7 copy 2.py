import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mysql.connector

# 从数据库获取数据
def fetch_data_from_db():
    # 数据库连接配置（根据实际情况修改）
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'port':'3306',
        'password': '123',
        'database': 'scene'  # 数据库名称
    }

    # 建立数据库连接
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # SQL 查询：获取特征和评分数据
    query = """
    SELECT ss.*, scores.score 
    FROM pp2_ss ss
    JOIN image_scores_safety scores ON ss.image = scores.image_id
    """
    cursor.execute(query)

    # 获取数据并转换为 DataFrame
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]  # 获取列名
    df = pd.DataFrame(rows, columns=columns)

    cursor.close()
    conn.close()

    return df

# 加载数据
df = fetch_data_from_db()

# 假设数据已经包含了 'score' 列作为目标值，其他列为特征
X = df.drop(columns=['score', 'image'])  # 假设 'image' 和 'image_id' 是不需要的列
y = df['score']

# 数据分割：将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建支持向量回归模型
svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)

# 训练模型
svr.fit(X_train, y_train)

# 预测
y_pred = svr.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# 可视化预测结果
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(f"Support Vector Regression")
plt.show()
