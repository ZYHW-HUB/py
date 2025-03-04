import xgboost as xgb
import pymysql
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 连接到数据库并获取数据
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'
)

query = """
SELECT ss.*, scores.score 
FROM pp2_ss ss
JOIN image_scores_safety scores ON ss.image = scores.image_id
"""
merged_df = pd.read_sql(query, conn)

# 假设'X'是特征，'y'是评分（目标值）
X = merged_df.drop(columns=['score','image'])
y = merged_df['score']

# 数据分割：将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 XGBoost 的回归模型
model = xgb.XGBRegressor(objective='reg:squarederror', 
                         colsample_bytree=0.3, 
                         learning_rate=0.1, 
                         max_depth=2, 
                         alpha=10, 
                         n_estimators=10)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)  # 均方误差
r2 = r2_score(y_test, y_pred)  # R^2得分

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
