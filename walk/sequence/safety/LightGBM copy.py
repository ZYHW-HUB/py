import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pymysql

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

# 创建 LightGBM 的回归模型
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 设置 LightGBM 的参数
params = {
    'objective': 'regression',  # 回归任务
    'metric': 'l2',  # 均方误差
    'boosting_type': 'gbdt',  # 使用梯度提升树
    'num_leaves': 31,  # 树的最大叶子节点数
    'learning_rate': 0.05,  # 学习率
    'feature_fraction': 0.9,  # 用于训练的特征比例
    'device': 'gpu',  # 启用 GPU 加速
    'verbose': 1  # 输出训练进度
}

# 定义早停回调函数
early_stopping = lgb.early_stopping(stopping_rounds=10, verbose=True)

# 训练模型，并使用早停回调
num_round = 100  # 迭代次数
callbacks = [early_stopping]
model = lgb.train(params, train_data, num_round, valid_sets=[test_data], callbacks=callbacks)

# 预测
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# 评估模型
mse = mean_squared_error(y_test, y_pred)  # 均方误差
r2 = r2_score(y_test, y_pred)  # R^2得分

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
