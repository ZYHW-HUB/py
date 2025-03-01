import pymysql
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

# 连接到MySQL数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'
)

# 从数据库中获取数据
query = """
SELECT ss.*, scores.score
FROM pp2_ss ss
JOIN image_scores_safety scores ON ss.image = scores.image_id
"""
data = pd.read_sql(query, conn)

# 关闭数据库连接
conn.close()

# 假设数据已加载到DataFrame中
X = data.drop(columns=['score', 'image'])  # 除去评分列和图像ID列
y = data['score']  # 评分列

# 数据分割：将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为 PyTorch tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 创建一个简单的线性回归模型（类似于SVM）
class SVMModel(nn.Module):
    def __init__(self):
        super(SVMModel, self).__init__()
        self.fc = nn.Linear(X_train_scaled.shape[1], 1)  # 线性回归层

    def forward(self, x):
        return self.fc(x)

# 创建模型实例
model = SVMModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 50
for epoch in tqdm(range(epochs)):
    model.train()

    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 训练过程中的损失
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 预测
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)

# 评估模型
y_pred = y_pred.numpy()
mse = mean_squared_error(y_test, y_pred)  # 均方误差
r2 = r2_score(y_test, y_pred)  # R^2得分

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
