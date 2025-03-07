import pymysql
import pandas as pd
from collections import defaultdict

# 连接到MySQL数据库
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123',
    database='scene',
    charset='utf8mb4'
)

# 读取评分数据
image_scores_beautiful_df = pd.read_sql(
    "SELECT image_id, score FROM image_scores_beautiful_s", 
    con=conn
)

# 将评分数据存储到字典中
scores = dict(zip(image_scores_beautiful_df['image_id'], 
                 image_scores_beautiful_df['score']))

# 读取比赛数据并过滤平局样本
data = pd.read_sql("""
    SELECT left_id, right_id, winner_label_encoded 
    FROM pp2_combined 
    WHERE category = 'beautiful' 
      AND winner_label_encoded != 0  -- 排除平局样本
    """, con=conn)

# 初始化统计变量
correct_predictions = 0
total_predictions = 0
misjudgments = defaultdict(int)

for _, row in data.iterrows():
    left_id = row['left_id']
    right_id = row['right_id']
    result = row['winner_label_encoded']  # 1=左胜，2=右胜
    
    # 获取评分（缺失值处理为0）
    left_score = scores.get(left_id, 0)
    right_score = scores.get(right_id, 0)
    
    # 预测结果（只考虑胜负）
    predicted_result = 1 if left_score > right_score else 2

    # 统计结果
    if predicted_result == result:
        correct_predictions += 1
    else:
        # 记录误判类型
        if result == 1 and predicted_result == 2:
            misjudgments["实际左胜预测为右胜"] += 1
        elif result == 2 and predicted_result == 1:
            misjudgments["实际右胜预测为左胜"] += 1
    
    total_predictions += 1

# 计算并输出结果
if total_predictions > 0:
    accuracy = correct_predictions / total_predictions * 100
else:
    accuracy = 0.0

print(f"有效对决数量（排除平局）: {total_predictions}")
print(f"预测准确率: {accuracy:.2f}%")
print("\n误判类型分布:")
for desc, count in misjudgments.items():
    print(f"{desc}: {count}次")

conn.close()