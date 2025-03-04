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
    charset='utf8mb4'  # 使用 utf8mb4，适应更广泛的字符编码
)

# 读取评分数据（假设已存在的表 image_scores）
image_scores_df = pd.read_sql("SELECT image_id, score FROM image_scores WHERE split = 'train'", con=conn)

# 将评分数据存储到字典中，方便快速查找
scores = dict(zip(image_scores_df['image_id'], image_scores_df['score']))

# 读取比赛数据
data = pd.read_sql("SELECT left_id, right_id, winner_label_encoded FROM pp2_combined WHERE category = 'safety'", con=conn)

# 初始化变量用于计算准确率
correct_predictions = 0
total_predictions = 0
misjudgments = defaultdict(int)  # 用于记录每种误判类型的次数

# 遍历每条记录，进行预测
for _, row in data.iterrows():
    left_id = row['left_id']
    right_id = row['right_id']
    result = row['winner_label_encoded']
    
    # 获取图片评分，如果没有评分则默认给定 0 分
    left_score = scores.get(left_id, 0)
    right_score = scores.get(right_id, 0)
    
    # 根据评分进行预测
    if left_score > right_score:
        predicted_result = 1  # 左图胜
    elif right_score > left_score:
        predicted_result = 2  # 右图胜
    else:
        predicted_result = 0  # 平局

    # 比较预测结果与实际结果
    if predicted_result == result:
        correct_predictions += 1
    else:
        # 统计误判类型
        if result == 1 and predicted_result == 2:
            misjudgments["Left win predicted as Right win"] += 1
        elif result == 2 and predicted_result == 1:
            misjudgments["Right win predicted as Left win"] += 1
        elif result == 0 and predicted_result == 1:
            misjudgments["Draw predicted as Left win"] += 1
        elif result == 0 and predicted_result == 2:
            misjudgments["Draw predicted as Right win"] += 1
        elif result == 1 and predicted_result == 0:
            misjudgments["Left win predicted as Draw"] += 1
        elif result == 2 and predicted_result == 0:
            misjudgments["Right win predicted as Draw"] += 1
    
    total_predictions += 1  # 统计总判断数

# 计算准确率
accuracy = correct_predictions / total_predictions * 100

# 输出总判断数、准确率和误判统计
print(f"Total Judgments: {total_predictions}")
print(f"Prediction Accuracy: {accuracy:.2f}%")

# 输出误判类型统计
print("\nMisjudgment Statistics:")
for misjudgment, count in misjudgments.items():
    print(f"{misjudgment}: {count} times")

# 关闭连接
conn.close()
