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

# 假设数据已加载到DataFrame中
data = pd.read_sql("SELECT left_id, right_id, winner_label_encoded FROM pp2_combined WHERE category = 'lively'", con=conn)

# 初始化一个字典来统计胜负
win_count = {}
lose_count = {}

# 遍历每条记录，统计胜负
for _, row in data.iterrows():
    left_id = row['left_id']
    right_id = row['right_id']
    result = row['winner_label_encoded']
    
    # 统计左图胜负
    if left_id not in win_count:
        win_count[left_id] = 0
        lose_count[left_id] = 0
    if result == 1:
        win_count[left_id] += 1
    elif result == 2:
        lose_count[left_id] += 1

    # 统计右图胜负
    if right_id not in win_count:
        win_count[right_id] = 0
        lose_count[right_id] = 0
    if result == 2:
        win_count[right_id] += 1
    elif result == 1:
        lose_count[right_id] += 1

# 计算得分，将输最多的图片设为0，赢最多的图片设为100
min_wins = min(win_count.values())  # 输得最多
max_wins = max(win_count.values())  # 赢得最多

# 为每张图片计算评分
scores = {}
for img_id in win_count.keys():
    score = (win_count[img_id] - min_wins) / (max_wins - min_wins) * 100
    scores[img_id] = score

# 找到输得最多和赢得最多的图片
loser_image = min(lose_count, key=lose_count.get)
winner_image = max(win_count, key=win_count.get)

# 找到胜场数最多和最少的图片
max_wins_image = max(win_count, key=win_count.get)  # 胜场数最多的图片
min_wins_image = min(win_count, key=win_count.get)  # 胜场数最少的图片

print(f"Score 0 (Most losses): {loser_image}")
print(f"Score 100 (Most wins): {winner_image}")
print(f"Maximum wins (Most wins): {max_wins_image} with {win_count[max_wins_image]} wins")
print(f"Minimum wins (Fewest wins): {min_wins_image} with {win_count[min_wins_image]} wins")

# 创建新表 image_scores
with conn.cursor() as cursor:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_scores_lively (
            image_id VARCHAR(255) NOT NULL,
            score FLOAT NOT NULL,
            PRIMARY KEY (image_id)
        )
    ''')

# 插入结果到数据库中
with conn.cursor() as cursor:
    for img_id, score in scores.items():
        cursor.execute(
            'INSERT INTO image_scores_lively (image_id, score) VALUES (%s, %s) ON DUPLICATE KEY UPDATE score = %s',
            (img_id, score, score)
        )
    conn.commit()

# 关闭连接
conn.close()
