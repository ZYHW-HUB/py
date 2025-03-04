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

# 从数据库中获取参与过比较的图像 id
data = pd.read_sql("SELECT DISTINCT left_id AS image_id FROM pp2_combined WHERE category = 'safety' "
                   "UNION "
                   "SELECT DISTINCT right_id AS image_id FROM pp2_combined WHERE category = 'safety'", con=conn)

# 获取景物比例数据，仅从这些参与过比较的图像中提取数据
image_ids = tuple(data['image_id'].tolist())  # 获取所有参与比较的图像ID

# 初始化一个字典来统计胜负
win_count = {}
lose_count = {}
total_count = {}

# 从 pp2_combined 中获取关于胜负的数据
comparisons = pd.read_sql("SELECT left_id, right_id, winner_label_encoded FROM pp2_combined WHERE category = 'safety'", con=conn)

# 遍历每条记录，统计胜负和总比较次数
for _, row in comparisons.iterrows():
    left_id = row['left_id']
    right_id = row['right_id']
    result = row['winner_label_encoded']
    
    # 更新总比较次数
    total_count[left_id] = total_count.get(left_id, 0) + 1
    total_count[right_id] = total_count.get(right_id, 0) + 1
    
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

# 计算胜率和负率
win_rate = {img_id: win_count[img_id] / total_count[img_id] for img_id in win_count}
lose_rate = {img_id: lose_count[img_id] / total_count[img_id] for img_id in lose_count}

# 找到胜率最高的图片
max_win_rate_image = max(win_rate, key=win_rate.get)
max_win_rate = win_rate[max_win_rate_image]

# 找到负率最高的图片（没有胜场的图片）
no_win_images = [img_id for img_id in lose_rate if img_id not in win_rate or win_count[img_id] == 0]
max_lose_rate_image = max(no_win_images, key=lose_rate.get) if no_win_images else None
max_lose_rate = lose_rate[max_lose_rate_image] if max_lose_rate_image else None

# 为每张图片计算评分
scores = {}
for img_id in win_count.keys():
    if win_count[img_id] > 0:
        score = ((win_rate[img_id] + max_lose_rate) / (max_win_rate + max_lose_rate)) * 100
    else:
        score = (max_lose_rate - lose_rate[img_id] / (max_win_rate + max_lose_rate)) * 100 if max_lose_rate else 0
    scores[img_id] = score

# 找到比较场数最多的和最少的图片
max_total_count_image = max(total_count, key=total_count.get)
min_total_count_image = min(total_count, key=total_count.get)

print(f"Score 100 (Highest win rate): {max_win_rate_image} with win rate {max_win_rate}")
print(f"Score 0 (Lowest lose rate among no win images): {max_lose_rate_image} with lose rate {max_lose_rate}" if max_lose_rate_image else "No images with no wins")
print(f"Maximum comparisons: {max_total_count_image} with {total_count[max_total_count_image]} comparisons")
print(f"Minimum comparisons: {min_total_count_image} with {total_count[min_total_count_image]} comparisons")

# 创建新表 image_scores
with conn.cursor() as cursor:
    cursor.execute('''CREATE TABLE IF NOT EXISTS image_scores_safety (
        image_id VARCHAR(255) NOT NULL,
        score FLOAT NOT NULL,
        PRIMARY KEY (image_id)
    )''')

# 插入结果到数据库中
with conn.cursor() as cursor:
    for img_id, score in scores.items():
        cursor.execute(
            'INSERT INTO image_scores_safety (image_id, score) VALUES (%s, %s) ON DUPLICATE KEY UPDATE score = %s',
            (img_id, score, score)
        )
    conn.commit()

# 关闭连接
conn.close()