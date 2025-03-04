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

# 初始化一个字典来统计胜负和平局
win_count = {}
lose_count = {}
draw_count = {}
total_count = {}

# 从 pp2_combined 中获取关于胜负的数据
comparisons = pd.read_sql("SELECT left_id, right_id, winner_label_encoded FROM pp2_combined WHERE category = 'safety'", con=conn)

# 遍历每条记录，统计胜负、平局和总比较次数
for _, row in comparisons.iterrows():
    left_id = row['left_id']
    right_id = row['right_id']
    result = row['winner_label_encoded']
    
    # 更新总比较次数
    total_count[left_id] = total_count.get(left_id, 0) + 1
    total_count[right_id] = total_count.get(right_id, 0) + 1
    
    # 统计左图胜负和平局
    if left_id not in win_count:
        win_count[left_id] = 0
        lose_count[left_id] = 0
        draw_count[left_id] = 0
    if result == 1:
        win_count[left_id] += 1
    elif result == 2:
        lose_count[left_id] += 1
    elif result == 0:
        draw_count[left_id] += 1

    # 统计右图胜负和平局
    if right_id not in win_count:
        win_count[right_id] = 0
        lose_count[right_id] = 0
        draw_count[right_id] = 0
    if result == 2:
        win_count[right_id] += 1
    elif result == 1:
        lose_count[right_id] += 1
    elif result == 0:
        draw_count[right_id] += 1

# 计算胜率和负率
win_rate = {img_id: win_count[img_id] / total_count[img_id] for img_id in win_count}
lose_rate = {img_id: lose_count[img_id] / total_count[img_id] for img_id in lose_count}

# 创建新表 image_stats_safety
with conn.cursor() as cursor:
    cursor.execute('''CREATE TABLE IF NOT EXISTS image_stats_safety (
        image_id VARCHAR(255) NOT NULL,
        win_count INT NOT NULL,
        lose_count INT NOT NULL,
        draw_count INT NOT NULL,
        total_count INT NOT NULL,
        win_rate FLOAT NOT NULL,
        lose_rate FLOAT NOT NULL,
        PRIMARY KEY (image_id)
    )''')

# 插入结果到数据库中
with conn.cursor() as cursor:
    for img_id in win_count.keys():
        cursor.execute(
            '''INSERT INTO image_stats_safety (image_id, win_count, lose_count, draw_count, total_count, win_rate, lose_rate)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON DUPLICATE KEY UPDATE
               win_count = VALUES(win_count),
               lose_count = VALUES(lose_count),
               draw_count = VALUES(draw_count),
               total_count = VALUES(total_count),
               win_rate = VALUES(win_rate),
               lose_rate = VALUES(lose_rate)''',
            (img_id, win_count[img_id], lose_count[img_id], draw_count[img_id], total_count[img_id], win_rate[img_id], lose_rate[img_id])
        )
    conn.commit()

# 关闭连接
conn.close()