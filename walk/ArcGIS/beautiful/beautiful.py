import pymysql
import csv

# 数据库配置
db_config = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123",
    "database": "scene",
    "charset": "utf8mb4"
}

def extract_coordinates(image_id):
    """
    从 image_id 中提取经纬度信息，
    示例：DinghaigangRoad_121.5508308_31.27726457_270_0
    返回 (121.5508308, 31.27726457)；如果格式不符合则返回空字符串
    """
    parts = image_id.split('_')
    if len(parts) >= 3:
        try:
            # 第二部分为经度，第三部分为纬度
            longitude = float(parts[1])
            latitude = float(parts[2])
            return longitude, latitude
        except ValueError:
            return '', ''
    return '', ''

def main():
    # 建立数据库连接
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()
    
    # 查询数据
    sql = "SELECT image_id, score FROM image_scores_yp WHERE split = 'beautiful'"
    cursor.execute(sql)
    rows = cursor.fetchall()
    
    # 打开CSV文件，写入表头及数据
    with open('walk/ArcGIS/beautiful/image_scores_beautiful.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 表头：image_id, score, longitude, latitude
        writer.writerow(['image_id', 'score', 'longitude', 'latitude'])
        for row in rows:
            image_id, score = row
            longitude, latitude = extract_coordinates(image_id)
            writer.writerow([image_id, score, longitude, latitude])
    
    # 关闭游标和连接
    cursor.close()
    conn.close()
    
if __name__ == "__main__":
    main()