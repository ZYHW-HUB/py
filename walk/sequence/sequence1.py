import pymysql
import pandas as pd

# 连接到MySQL数据库
conn = pymysql.Connect(
    host='localhost',  # 数据库主机地址
    port=3306,         # 数据库端口
    user='root',       # 数据库用户名
    passwd='123',      # 数据库密码
    db='scene',        # 数据库名称
    charset='utf8'     # 数据库字符集
)

# 创建游标对象，用于执行SQL查询
cursor = conn.cursor()

# 查询数据库中的景物比例数据
query = """
SELECT image, Road, Sidewalk, Building, Wall, Fence, Pole, Traffic_Light, Traffic_Sign, 
       Vegetation, Terrain, Sky, Person, Rider, Car, Truck, Bus, Train, Motorcycle, Bicycle, Other
FROM pp2_ss;
"""

# 执行查询
cursor.execute(query)

# 获取所有查询结果
data = cursor.fetchall()

# 关闭游标和连接
cursor.close()
conn.close()

# 将查询结果转换为DataFrame
df = pd.DataFrame(data, columns=["image", "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", 
                                 "Traffic_Light", "Traffic_Sign", "Vegetation", "Terrain", "Sky", 
                                 "Person", "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle", 
                                 "Bicycle", "Other"])

# 查看前几行数据
print(df.head())
