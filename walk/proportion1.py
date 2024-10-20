import cv2
import numpy as np
from collections import defaultdict
import os, glob, sys
import csv
from pathlib import Path
import pymysql
# 定义类别颜色映射[b,g,r]
color_map = {
    0: [128, 64, 128],       # 路面
    1: [244, 35, 232],       # 人行道
    2: [70, 70, 70],         # 建筑物
    3: [102, 102, 156],      # 墙壁
    4: [190, 153, 153],      # 栅栏
    5: [153, 153, 153],      # 桩
    6: [250, 170, 30],       # 交通灯
    7: [220, 220, 0],        # 交通标志
    8: [107, 142, 35],       # 植被
    9: [152, 251, 152],      # 地形
    10: [70, 130, 180],      # 天空
    11: [220, 20, 60],       # 人
    12: [255, 0, 0],         # 骑行者
    13: [0, 0, 142],         # 汽车
    14: [0, 0, 70],          # 卡车
    15: [0, 60, 100],        # 巴士
    16: [0, 80, 100],        # 火车
    17: [0, 0, 230],         # 摩托车
    18: [119, 11, 32],       # 自行车
    19: [81, 0, 81]          # 其他
}

def count_categories(segmentation_image):
    # 初始化类别计数字典
    category_counts = defaultdict(int)
    
    # 将图像转换为 NumPy 数组
    segmentation_image = np.array(segmentation_image)
    
    # 遍历图像中的每个像素
    for row in range(segmentation_image.shape[0]):
        for col in range(segmentation_image.shape[1]):
            pixel = segmentation_image[row, col]
            for class_id, color in color_map.items():
                if np.array_equal(pixel, color):
                    category_counts[class_id] += 1
    # 确保所有类别都在字典中，即使它们的计数为 0
    for class_id in color_map.keys():
        if class_id not in category_counts:
            category_counts[class_id] = 0
    
    return category_counts

def calculate_category_percentages(category_counts, total_pixels):
    percentages = {}
    for category, count in category_counts.items():
        percentages[category] = (count / total_pixels) * 100
    return percentages

def process_segmentation_images(image_paths):
    results = []
    
    for path in image_paths:
        # 读取图像
        segmentation_image = cv2.imread(path)
        
        # 统计每个类别的像素数量
        category_counts = count_categories(segmentation_image)
        
        # 计算总面积
        total_pixels = segmentation_image.shape[0] * segmentation_image.shape[1]
        
        # 计算类别占比
        category_percentages = calculate_category_percentages(category_counts, total_pixels)
        
        # 存储结果
        results.append({
            'path': path,
            'category_counts': category_counts,
            'category_percentages': category_percentages
        })
    
    return results

os.environ['IMAGES_DATASET'] ='D:/py/walk/deeplabv3-master/training_logs/model_eval_seq'
picturepath = os.environ['IMAGES_DATASET'] 
# 示例图片路径列表
searchimage = os.path.join(picturepath ,'*_pred.png')
# image_paths = [
#     'walk/deeplabv3-master/training_logs/model_eval_seq/berlin_000000_000019_pred.png',
# ]

# search files
image_paths = glob.glob(searchimage)
image_paths.sort()
# category_names = ["Road", "Sidewalk", "Building", "Wall", "Fence", "Pole", "Traffic Light", "Traffic Sign", "Vegetation", "Terrain", "Sky", "Person", "Rider", "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle", "Other"]
print(image_paths)
with open("walk/pp2_ss.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(("image","Road", "Sidewalk", "Building", "Wall",
                     "Fence", "Pole", "Traffic_Light", "Traffic_Sign", 
                     "Vegetation", "Terrain", "Sky", "Person", "Rider", 
                     "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle"
                     , "Other"))
# 处理分割好的图片
# results = process_segmentation_images(image_paths)
conn = pymysql.Connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='123',
    db='scene',
    charset='utf8'
)
cur = conn.cursor()
print(len(image_paths))
for i in range(len(image_paths)):
    print(i)
    # 创建Path对象
    path_object = Path(image_paths[i])

    # 获取文件名
    image_name = path_object.name[:-9]
    print(image_name)
    # 读取图像
    segmentation_image = cv2.imread(image_paths[i])
    
    # 统计每个类别的像素数量
    category_counts = count_categories(segmentation_image)
    
    # 计算总面积
    total_pixels = segmentation_image.shape[0] * segmentation_image.shape[1]
    
    # 计算类别占比
    category_percentages = calculate_category_percentages(category_counts, total_pixels)
    # print(category_counts[8])
    with open("walk/pp2_ss.csv", "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow((image_name,category_percentages[0], category_percentages[1], category_percentages[2], category_percentages[3],
                        category_percentages[4], category_percentages[5], category_percentages[6], category_percentages[7],
                        category_percentages[8], category_percentages[9], category_percentages[10], category_percentages[11],
                        category_percentages[12], category_percentages[13], category_percentages[14], category_percentages[15],
                        category_percentages[16], category_percentages[17], category_percentages[18], category_percentages[19]))
    sql = 'insert into pp2_ss(image,Road, Sidewalk, Building, Wall,Fence, Pole, Traffic_Light, Traffic_Sign, Vegetation, Terrain, Sky, Person, Rider, Car, Truck, Bus, Train, Motorcycle, Bicycle, Other) values("{}",{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})'.format(
                        image_name,category_percentages[0], category_percentages[1], category_percentages[2], category_percentages[3],
                        category_percentages[4], category_percentages[5], category_percentages[6], category_percentages[7],
                        category_percentages[8], category_percentages[9], category_percentages[10], category_percentages[11],
                        category_percentages[12], category_percentages[13], category_percentages[14], category_percentages[15],
                        category_percentages[16], category_percentages[17], category_percentages[18], category_percentages[19])
    print(sql)
    # 执行
    cur.execute(sql)
    # 提交
    conn.commit()
# 数据库连接中断
cur.close()
conn.close()
# # 输出结果
# for result in results:
#     from pathlib import Path

#     # 定义文件路径
#     file_path = '/py/walk/deeplabv3-master/training_logs/model_eval_seq\\berlin_000000_000019_pred.png'

#     # 创建Path对象
#     path_object = Path(file_path)

#     # 获取文件名
#     file_name = path_object.name

#     print("文件名:", file_name)
#     print(result)
#     print(f"Image Path: {result['path']}")
#     print(f"Category Counts: {result['category_counts']}")
#     print(f"Category Percentages: {result['category_percentages']}\n")

         
#     with open("walk/pp2_ss.csv", "w"|"a"|"r+", newline="", encoding="utf-8-sig") as f: