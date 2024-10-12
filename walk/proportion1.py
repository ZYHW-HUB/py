import cv2
import numpy as np
from collections import defaultdict
import os, glob, sys

# 定义类别颜色映射
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
print(image_paths)

# # 处理分割好的图片
# results = process_segmentation_images(image_paths)

# # 输出结果
# for result in results:
#     print(f"Image Path: {result['path']}")
#     print(f"Category Counts: {result['category_counts']}")
#     print(f"Category Percentages: {result['category_percentages']}\n")