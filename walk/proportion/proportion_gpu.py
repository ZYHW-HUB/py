import torch
import glob
import cv2
import os
from pathlib import Path
import csv
import pymysql

# 检查点文件
checkpoint_file = 'py/walk/proportion/proportion_checkpoint.txt'

def load_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return int(f.read().strip())
    return 0

def save_checkpoint(index):
    with open(checkpoint_file, 'w') as f:
        f.write(str(index))
if __name__ == "__main__":
    # 打印检查点文件的位置
    print(f"Checkpoint file location: {os.path.abspath(checkpoint_file)}")

    # 检查检查点文件是否存在
    if os.path.exists(checkpoint_file):
        print("Checkpoint file exists.")
    else:
        print("Checkpoint file does not exist.")

# # 加载检查点
# current_image_index = load_checkpoint()
            
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

def count_categories(segmentation_image_tensor, color_map_tensor):
    category_counts = torch.zeros(len(color_map), dtype=torch.int64, device=segmentation_image_tensor.device)
    
    for idx, color in enumerate(color_map_tensor):
        mask = torch.all(segmentation_image_tensor == color, dim=-1)
        category_counts[idx] = mask.sum()

    # 确保所有类别都在字典中，即使它们的计数为 0
    return category_counts

def calculate_category_percentages(category_counts, total_pixels):
    percentages = (category_counts.float() / total_pixels) * 100
    return percentages

os.environ['IMAGES_DATASET'] = 'D:/py/walk/deeplabv3-master/training_logs/model_eval_seq'
picturepath = os.environ['IMAGES_DATASET']
searchimage = os.path.join(picturepath, '*_pred.png')

# search files
image_paths = glob.glob(searchimage)
image_paths.sort()

with open("walk/proportion/pp2_ss.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(("image","Road", "Sidewalk", "Building", "Wall",
                     "Fence", "Pole", "Traffic_Light", "Traffic_Sign", 
                     "Vegetation", "Terrain", "Sky", "Person", "Rider", 
                     "Car", "Truck", "Bus", "Train", "Motorcycle", "Bicycle"
                     , "Other"))

# Load checkpoint
current_image_index = load_checkpoint()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert color map to tensor
color_map_tensor = torch.tensor(list(color_map.values()), dtype=torch.uint8).to(device)

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
    if current_image_index >= i + 1:
        continue
    # 创建Path对象
    path_object = Path(image_paths[i])

    # 获取文件名
    image_name = path_object.name[:-9]
    print(image_name)

    # 读取图像并转换为Tensor
    segmentation_image = cv2.imread(image_paths[i])
    segmentation_image_tensor = torch.tensor(segmentation_image, dtype=torch.uint8).to(device)
    
    # 统计每个类别的像素数量
    category_counts = count_categories(segmentation_image_tensor, color_map_tensor)
    
    # 计算总面积
    total_pixels = segmentation_image_tensor.shape[0] * segmentation_image_tensor.shape[1]
    
    # 计算类别占比
    category_percentages = calculate_category_percentages(category_counts, total_pixels)
    
    with open("walk/proportion/pp2_ss.csv", "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow((image_name, *[category_percentages[j].item() for j in range(len(category_percentages))]))
    
    sql = 'insert into pp2_ss(image,Road, Sidewalk, Building, Wall,Fence, Pole, Traffic_Light, Traffic_Sign, Vegetation, Terrain, Sky, Person, Rider, Car, Truck, Bus, Train, Motorcycle, Bicycle, Other) values("{}",{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})'.format(
                        image_name, *[category_percentages[j].item() for j in range(len(category_percentages))])
    
    cur.execute(sql)
    conn.commit()
    current_image_index += 1
    save_checkpoint(current_image_index)

cur.close()
conn.close()