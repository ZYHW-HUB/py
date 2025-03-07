import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pymysql
import os
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import psutil

# --------------------------
# 图像路径序列
# --------------------------
class SegmentationPairDataset(Dataset):
    def __init__(self, pair_list, transform=None):
        self.pair_list = pair_list
        self.transform = transform
        self.cache = OrderedDict()
        self.max_cache_size = 1600  # 根据内存容量调整
        # 新增：收集所有唯一图像路径
        self.all_image_paths = list(OrderedDict.fromkeys(
            [get_image_path(p[0]) for p in pair_list] + 
            [get_image_path(p[1]) for p in pair_list]
        ))

# 定义您的 Siamese 网络（只用其中一个分支作为评分器）
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # 这里与训练时的架构保持一致，注意输入通道为 1（灰度图）
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self._init_weights()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward_once(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.conv(x)
        x = self.avgpool(x)
        x = self.fc(x.flatten(1))
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)
# --------------------------
# 自定义转换：这里不再转换为类别标签，而是直接转换为灰度图的 Tensor
# --------------------------
class ConvertToGray(object):
    def __call__(self, image):
        # image: PIL Image
        # 转换为灰度图（单通道）
        image = image.convert("L")
        return transforms.ToTensor()(image)
# 加载模型权重（请将路径替换为您的实际模型文件路径）
def load_trained_model(model_path, device='cpu'):
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 定义图像预处理（与训练时保持一致，输入为灰度图像）
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()  # 对于灰度图，将生成单通道 tensor
])

# 定义评分函数：调用模型的 forward_once 得到单张图像的评分
def score_image(model, image_path, device='cpu'):
    image = Image.open(image_path).convert("L")  # 转为灰度图
    image_tensor = preprocess(image).unsqueeze(0)  # 增加 batch 维度
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        score = model.forward_once(image_tensor)
    return score.item()

# 数据库配置
db_config = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123",
    "database": "scene",
    "charset": "utf8mb4"
}

def load_pair_list(db_config):
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        query = "SELECT left_id, right_id, winner FROM pp2 WHERE category = 'safety'"
        cursor.execute(query)
        return [(p[0], p[1], p[2]) for p in cursor.fetchall()]
    except pymysql.MySQLError as e:
        print(f"数据库错误: {e}")
        return []
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
def insert_scores(db_config, scores, split, conn, cursor, batch_size=1000):
    try:
        query = "INSERT INTO image_scores_safety_S VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE score=VALUES(score)"
        items = [(str(k), v, split) for k, v in scores.items()]
        for i in range(0, len(items), batch_size):
            cursor.executemany(query, items[i:i+batch_size])
            conn.commit()
    except pymysql.MySQLError as e:
        print(f"数据库错误: {e.args}")

# 原始图像路径获取函数（用于非预处理模式）
def get_image_path(image_id):
    return os.path.join(image_dir, f"{image_id}_pred.png")

def hardware_config():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 根据 GPU/CPU 内存动态计算批次大小
    if device.type == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch_size = min(64, int(total_mem * 0.7 // 0.25))
    else:
        avail_mem = psutil.virtual_memory().available / 1024**3
        batch_size = min(32, int(avail_mem * 0.6 // 0.15))
    
    grad_accum_steps = 2 if batch_size < 16 else 1
    
    return (
        device,
        max(batch_size, 8),  # 保证最小批次大小
        min(4, os.cpu_count() // 2),  # 工作进程数
        grad_accum_steps
    )
    
# 初始化硬件配置
device, batch_size, num_workers, grad_accum = hardware_config()
            
if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据加载
    pair_list = load_pair_list(db_config)
    # 模型权重路径（请根据实际情况修改）
    model_path = "walk/deeplabv3-master/training_logs/siamese6/saved_models/best_model.pth"
    model = load_trained_model(model_path, device=device)
    dataset = SegmentationPairDataset(pair_list, preprocess)
    all_paths = dataset.all_image_paths  # 获取所有唯一路径列表
    image_dir = "walk/deeplabv3-master/training_logs/model_eval_seq"
    # 测试图像路径（请根据实际情况修改）

    test_image_path = "walk/deeplabv3-master/training_logs/model_eval_seq/some_test_image_pred.png"
    
    score = score_image(model, test_image_path, device=device)
    print(f"图像评分: {score:.4f}")
