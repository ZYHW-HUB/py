import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pymysql
import os
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import psutil
from tqdm import tqdm

# --------------------------
# 初始化配置（与训练代码保持一致）
# --------------------------
image_dir = "walk/deeplabv3-master/training_logs/model_eval_seq_pp2"  # 修改为实际路径
db_config = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123",
    "database": "scene",
    "charset": "utf8mb4"
}

# --------------------------
# 数据加载模块（与训练代码保持一致）
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

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        # 此处可保持原样，因我们只需要路径列表
        pass  

def get_image_path(image_id):
    return os.path.join(image_dir, f"{image_id}_pred.png")

def load_pair_list(db_config):
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        query = "SELECT left_id, right_id, winner FROM pp2 WHERE category = 'beautiful'"  # 修改为对应的类别
        cursor.execute(query)
        return [(p[0], p[1], p[2]) for p in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

# --------------------------
# 模型定义（必须与训练时完全一致）
# --------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward_once(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.conv(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))

# --------------------------
# 核心评分函数（修复版本）
# --------------------------
class PathDataset(Dataset):  # 移出函数成为全局类
    def __init__(self, paths):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("L")
            return self.transform(img), self.paths[idx]
        except Exception as e:
            print(f"无法加载图像: {self.paths[idx]}，错误: {str(e)}")
            return torch.zeros(1, 128, 128), self.paths[idx]  # 返回占位数据

def score_all_images(model, dataset, device, batch_size=32):
    # 初始化
    path_dataset = PathDataset(dataset.all_image_paths)
    
    # 修改DataLoader参数
    loader = DataLoader(
        path_dataset,
        batch_size=batch_size,
        num_workers=0 if os.name == 'nt' else 4,  # Windows系统设为0
        pin_memory=True,
        persistent_workers=False  # 禁用持久化工作进程
    )
    scores = {}

    # 批量评分
    model.eval()
    with torch.no_grad():
        for batch, paths in tqdm(loader, desc="评分进度"):
            batch = batch.to(device)
            outputs = model.forward_once(batch)
            
            for path, score in zip(paths, outputs.cpu().numpy().flatten()):
                image_id = os.path.basename(path).split("_")[0]
                scores[image_id] = float(score)

    return scores

# --------------------------
# 数据库操作
# --------------------------
def create_scores_table(db_config):
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_scores_beautiful_S (
                image_id VARCHAR(255),
                score FLOAT,
                split VARCHAR(10),
                PRIMARY KEY (image_id, split)
            )
        """)
        conn.commit()
    finally:
        cursor.close()
        conn.close()

def save_scores_to_db(db_config, scores, split='all'):
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        query = """
            INSERT INTO image_scores_beautiful_S 
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE score=VALUES(score)
        """
        items = [(k, v, split) for k, v in scores.items()]
        
        # 分批提交
        batch_size = 500
        for i in tqdm(range(0, len(items), batch_size), desc="入库进度"):
            cursor.executemany(query, items[i:i+batch_size])
            conn.commit()
            
    except Exception as e:
        print(f"数据库操作失败: {str(e)}")
    finally:
        cursor.close()
        conn.close()

# --------------------------
# 主流程
# --------------------------
if __name__ == '__main__':
    # 初始化配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "walk/deeplabv3-master/training_logs/siamese_beautiful/saved_models/best_model.pth"  # 修改为实际路径
    
    # 加载数据
    pair_list = load_pair_list(db_config)
    dataset = SegmentationPairDataset(pair_list)
    
    # 加载模型
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 创建评分表
    create_scores_table(db_config)
    
    # 执行评分
    scores = score_all_images(model, dataset, device)
    
    # 保存结果
    save_scores_to_db(db_config, scores)
    
    print(f"成功入库 {len(scores)} 条评分记录")