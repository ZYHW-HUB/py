import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pymysql
import os
from torch.utils.data import Dataset, DataLoader
import psutil
from tqdm import tqdm
import glob

# --------------------------
# 初始化配置
# --------------------------
image_dir = "walk/deeplabv3-master/training_logs/model_eval_seq"
db_config = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123",
    "database": "scene",
    "charset": "utf8mb4"
}

# --------------------------
# 模型定义（保持不变）
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
# 核心评分组件
# --------------------------
class PathDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.CenterCrop((512, 682)),  # 从768x512裁剪为682x512（左右各裁43像素）
            transforms.Resize((128, 128)),      # 缩放到模型输入尺寸
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            path = self.paths[idx]
            if not os.path.exists(path):
                raise FileNotFoundError(f"图片不存在: {path}")
                
            img = Image.open(path).convert("L")
            if img.size != (768, 512):
                print(f"警告：非常规尺寸 {path} - {img.size}")
            return self.transform(img), path
        except Exception as e:
            print(f"处理失败 [{path}]: {str(e)}")
            return torch.zeros(1, 128, 128), path

class ImagePathDataset:
    def __init__(self, image_paths):
        self.all_image_paths = image_paths

def score_all_images(model, dataset, device, batch_size=32):
    # 初始化
    path_dataset = PathDataset(dataset.all_image_paths)
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
                scores[os.path.basename(path).replace("_pred.png", "")] = float(score)  # 保留文件名作为标识
    return scores

# --------------------------
# 数据库操作
# --------------------------
def create_scores_table(db_config):
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_scores_YP (
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

def save_scores_to_db(db_config, scores, split='null'):
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        query = """
            INSERT INTO image_scores_YP 
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = "walk/deeplabv3-master/training_logs/siamese_boring/saved_models/best_model.pth"  # 修改为实际路径
    
    # 获取所有图片路径
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*_pred.png')))
    print(f"找到 {len(image_paths)} 张待评分图片")
    
    # 加载模型
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 创建数据集
    dataset = ImagePathDataset(image_paths)
    
    # 创建评分表
    create_scores_table(db_config)
    # 执行评分
    scores = score_all_images(model, dataset, device)
    
    # 保存结果
    save_scores_to_db(db_config, scores, split='boring')
    print(f"成功入库 {len(scores)} 条评分记录")