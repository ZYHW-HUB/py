import os
import pymysql
import psutil
from collections import OrderedDict
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
import json
from torch.cuda.amp import autocast, GradScaler
import multiprocessing
import time
from datetime import timedelta
import csv
from datetime import datetime
from tqdm import tqdm  # 新增

# --------------------------
# 1. 数据库配置与数据读取
# --------------------------
db_config = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123",
    "database": "scene",
    "charset": "utf8mb4"
}

# 图像目录
image_dir = "walk/deeplabv3-master/training_logs/model_eval_seq"

# 数据加载函数
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
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

# 图像路径获取函数
def get_image_path(image_id):
    return os.path.join(image_dir, f"{image_id}_pred.png")

# 创建评分表
def create_scores_table(db_config):
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_scores_safety_S (
                image_id VARCHAR(255),
                score FLOAT,
                split VARCHAR(10),
                PRIMARY KEY (image_id, split)
            )
        """)
        conn.commit()
    except pymysql.MySQLError as e:
        print(f"数据库错误: {e}")
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

# 插入评分到数据库
def insert_scores(db_config, scores, split, conn, cursor, batch_size=1000):
    try:
        query = "INSERT INTO image_scores_safety_S VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE score=VALUES(score)"
        items = [(str(k), v, split) for k, v in scores.items()]
        for i in range(0, len(items), batch_size):
            cursor.executemany(query, items[i:i+batch_size])
            conn.commit()
    except pymysql.MySQLError as e:
        print(f"数据库错误: {e.args}")

# --------------------------
# 2. 智能缓存数据集
# --------------------------
class SegmentationPairDataset(Dataset):
    def __init__(self, pair_list, transform=None):
        self.pair_list = pair_list
        self.transform = transform
        # 标签映射: "left" 表示左边获胜，转换为 1；"right" 表示右边获胜，转换为 -1；"equal" 转换为 0
        self.label_map = {"left": 1, "right": -1, "equal": 0}
        self.cache = OrderedDict()
        self.max_cache_size = 800  # 基于内存容量调整

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        
        left_id, right_id, winner = self.pair_list[idx]
        left_path = get_image_path(left_id)
        right_path = get_image_path(right_id)

        if not all(os.path.exists(p) for p in [left_path, right_path]):
            raise FileNotFoundError(f"缺失文件: {left_path} 或 {right_path}")

        img_left = Image.open(left_path).convert("RGB")
        img_right = Image.open(right_path).convert("RGB")

        if self.transform:
            img_left = self.transform(img_left)
            img_right = self.transform(img_right)

        label = torch.tensor(self.label_map[winner], dtype=torch.float)
        
        if len(self.cache) >= self.max_cache_size:
            self.cache.popitem(last=False)
        self.cache[idx] = (img_left, img_right, label, left_id, right_id)
        
        return self.cache[idx]

# --------------------------
# 3. 内存优化模型
# --------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        
        self._init_weights()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward_once(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            x = self.conv(x)
            x = self.avgpool(x)
            return self.fc(x.flatten(1))

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# --------------------------
# 4. 动态损失计算
# --------------------------
def compute_loss(r1, r2, label, margin=0.5):
    loss = 0.0
    margin_loss = nn.MarginRankingLoss(margin=margin)
    mse_loss = nn.MSELoss()

    mask = label == 1
    if mask.any():
        loss += margin_loss(r1[mask], r2[mask], torch.ones_like(r1[mask]))
    
    mask = label == -1
    if mask.any():
        loss += margin_loss(r2[mask], r1[mask], torch.ones_like(r2[mask]))
    
    mask = label == 0
    if mask.any():
        loss += mse_loss(r1[mask], r2[mask])
    
    return loss

# --------------------------
# 5. 自适应硬件配置
# --------------------------
def hardware_config():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 动态内存计算
    if device.type == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch_size = min(64, int(total_mem * 0.7 // 0.25))
    else:
        avail_mem = psutil.virtual_memory().available / 1024**3
        batch_size = min(32, int(avail_mem * 0.6 // 0.15))
    
    grad_accum_steps = 2 if batch_size < 16 else 1
    
    return (
        device,
        max(batch_size, 8),  # 最小批次大小
        min(4, os.cpu_count()//2),  # 工作进程数
        grad_accum_steps
    )

# --------------------------
# 6. 训练流程
# --------------------------
if __name__ == '__main__':
    # 新增日志目录配置
    log_dir = "walk/deeplabv3-master/training_logs/siamese1"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d%H%M')}.csv")
    model_dir = os.path.join(log_dir, "saved_models")
    os.makedirs(model_dir, exist_ok=True)

    # 初始化最佳指标跟踪器
    best_metrics = {
        'epoch': 0,
        'val_loss': float('inf'),
        'val_acc': 0.0,
        'train_acc': 0.0
    }

    # 创建CSV日志文件头
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'val_loss', 
            'train_acc', 'val_acc', 'epoch_time',
            'model_path'
        ])

    multiprocessing.freeze_support()

    # 初始化配置
    device, batch_size, num_workers, grad_accum = hardware_config()
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # 数据加载
    pair_list = load_pair_list(db_config)
    dataset = SegmentationPairDataset(pair_list, transform)
    train_set, val_set = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator(device='cpu'),
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=1  # 降低预加载
    )

    val_loader = DataLoader(
        val_set,
        batch_size=max(batch_size//2, 8),
        shuffle=False,
        num_workers=max(1, num_workers//2),
        pin_memory=True
    )

    # 模型初始化
    model = SiameseNetwork().to(device, memory_format=torch.channels_last)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    model_save_path = "walk/deeplabv3-master/training_logs/siamese1/siamese_model.pth"
    scores_save_path = "walk/deeplabv3-master/training_logs/siamese1/scores.json"
    create_scores_table(db_config)

    # 数据库连接
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
    except pymysql.MySQLError as e:
        print(f"数据库连接失败: {e}")
        exit(1)

    # 训练循环
    total_start = time.time()
    for epoch in range(10):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        
        # 初始化训练统计
        running_loss = 0.0
        running_correct = 0  # 新增训练正确数统计
        total_samples = 0    # 新增训练样本总数统计
        train_scores = {}

        # 训练阶段
        for i, (img_l, img_r, labels, ids_l, ids_r) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):  # 使用tqdm显示进度
            # 数据传输优化
            img_l = img_l.to(device, non_blocking=True, memory_format=torch.channels_last)
            img_r = img_r.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            # 混合精度训练块
            with autocast(enabled=torch.cuda.is_available()):
                out1, out2 = model(img_l, img_r)
                loss = compute_loss(out1, out2, labels) / grad_accum

                # 新增准确率计算（训练阶段）
                with torch.no_grad():
                    preds = torch.where(out1 > out2, 1, -1)
                    # 处理平局标签（统一转换为1）
                    true_labels = labels.clone()
                    true_labels[true_labels == 0] = 1  # 使用clone避免修改原始标签
                    correct = (preds.flatten() == true_labels).sum().item()
                    running_correct += correct
                    total_samples += labels.size(0)

            # 反向传播
            if torch.cuda.is_available():
                scaler.scale(loss).backward()
                if (i+1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (i+1) % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

        # 计算训练准确率
        train_acc = running_correct / total_samples if total_samples > 0 else 0.0

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0  # 新增验证正确数统计
        val_total = 0    # 新增验证样本总数统计
        val_scores = {}
        
        with torch.inference_mode(), autocast(enabled=torch.cuda.is_available()):
            for img_l, img_r, labels, ids_l, ids_r in tqdm(val_loader, desc="Validation"):  # 使用tqdm显示进度
                img_l = img_l.to(device, non_blocking=True, memory_format=torch.channels_last)
                img_r = img_r.to(device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                
                out1, out2 = model(img_l, img_r)
                val_loss += compute_loss(out1, out2, labels).item()

                # 新增准确率计算（验证阶段）
                with torch.no_grad():
                    preds = torch.where(out1 > out2, 1, -1)
                    # 保持与训练阶段相同的标签处理方式
                    true_labels = labels.clone()
                    true_labels[true_labels == 0] = 1  # 与训练一致
                    val_correct += (preds.flatten() == true_labels).sum().item()
                    val_total += labels.size(0)

                # ... [分数记录代码不变] ...

        # 计算验证指标
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        # 保存每个epoch模型
        epoch_model_path = os.path.join(model_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_path)

        # 更新最佳模型指标
        if val_acc > best_metrics['val_acc'] or \
           (val_acc == best_metrics['val_acc'] and avg_val_loss < best_metrics['val_loss']):
            best_metrics.update({
                'epoch': epoch+1,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'train_acc': train_acc
            })
            # 保存最佳模型副本
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))

        # 记录日志
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, 
                running_loss/len(train_loader) if len(train_loader) > 0 else 0.0,
                avg_val_loss,
                train_acc,
                val_acc,
                time.time()-epoch_start,
                epoch_model_path
            ])

        # 输出统计信息（新增准确率显示）
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/10 | "
              f"Train Loss: {running_loss/len(train_loader):.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_acc:.2%} | "  # 新增
              f"Val Acc: {val_acc:.2%} | "     # 新增
              f"Time: {timedelta(seconds=int(epoch_time))}")

    # 最终输出最佳模型信息
    print(f"\n🏆 最佳模型：Epoch {best_metrics['epoch']}")
    print(f"● 验证集损失: {best_metrics['val_loss']:.4f}")
    print(f"● 训练准确率: {best_metrics['train_acc']:.2%}")
    print(f"● 验证准确率: {best_metrics['val_acc']:.2%}")
    print(f"● 模型路径: {os.path.join(model_dir, 'best_model.pth')}")

    # 最终处理
    total_time = timedelta(seconds=int(time.time()-total_start))
    print(f"\n训练完成！总耗时: {total_time}")
    
    cursor.close()
    conn.close()
    
    with open(scores_save_path, 'w') as f:
        json.dump({'train_scores': train_scores, 'val_scores': val_scores}, f)
        print(f"评分已保存到 {scores_save_path}")