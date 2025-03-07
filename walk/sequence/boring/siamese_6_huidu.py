import os
import sys
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
from datetime import timedelta, datetime
import csv
from tqdm import tqdm
import numpy as np

# --------------------------
# 全局配置：直接读取灰度图像，不使用预处理数据
# --------------------------
use_preprocessed = False  # 关闭预处理
# 原始图像目录（存放灰度图像或彩色图像，程序会转换为灰度）
image_dir = "walk/deeplabv3-master/training_logs/model_eval_seq_pp2"

# --------------------------
# 自定义转换：这里不再转换为类别标签，而是直接转换为灰度图的 Tensor
# --------------------------
class ConvertToGray(object):
    def __call__(self, image):
        # image: PIL Image
        # 转换为灰度图（单通道）
        image = image.convert("L")
        return transforms.ToTensor()(image)

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

# 数据加载函数
def load_pair_list(db_config):
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        query = "SELECT left_id, right_id, winner FROM pp2 WHERE category = 'boring'"
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

# 原始图像路径获取函数（用于非预处理模式）
def get_image_path(image_id):
    return os.path.join(image_dir, f"{image_id}_pred.png")

# --------------------------
# 创建评分表与插入评分函数（修着修着被闲置了）
# --------------------------
def create_scores_table(db_config):
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_scores_boring_S (
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
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def insert_scores(db_config, scores, split, conn, cursor, batch_size=1000):
    try:
        query = "INSERT INTO image_scores_boring_S VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE score=VALUES(score)"
        items = [(str(k), v, split) for k, v in scores.items()]
        for i in range(0, len(items), batch_size):
            cursor.executemany(query, items[i:i+batch_size])
            conn.commit()
    except pymysql.MySQLError as e:
        print(f"数据库错误: {e.args}")

# --------------------------
# 2. 数据集：直接读取灰度图像
# --------------------------
class SegmentationPairDataset(Dataset):
    def __init__(self, pair_list, transform=None):
        self.pair_list = pair_list
        self.transform = transform
        # 标签映射： "left" 表示左边获胜，转换为 1；"right" 表示右边获胜，转换为 -1；"equal" 转换为 0
        self.label_map = {"left": 1, "right": -1, "equal": 0}
        self.cache = OrderedDict()
        self.max_cache_size = 1200  # 根据内存容量调整

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]

        left_id, right_id, winner = self.pair_list[idx]
        
        # 直接读取灰度图像
        left_path = get_image_path(left_id)
        right_path = get_image_path(right_id)
        if not all(os.path.exists(p) for p in [left_path, right_path]):
            raise FileNotFoundError(f"缺失文件: {left_path} 或 {right_path}")
        img_left = Image.open(left_path).convert("L")  # 转换为灰度图
        img_right = Image.open(right_path).convert("L")
        if self.transform:
            img_left = self.transform(img_left)
            img_right = self.transform(img_right)

        label = torch.tensor(self.label_map[winner], dtype=torch.float)

        if len(self.cache) >= self.max_cache_size:
            self.cache.pop(next(iter(self.cache)))  # 删除最早插入的缓存
        self.cache[idx] = (img_left, img_right, label, left_id, right_id)
        return self.cache[idx]

# --------------------------
# 3. 内存优化模型
# --------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 注意：输入通道为 1，因为读取的是灰度图像
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
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
        # 如不需要节省显存，可不使用 checkpoint（此处注释掉）
        # x = torch.utils.checkpoint.checkpoint(self.conv, x)
        x = self.conv(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# --------------------------
# 4. 动态损失计算
# --------------------------
def compute_loss(r1, r2, label, margin=0.5):
    loss = torch.tensor(0.0, device=r1.device)
    margin_loss = nn.MarginRankingLoss(margin=margin, reduction="sum")
    mse_loss = nn.MSELoss(reduction="sum")

    pos_mask = label == 1
    neg_mask = label == -1
    eq_mask = label == 0

    if pos_mask.any():
        loss += margin_loss(r1[pos_mask], r2[pos_mask], torch.ones_like(r1[pos_mask]))
    if neg_mask.any():
        loss += margin_loss(r2[neg_mask], r1[neg_mask], torch.ones_like(r2[neg_mask]))
    if eq_mask.any():
        loss += mse_loss(r1[eq_mask], r2[eq_mask])

    return loss / label.numel()

# --------------------------
# 5. 自适应硬件配置
# --------------------------
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

# --------------------------
# 6. 训练流程
# --------------------------
if __name__ == '__main__':
    # # 如果使用预处理数据，检查预处理目录是否存在或为空，若需要则预处理
    # if use_preprocessed:
    #     if not os.path.exists(label_dir) or len(os.listdir(label_dir)) == 0:
    #         print("预处理数据不存在，开始预处理...")
    #         preprocess_all_images(image_dir, label_dir, color_map)

    # 日志与模型保存目录配置
    log_dir = "walk/deeplabv3-master/training_logs/siamese_boring"
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

    # 创建 CSV 日志文件头
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'val_loss', 
            'train_acc', 'val_acc', 'epoch_time',
            'model_path'
        ])

    multiprocessing.freeze_support()

    # 初始化硬件配置
    device, batch_size, num_workers, grad_accum = hardware_config()
    
    # 使用转换：调整尺寸并转换为灰度图（单通道）
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()  # 若图像已转为灰度，ToTensor()会自动生成单通道 tensor
    ])

    # 数据加载
    pair_list = load_pair_list(db_config)
    # dataset = SegmentationPairDataset(pair_list, transform if not use_preprocessed else None)
    dataset = SegmentationPairDataset(pair_list, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator(device='cpu'),
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=1  # 降低预加载，减少内存压力
    )

    val_loader = DataLoader(
        val_set,
        batch_size=max(batch_size // 2, 8),
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True
    )

    # 模型初始化并加载到设备（使用 channel_last 内存格式）
    model = SiameseNetwork().to(device, memory_format=torch.channels_last)
    
    # 如果支持 torch.compile 且非 Windows，则编译模型以提升速度
    if sys.platform != "win32" and hasattr(torch, "compile"):
        model = torch.compile(model)
    else:
        print("torch.compile 在 Windows 平台不可用，跳过编译。")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    model_save_path = os.path.join(model_dir, "siamese_model.pth")
    scores_save_path = os.path.join(log_dir, "scores.json")
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
    for epoch in range(12):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        
        # 初始化训练统计指标
        running_loss = 0.0
        running_correct = 0  # 训练正确样本数
        total_samples = 0    # 训练有效样本数
        train_scores = {}

        # 训练阶段
        for i, (img_l, img_r, labels, ids_l, ids_r) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
            # 数据传输：采用非阻塞传输和 channels_last 格式
            img_l = img_l.to(device, non_blocking=True, memory_format=torch.channels_last)
            img_r = img_r.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=torch.cuda.is_available()):
                out1, out2 = model(img_l, img_r)
                loss = compute_loss(out1, out2, labels) / grad_accum

                with torch.no_grad():
                    preds = torch.where(out1 > out2, 1, -1)
                    mask = labels != 0  # 排除平局样本
                    correct = (preds.flatten()[mask] == labels[mask]).sum().item()
                    running_correct += correct
                    total_samples += mask.sum().item()

            if torch.cuda.is_available():
                scaler.scale(loss).backward()
                if (i + 1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += loss.item()

        train_acc = running_correct / total_samples if total_samples > 0 else 0.0

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0  # 验证正确样本数
        val_total = 0    # 验证有效样本数
        val_scores = {}
        
        with torch.inference_mode(), autocast(enabled=torch.cuda.is_available()):
            for img_l, img_r, labels, ids_l, ids_r in tqdm(val_loader, desc="Validation", leave=False):
                img_l = img_l.to(device, non_blocking=True, memory_format=torch.channels_last)
                img_r = img_r.to(device, non_blocking=True, memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                
                out1, out2 = model(img_l, img_r)
                val_loss += compute_loss(out1, out2, labels).item()

                with torch.no_grad():
                    preds = torch.where(out1 > out2, 1, -1)
                    mask = labels != 0  # 排除平局样本
                    val_correct += (preds.flatten()[mask] == labels[mask]).sum().item()
                    val_total += mask.sum().item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        # 保存每个 epoch 的模型
        epoch_model_path = os.path.join(model_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_path)

        # 更新最佳模型指标
        if val_acc > best_metrics['val_acc'] or (val_acc == best_metrics['val_acc'] and avg_val_loss < best_metrics['val_loss']):
            best_metrics.update({
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'train_acc': train_acc
            })
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))

        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                running_loss / len(train_loader) if len(train_loader) > 0 else 0.0,
                avg_val_loss,
                train_acc,
                val_acc,
                time.time() - epoch_start,
                epoch_model_path
            ])

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/12 | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | Time: {timedelta(seconds=int(epoch_time))}")

        scheduler.step()

    print(f"\n🏆 最佳模型：Epoch {best_metrics['epoch']}")
    print(f"● 验证集损失: {best_metrics['val_loss']:.4f}")
    print(f"● 训练准确率: {best_metrics['train_acc']:.2%}")
    print(f"● 验证准确率: {best_metrics['val_acc']:.2%}")
    print(f"● 模型路径: {os.path.join(model_dir, 'best_model.pth')}")

    total_time = timedelta(seconds=int(time.time() - total_start))
    print(f"\n训练完成！总耗时: {total_time}")
    
    cursor.close()
    conn.close()
    
    with open(scores_save_path, 'w') as f:
        json.dump({'train_scores': train_scores, 'val_scores': val_scores}, f)
        print(f"评分已保存到 {scores_save_path}")
