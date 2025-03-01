import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import pymysql
from PIL import Image
from tqdm import tqdm  # 导入 tqdm
import numpy as np

# 定义自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.IMG_HEIGHT = 512
        self.IMG_WIDTH = 682

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_filename = self.df.iloc[idx]['image_id']
        score = self.df.iloc[idx]['score']
        
        # 构造完整文件名：基础文件名加上 _pred.png 后缀
        full_filename = os.path.join(self.image_dir, f"{image_filename}_pred.png")
        image = Image.open(full_filename).convert('RGB')
        # full_filename = os.path.join(self.image_dir, f"{image_filename}.png")
        # image = Image.open(full_filename).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 生成位置信息
        x_coords = torch.arange(self.IMG_WIDTH).repeat(self.IMG_HEIGHT, 1).float() / self.IMG_WIDTH
        y_coords = torch.arange(self.IMG_HEIGHT).repeat(self.IMG_WIDTH, 1).t().float() / self.IMG_HEIGHT
        pos = torch.stack([x_coords, y_coords], dim=0)  # 形状为 (2, IMG_HEIGHT, IMG_WIDTH)
        
        return image, pos, torch.tensor(score, dtype=torch.float32)

def main():
    # 检查 GPU 是否可用并打印
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 路径设置：语义分割结果图像所在目录
    image_dir = "walk/deeplabv3-master/training_logs/model_eval_seq"  # 存放分割结果图像的目录

    # 连接到数据库并获取数据
    conn = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='123',
        database='scene',
        charset='utf8mb4'
    )

    query = """
    SELECT * FROM image_scores_safety
    """
    df = pd.read_sql(query, conn)

    print("样本数量:", len(df))

    # 图像参数：更新为 512 x 682（height x width）
    IMG_HEIGHT = 512
    IMG_WIDTH = 682
    BATCH_SIZE = 4  # 减小批次大小

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),  # 调整图像大小为 512 x 682
        transforms.ToTensor()  # 将图像转换为 tensor 并归一化到 [0, 1]
    ])

    # 创建数据集实例
    dataset = ImageDataset(df, image_dir, transform=transform)

    # 分割数据集：例如 80% 训练，20% 验证
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 构建简单的 CNN 回归模型
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(128 * (IMG_HEIGHT // 8) * (IMG_WIDTH // 8) + 2 * (IMG_HEIGHT // 8) * (IMG_WIDTH // 8), 128)
            self.fc2 = nn.Linear(128, 1)

        def forward(self, x, pos):
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool(torch.relu(self.conv3(x)))
            x = x.view(x.size(0), -1)  # 展平特征
            pos = pos.view(pos.size(0), -1)  # 展平位置信息
            x = torch.cat((x, pos), dim=1)  # 合并图像特征和位置信息
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleCNN().to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, pos, scores in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
            images, pos, scores = images.to(device), pos.to(device), scores.to(device)

            optimizer.zero_grad()
            outputs = model(images, pos)
            loss = criterion(outputs.squeeze(), scores)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 计算训练集的 MSE、RMSE、MAE 和 R-squared
        model.eval()
        train_preds = []
        train_true = []
        with torch.no_grad():
            for images, pos, scores in tqdm(train_loader, desc="Train Prediction"):
                images, pos, scores = images.to(device), pos.to(device), scores.to(device)
                outputs = model(images, pos)
                train_preds.extend(outputs.cpu().numpy().flatten())
                train_true.extend(scores.cpu().numpy())
        
        train_mse = mean_squared_error(train_true, train_preds)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(train_true, train_preds)
        train_r2 = r2_score(train_true, train_preds)

        # 计算验证集的 MSE、RMSE、MAE 和 R-squared
        val_preds = []
        val_true = []
        with torch.no_grad():
            for images, pos, scores in tqdm(val_loader, desc="Validation Prediction"):
                images, pos, scores = images.to(device), pos.to(device), scores.to(device)
                outputs = model(images, pos)
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_true.extend(scores.cpu().numpy())
        
        val_mse = mean_squared_error(val_true, val_preds)
        val_rmse = np.sqrt(val_mse)
        val_mae = mean_absolute_error(val_true, val_preds)
        val_r2 = r2_score(val_true, val_preds)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  训练集 - Loss: {running_loss/len(train_loader):.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R-squared: {train_r2:.4f}")
        print(f"  验证集 - MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R-squared: {val_r2:.4f}")
        print(torch.cuda.memory_summary(device=device, abbreviated=False))  # 检查 GPU 内存使用情况

        # 清理缓存
        torch.cuda.empty_cache()

        # 删除不必要的变量
        del images, pos, scores, outputs, loss
        torch.cuda.empty_cache()

    # 预测并评估模型
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, pos, scores in tqdm(val_loader, desc="Final Validation Prediction"):
            images, pos, scores = images.to(device), pos.to(device), scores.to(device)
            outputs = model(images, pos)
            y_pred.extend(outputs.cpu().numpy().flatten())
            y_true.extend(scores.cpu().numpy())

    mse_value = mean_squared_error(y_true, y_pred)
    rmse_value = np.sqrt(mse_value)
    mae_value = mean_absolute_error(y_true, y_pred)
    r2_value = r2_score(y_true, y_pred)
    print(f"Mean Squared Error: {mse_value:.4f}")
    print(f"Root Mean Squared Error: {rmse_value:.4f}")
    print(f"Mean Absolute Error: {mae_value:.4f}")
    print(f"R-squared: {r2_value:.4f}")

    # 可视化预测结果
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("True Score")
    plt.ylabel("Predicted Score")
    plt.title("True vs Predicted Scores")
    plt.show()

if __name__ == '__main__':
    main()