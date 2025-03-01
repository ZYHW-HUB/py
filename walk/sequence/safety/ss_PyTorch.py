import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pymysql
from PIL import Image
from tqdm import tqdm  # 导入 tqdm

# 定义自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_filename = self.df.iloc[idx]['image_id']
        score = self.df.iloc[idx]['score']
        
        # 构造完整文件名：基础文件名加上 _pred.png 后缀
        # full_filename = os.path.join(self.image_dir, f"{image_filename}_pred.png")
        # image = Image.open(full_filename).convert('RGB')
        full_filename = os.path.join(self.image_dir, f"{image_filename}.png")
        image = Image.open(full_filename).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(score, dtype=torch.float32)

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
    BATCH_SIZE = 8  # 减小批次大小

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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 构建简单的 CNN 回归模型
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(128 * (IMG_HEIGHT // 8) * (IMG_WIDTH // 8), 128)
            self.fc2 = nn.Linear(128, 1)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            x = x.view(-1, 128 * (IMG_HEIGHT // 8) * (IMG_WIDTH // 8))
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
        for images, scores in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
            images, scores = images.to(device), scores.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), scores)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        print(torch.cuda.memory_summary(device=device, abbreviated=False))  # 检查 GPU 内存使用情况

    # 清理缓存
    torch.cuda.empty_cache()

    # 预测并评估模型
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, scores in tqdm(val_loader, desc="Validation"):
            images, scores = images.to(device), scores.to(device)
            outputs = model(images)
            y_pred.extend(outputs.cpu().numpy().flatten())
            y_true.extend(scores.cpu().numpy())

    mse_value = mean_squared_error(y_true, y_pred)
    r2_value = r2_score(y_true, y_pred)
    print(f"Mean Squared Error: {mse_value}")
    print(f"R-squared: {r2_value}")

    # 可视化预测结果
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("True Score")
    plt.ylabel("Predicted Score")
    plt.title("True vs Predicted Scores")
    plt.show()

if __name__ == '__main__':
    main()