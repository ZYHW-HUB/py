import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 模型权重路径（请根据实际情况修改）
    model_path = "walk/deeplabv3-master/training_logs/siamese6/saved_models/best_model.pth"
    model = load_trained_model(model_path, device=device)
    
    # 测试图像路径（请根据实际情况修改）
    test_image_path = "walk/deeplabv3-master/training_logs/model_eval_seq/some_test_image_pred.png"
    
    score = score_image(model, test_image_path, device=device)
    print(f"图像评分: {score:.4f}")
