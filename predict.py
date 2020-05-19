# ライブラリのインポート
import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn

from torchvision.models import densenet161
from torchvision import transforms


densenet = densenet161(pretrained=True)
data_labels = ("橋本環奈", "石原さとみ", "深田恭子", "新垣結衣", "本田翼")
MODEL_PATH = "./model/model.pt"

size = 224
std = (0.281, 0.275, 0.274)
mean = (0.591, 0.520, 0.491)


# 中央上部を切り取る（前処理）
class CenterTopCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = np.array(img)
        shape = img.shape
        h, w = shape[0], shape[1]
        padding_side = int((w - self.size) / 2)
        img = img[:self.size, padding_side:w-padding_side]
        if img.shape[1] == 225:
            img = img[:, 1:]

        return Image.fromarray(np.uint8(img))


# ぼかしを加える（前処理）
class Gaussian:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, img):
        img = img.filter(ImageFilter.GaussianBlur(self.alpha))
        return img


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = densenet
        self.additional_layers = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.Dropout2d(p=0.4),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.additional_layers(x)
        return x


def pred_actress(img_path):
    img = Image.open(img_path)
    transformed = transform(img)  # 前処理をかける
    transformed.unsqueeze_(0)  # 次元の追加

    if torch.cuda.is_available():
        transformed = transformed.to("cuda:0")

    with torch.no_grad():
        calc_score = model(transformed)

    sum = torch.sum(np.exp(calc_score), dim=1)
    max = torch.max(np.exp(calc_score), dim=1)
    y_label = max.indices
    score = max.values / sum
    act_name = data_labels[y_label]

    return act_name, score


transform = transforms.Compose([
    transforms.Resize(size),   # リサイズ
    CenterTopCrop(size),   # リサイズした画像の中央の正方形を切り取る
    Gaussian(1.2),   # ぼかしを加える
    transforms.ToTensor(),   # Tensor型に変換
    transforms.Normalize(mean=mean, std=std)   # 標準化
])

if torch.cuda.is_available():
    device = "cuda:0"
    model = Net().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

# 推論モード
model.eval()