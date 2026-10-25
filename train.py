import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

from model import EmotionCNN


# =========================
# 1. 基本设置
# =========================
SEED = 42
BATCH_SIZE = 128
EPOCHS = 35
LEARNING_RATE = 0.001

DATA_DIR = Path("data")
MODEL_DIR = Path("checkpoints")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model_v3.pth"

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# =========================
# 2. 数据处理
# =========================

# 训练时增加轻微的数据变化，帮助模型学习更稳健的特征
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 验证和测试时不做随机增强
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 分别加载数据，确保训练增强不会影响验证集
train_dataset_aug = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform
)

train_dataset_eval = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=eval_transform
)

classes = train_dataset_aug.classes

assert classes == train_dataset_eval.classes
assert classes == test_dataset.classes

print(f"情绪类别: {classes}")
print(f"训练数据总量: {len(train_dataset_aug)}")
print(f"测试数据总量: {len(test_dataset)}")

# =========================
# 3. 划分训练集和验证集
# =========================

generator = torch.Generator().manual_seed(SEED)

train_size = int(len(train_dataset_aug) * 0.9)
val_size = len(train_dataset_aug) - train_size

train_indices, val_indices = random_split(
    range(len(train_dataset_aug)),
    [train_size, val_size],
    generator=generator
)

# 训练子集使用数据增强
train_subset = Subset(
    train_dataset_aug,
    train_indices.indices
)

# 验证子集不使用数据增强
val_subset = Subset(
    train_dataset_eval,
    val_indices.indices
)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print(f"实际训练集: {len(train_subset)}")
print(f"验证集: {len(val_subset)}")

# =========================
# 4. 创建模型
# =========================

model = EmotionCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# =========================
# 5. 评估函数
# =========================

def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# =========================
# 6. 开始训练
# =========================

best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    val_loss, val_accuracy = evaluate(model, val_loader)

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2%} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2%}"
    )

    # 只根据验证集保存最佳模型，不使用测试集挑选模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "epoch": epoch + 1,
            "val_accuracy": val_accuracy
        }, MODEL_PATH)

        print(f"已保存当前最佳模型: {MODEL_PATH}")

# =========================
# 7. 最后测试一次
# =========================

print("\n训练结束，加载验证集表现最好的模型。")

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

test_loss, test_accuracy = evaluate(model, test_loader)

print(f"\n最佳模型轮次: {checkpoint['epoch']}")
print(f"最佳验证集准确率: {checkpoint['val_accuracy']:.2%}")
print(f"最终测试集准确率: {test_accuracy:.2%}")
print(f"模型保存位置: {MODEL_PATH}")