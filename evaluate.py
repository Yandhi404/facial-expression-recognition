
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import EmotionCNN


# 1. 选择运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")


# 2. 加载测试数据
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.ImageFolder("data/test", transform=transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)


# 3. 加载训练时保存的最佳模型
checkpoint_path = Path("checkpoints/best_model_v3.pth")

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

classes = checkpoint["classes"]

model = EmotionCNN(num_classes=len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"最佳模型来自第 {checkpoint['epoch']} 轮")
print(f"最佳验证集准确率：{checkpoint['val_accuracy']:.2%}")


# 4. 统计混淆矩阵
num_classes = len(classes)
confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu()

        for true_label, pred_label in zip(labels, predictions):
            confusion[true_label, pred_label] += 1


# 5. 打印每类识别准确率
print("\n各类表情的识别准确率：")

for i, emotion in enumerate(classes):
    total = confusion[i].sum().item()
    correct = confusion[i, i].item()
    accuracy = correct / total if total else 0

    print(
        f"{emotion:10s}: {accuracy:.2%} "
        f"({correct}/{total})"
    )

overall = confusion.diag().sum().item() / confusion.sum().item()
print(f"\n总体准确率：{overall:.2%}")


# 6. 绘制混淆矩阵
matrix = confusion.numpy()

fig, ax = plt.subplots(figsize=(9, 7))
image = ax.imshow(matrix, cmap="Blues")

ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(classes, rotation=45)
ax.set_yticklabels(classes)

ax.set_xlabel("Predicted emotion")
ax.set_ylabel("True emotion")
ax.set_title("FER-2013 Confusion Matrix")

for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, str(matrix[i, j]),
                ha="center", va="center",
                color="white" if matrix[i, j] > matrix.max() * 0.5 else "black")

fig.colorbar(image, ax=ax)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

print("混淆矩阵已保存至 confusion_matrix.png")