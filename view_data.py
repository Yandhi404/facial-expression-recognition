import random
import matplotlib.pyplot as plt
from torchvision import datasets

# 读取训练集
dataset = datasets.ImageFolder("data/train")

# 随机抽取 12 张图片
samples = random.sample(dataset.samples, 12)

fig, axes = plt.subplots(3, 4, figsize=(10, 8))

for ax, (image_path, label) in zip(axes.flat, samples):
    image = plt.imread(image_path)

    ax.imshow(image, cmap="gray")
    ax.set_title(dataset.classes[label])
    ax.axis("off")

plt.tight_layout()
plt.show()