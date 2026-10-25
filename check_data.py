from pathlib import Path
from torchvision import datasets

# 数据集所在位置
DATA_DIR = Path("data")

# 七种表情
EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

def check_dataset(split):
    folder = DATA_DIR / split

    if not folder.exists():
        print(f"错误：找不到文件夹 {folder.resolve()}")
        return

    dataset = datasets.ImageFolder(str(folder))

    print(f"\n===== {split.upper()} 数据集 =====")
    print(f"图片总数：{len(dataset)}")
    print(f"识别到的类别：{dataset.classes}")

    counts = {emotion: 0 for emotion in EMOTIONS}

    for _, label in dataset.samples:
        emotion = dataset.classes[label]
        counts[emotion] += 1

    for emotion, count in counts.items():
        print(f"{emotion:10s}: {count} 张")

if __name__ == "__main__":
    check_dataset("train")
    check_dataset("test")