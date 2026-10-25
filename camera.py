
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import EmotionCNN


# 1. 基本设置
MODEL_PATH = Path("checkpoints") / "best_model_v3.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载训练好的模型
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"找不到模型文件：{MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device)

classes = checkpoint["classes"]

model = EmotionCNN(num_classes=len(classes)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"模型加载成功，最佳轮次：{checkpoint['epoch']}")
print(f"运行设备：{device}")
print(f"表情类别：{classes}")


# 3. 图片预处理，保持与训练时一致
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 4. 使用 OpenCV 自带的人脸检测器
cascade_path = (
    Path(cv2.data.haarcascades)
    / "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(str(cascade_path))

if face_detector.empty():
    raise RuntimeError("人脸检测器加载失败，请检查 OpenCV 安装。")


# 5. 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "无法打开摄像头，请检查摄像头权限或尝试修改 VideoCapture(0)。"
    )

print("摄像头已启动。按 Q 键退出。")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("无法读取摄像头画面。")
            break

        # 镜像显示，更符合日常使用习惯
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            # 从原始彩色画面裁剪人脸
            face = frame[y:y + h, x:x + w]

            if face.size == 0:
                continue

            # 转为 RGB，再交给 torchvision 处理
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_face = Image.fromarray(face_rgb)
            tensor = preprocess(pil_face).unsqueeze(0).to(device)

            # 6. 预测表情
            with torch.inference_mode():
                logits = model(tensor)
                probabilities = F.softmax(logits, dim=1)[0]
                index = int(probabilities.argmax().item())

            emotion = classes[index]
            confidence = probabilities[index].item() * 100

            # 绘制人脸框和预测结果
            label = f"{emotion}: {confidence:.1f}%"

            cv2.rectangle(
                frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2
            )

            text_y = max(y - 10, 25)

            cv2.putText(
                frame, label, (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

        cv2.imshow("Real-time Facial Expression Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("摄像头已关闭。")
