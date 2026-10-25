import torch
import cv2
import numpy as np

print("=" * 40)
print("环境检查")
print("=" * 40)

print("PyTorch 版本：", torch.__version__)
print("OpenCV 版本：", cv2.__version__)
print("NumPy 版本：", np.__version__)

print("CUDA 是否可用：", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 型号：", torch.cuda.get_device_name(0))
else:
    print("暂未检测到可用 GPU")

print("=" * 40)
print("环境检查完成！")