import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
#import src.config
# --------------------------
# 配置参数（请根据你的模型调整）
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./saved_models/gen_best.pth"
IMAGE_SIZE = None  # 使用原图尺寸进行推理
BRUSH_SIZE = 2  # 画笔大小


# --------------------------
# 关键：导入你的模型类（必须替换为你训练时的模型定义）
# --------------------------
# 示例：假设你的模型类在 model.py 中，名为 InpaintingGAN
# from model import InpaintingGAN  # 请替换为你的实际导入路径和模型类名
# 这里提供一个与你训练日志匹配的简化版模型框架（如果你的模型结构不同，务必替换为你的模型类）
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.models import UNetGenerator

class InpaintingGAN(UNetGenerator):
    def __init__(self):
        super().__init__()


# --------------------------
# 交互涂画工具（生成修复区域掩码）
# --------------------------
class MaskDrawer:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.original = self.image.copy()
        self.mask = np.zeros_like(self.image[:, :, 0], dtype=np.uint8)
        self.drawing = False
        self.brush_size = BRUSH_SIZE
        self.zoom = 100
        self.window = "Draw Mask (Left: Paint, Right: Erase, 's' to Save)"

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Brush", self.window, int(self.brush_size), 200, self._on_brush_change)
        cv2.createTrackbar("Zoom%", self.window, int(self.zoom), 400, self._on_zoom_change)
        cv2.setMouseCallback(self.window, self._mouse_callback)
        self._update_display()

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self._draw(x, y, erase=False)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self._draw(x, y, erase=False)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self._draw(x, y, erase=True)
        elif event == cv2.EVENT_RBUTTONUP:
            self.drawing = False

    def _draw(self, x, y, erase=False):
        scale = max(self.zoom, 1) / 100.0
        ox = int(x / scale)
        oy = int(y / scale)
        h, w = self.mask.shape
        ox = max(0, min(w - 1, ox))
        oy = max(0, min(h - 1, oy))
        color = 0 if erase else 255
        cv2.circle(self.mask, (ox, oy), int(self.brush_size), color, -1)
        self._update_display()

    def _on_brush_change(self, val):
        self.brush_size = max(1, int(val))

    def _on_zoom_change(self, val):
        self.zoom = max(10, int(val))
        self._update_display()

    def _update_display(self):
        scale = max(self.zoom, 1) / 100.0
        base = self.original.copy()
        overlay = base.copy()
        overlay[self.mask == 255] = [0, 0, 255]
        disp = cv2.addWeighted(overlay, 0.5, base, 0.5, 0)
        disp = cv2.resize(disp, (int(base.shape[1] * scale), int(base.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(self.window, disp)

    def run(self):
        self._update_display()
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                cv2.destroyAllWindows()
                return self.original, self.mask  # 返回原图和掩码
            elif key == 27:  # ESC退出
                cv2.destroyAllWindows()
                return None, None


# --------------------------
# 预处理（与模型训练时保持一致）
# --------------------------
def preprocess(image, mask, target_size=0):
    """将图像和掩码转换为模型输入格式。
    - 若 target_size>0：先将图像与掩码统一缩放为 target_size×target_size（匹配训练尺寸有助于细节）。
    - 否则：保留原图尺寸，必要时边缘补齐到32倍数以适配下采样结构。
    """
    h, w = image.shape[:2]
    if target_size and target_size > 0:
        image = cv2.resize(image, (target_size, target_size))
        mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        h, w = target_size, target_size
    else:
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        pad_h = (32 - (h % 32)) % 32
        pad_w = (32 - (w % 32)) % 32
        if pad_h or pad_w:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
            h = image.shape[0]
            w = image.shape[1]

    # BGR→RGB（训练用PIL读取为RGB；推理需统一色彩顺序）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    pad_h = (32 - (h % 32)) % 32
    pad_w = (32 - (w % 32)) % 32
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

    # 归一化（请根据你的训练配置修改mean和std）
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 示例：转为[-1,1]
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 转为 tensor 并增加批次维度
    image_tensor = img_transform(image_rgb).unsqueeze(0).to(DEVICE)
    mask_tensor = mask_transform(mask).unsqueeze(0).to(DEVICE)  # 1通道掩码
    mask_tensor = (mask_tensor > 0.5).float()
    return image_tensor, mask_tensor


# --------------------------
# 后处理（将模型输出转为图像）
# --------------------------
def postprocess(output_tensor, original_size):
    """将模型输出 tensor 转换为可显示的图像"""
    # 移除批次维度并转回CPU
    output = output_tensor.squeeze().cpu().detach()
    ow, oh = original_size
    output = output[:, :oh, :ow]
    # 反归一化（与预处理对应）
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # 从[-1,1]转回[0,1]
        transforms.ToPILImage()
    ])
    # 恢复原图尺寸：此处已按原图裁剪，无需再插值缩放
    output_img = inv_transform(output)
    return np.array(output_img)


# --------------------------
# 核心：调用模型进行修复（已修复加载逻辑）
# --------------------------
def load_mask_file(mask_path, size, invert=False, dilate=0):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(mask_path)
    m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    if invert:
        m = 255 - m
    if dilate and dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * int(dilate) + 1, 2 * int(dilate) + 1))
        m = cv2.dilate(m, k)
    return m

def main(image_path, mask_path=None, invert_mask=False, dilate=0):
    # 1. 修复：创建模型实例 + 加载权重（关键修正）
    model = InpaintingGAN().to(DEVICE)  # 初始化模型
    weights = torch.load(MODEL_PATH, map_location=DEVICE)  # 加载权重字典（OrderedDict）
    model.load_state_dict(weights)  # 将权重加载到模型实例中
    model.eval()  # 现在模型实例有eval()方法了
    print(f"✅ 已加载模型权重：{MODEL_PATH}")

    # 2. 交互绘制掩码
    if mask_path:
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise FileNotFoundError(image_path)
        mask = load_mask_file(mask_path, (original_img.shape[1], original_img.shape[0]), invert=invert_mask, dilate=dilate)
    else:
        drawer = MaskDrawer(image_path)
        original_img, mask = drawer.run()
    if original_img is None:
        print("❌ 操作已取消")
        return
    original_size = (original_img.shape[1], original_img.shape[0])  # (宽, 高)

    # 3. 预处理（可选：传入 --size 与训练尺寸一致可提升细节）
    target_size = 0
    img_tensor, mask_tensor = preprocess(original_img, mask, target_size)

    # 4. 模型推理（核心调用）
    with torch.no_grad():  # 关闭梯度计算，加速推理
        repaired_tensor = model(img_tensor, mask_tensor)  # 模型输入：图像+掩码

    # 5. 后处理并显示结果
    repaired_img = postprocess(repaired_tensor, original_size)

    # 显示对比图
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(132)
    plt.title("Mask (Red = Area to Repair)")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.subplot(133)
    plt.title("Repaired Result")
    plt.imshow(repaired_img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 保存结果
    save_path = "repaired_result.png"
    cv2.imwrite(save_path, cv2.cvtColor(repaired_img, cv2.COLOR_RGB2BGR))
    print(f"✅ 修复结果已保存至：{save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=False, default="test2.jpeg")
    parser.add_argument("--mask", default=None)
    parser.add_argument("--invert_mask", action="store_true")
    parser.add_argument("--dilate", type=int, default=0)
    args = parser.parse_args()
    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)
    main(args.image, args.mask, args.invert_mask, args.dilate)