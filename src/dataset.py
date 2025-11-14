import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset
from config import SCRATCH_NUM, SCRATCH_WIDTH

class ImageInpaintingDataset(Dataset):
    """
    图像修复（Inpainting）数据集。

    - 功能：从目录读取图像，生成或加载掩码，返回用于修复任务的样本。
    - 掩码语义：二值掩码中 1 表示缺失区域（需要被修复），0 表示保留区域。
    - 外部掩码：默认认为外部掩码的白色区域表示缺失区域；若你的外部掩码表示的是“有效区域”（如人脸轮廓），使用前需将其取反。

    参数说明：
    - data_dir：图像目录（支持 png/jpg/jpeg）
    - transform：torchvision.transforms 组成的预处理/增强流水线
    - mask_type：随机掩码类型，可选 "random_rect" / "center" / "scratches" / "irregular"
    - mask_ratio：随机掩码的覆盖比例（0~1），在不同类型中表示方式略有差异
    - mask_dir：外部掩码目录（掩码需与图像同名，扩展名可为 png/jpg/jpeg）
    - use_external_mask：是否优先使用外部掩码（找不到时回退到随机掩码）
    - return_masked：是否同时返回掩蔽后的图像（masked = img * (1 - mask)）
    - max_retry：个别坏样本读取失败时的重试次数
    - seed：随机种子；不给定则使用全局随机
    """
    def __init__(self, data_dir, transform=None, mask_type="random_rect", mask_ratio=0.2, mask_dir=None, use_external_mask=False, return_masked=False, max_retry=5, seed=None, external_mask_is_valid_region=False, external_mask_dilate=0, external_mask_mode="name"):
        self.data_dir = data_dir
        self.image_paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, f)) and f.lower().endswith(("png", "jpg", "jpeg"))
        ]
        self.transform = transform
        self.mask_type = mask_type
        self.mask_ratio = float(mask_ratio)
        self.mask_dir = mask_dir
        self.use_external_mask = use_external_mask and mask_dir is not None
        self.return_masked = return_masked
        self.max_retry = int(max_retry)
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.external_mask_is_valid_region = bool(external_mask_is_valid_region)
        self.external_mask_dilate = int(external_mask_dilate)
        self.external_mask_mode = str(external_mask_mode)
        self.mask_paths_sorted = []
        if self.use_external_mask:
            try:
                self.mask_paths_sorted = [
                    os.path.join(mask_dir, f)
                    for f in sorted(os.listdir(mask_dir))
                    if os.path.isfile(os.path.join(mask_dir, f)) and f.lower().endswith(("png", "jpg", "jpeg"))
                ]
            except Exception:
                self.mask_paths_sorted = []

    def __len__(self):
        return len(self.image_paths)

    def _generate_mask(self, h, w):
        """根据设置的 mask_type 生成尺寸为 (1, h, w) 的二值掩码。
        语义：1 表示缺失区域；0 表示保留区域。"""
        mask = torch.zeros(1, h, w)
        if self.mask_type == "random_rect":
            # 通过铺设若干随机矩形，使覆盖比例接近 mask_ratio
            target = max(min(self.mask_ratio, 0.95), 0.0)
            tol = 0.05
            count = 0
            while (mask.sum().item() / (h * w)) < max(target - tol, 0.0) and count < 32:
                rect_h = self.rng.randint(int(h * 0.1), int(h * 0.3))
                rect_w = self.rng.randint(int(w * 0.1), int(w * 0.3))
                x = self.rng.randint(0, max(1, w - rect_w))
                y = self.rng.randint(0, max(1, h - rect_h))
                mask[:, y:y+rect_h, x:x+rect_w] = 1
                count += 1
        elif self.mask_type == "center":
            # 以图像中心为基准构造一个近似正方形的缺失区域，面积受 mask_ratio 控制
            center_h, center_w = h // 2, w // 2
            size = int(min(h, w) * np.sqrt(self.mask_ratio))
            y0, y1 = center_h - size // 2, center_h + size // 2
            x0, x1 = center_w - size // 2, center_w + size // 2
            mask[:, max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 1
        elif self.mask_type == "scratches":
            # 以多条线段模拟划痕类缺损
            scratch_img = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(scratch_img)
            for _ in range(SCRATCH_NUM):
                x1 = self.rng.randint(0, w)
                y1 = self.rng.randint(0, h)
                length = self.rng.randint(int(min(h, w) * 0.2), int(min(h, w) * 0.6))
                angle = self.rng.uniform(0, 2 * np.pi)
                x2 = int(np.clip(x1 + length * np.cos(angle), 0, w - 1))
                y2 = int(np.clip(y1 + length * np.sin(angle), 0, h - 1))
                width = self.rng.randint(SCRATCH_WIDTH[0], SCRATCH_WIDTH[1] + 1)
                draw.line([(x1, y1), (x2, y2)], fill=255, width=width)
            scratch_tensor = torch.from_numpy(np.array(scratch_img)).float().unsqueeze(0) / 255.0
            mask = (scratch_tensor > 0.5).float()
        elif self.mask_type == "irregular":
            # 以折线笔触随机游走生成不规则缺损区域，贴近真实破损
            irr_img = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(irr_img)
            strokes = self.rng.randint(8, 18)
            for _ in range(strokes):
                points = []
                x, y = self.rng.randint(0, w), self.rng.randint(0, h)
                for _ in range(self.rng.randint(3, 8)):
                    dx = self.rng.randint(-int(w * 0.15), int(w * 0.15))
                    dy = self.rng.randint(-int(h * 0.15), int(h * 0.15))
                    x = int(np.clip(x + dx, 0, w - 1))
                    y = int(np.clip(y + dy, 0, h - 1))
                    points.append((x, y))
                width = self.rng.randint(3, 12)
                if len(points) >= 2:
                    draw.line(points, fill=255, width=width)
            irr_tensor = torch.from_numpy(np.array(irr_img)).float().unsqueeze(0) / 255.0
            mask = (irr_tensor > 0.5).float()
        return mask

    def _load_external_mask(self, img_path, h, w, idx=None):
        """优先加载与图像同名的外部掩码文件，并调整到图像尺寸。

        - 同名匹配：使用图像的基名（不含扩展名）在 mask_dir 中查找 png/jpg/jpeg。
        - 二值化：阈值 0.5；返回 1=缺失区域。
        - 找不到文件：返回 None，由调用者回退到随机掩码生成。
        """
        if not self.use_external_mask:
            return None
        p = None
        mode = (self.external_mask_mode or "name").lower()
        if mode == "name":
            base = os.path.splitext(os.path.basename(img_path))[0]
            candidates = [os.path.join(self.mask_dir, base + ext) for ext in (".png", ".jpg", ".jpeg")]
            for cand in candidates:
                if os.path.isfile(cand):
                    p = cand
                    break
        elif mode == "index" and idx is not None and len(self.mask_paths_sorted) > 0:
            if 0 <= idx < len(self.mask_paths_sorted):
                p = self.mask_paths_sorted[idx]
        elif mode == "cycle" and idx is not None and len(self.mask_paths_sorted) > 0:
            p = self.mask_paths_sorted[idx % len(self.mask_paths_sorted)]
        elif mode == "random" and len(self.mask_paths_sorted) > 0:
            p = self.rng.choice(self.mask_paths_sorted)

        if p and os.path.isfile(p):
            m = Image.open(p).convert("L").resize((w, h))
            if self.external_mask_dilate and self.external_mask_dilate > 0:
                k = max(1, 2 * int(self.external_mask_dilate) + 1)
                m = m.filter(ImageFilter.MaxFilter(k))
            arr = np.array(m)
            t = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
            mask = (t > 0.5).float()
            if self.external_mask_is_valid_region:
                mask = 1.0 - mask
            return mask
        return None

    def __getitem__(self, idx):
        """读取单个样本。
        返回：
        - (img, mask) 或 (img, mask, masked) 其中 masked=img*(1-mask)
        - 遇到坏样本会按 max_retry 次数跳过并继续
        """
        attempts = 0
        while attempts < self.max_retry:
            path = self.image_paths[idx]
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                h, w = img.shape[1], img.shape[2]
                mask = self._load_external_mask(path, h, w, idx)
                if mask is None:
                    mask = self._generate_mask(h, w)
                if self.return_masked:
                    # 将缺失区域置零，得到掩蔽图像
                    masked = img * (1 - mask)
                    return img, mask, masked
                return img, mask
            except Exception:
                # 读取失败：尝试下一个样本，最多重试 max_retry 次
                idx = (idx + 1) % len(self.image_paths)
                attempts += 1
        raise RuntimeError("Failed to load image after multiple attempts")


class TrainTestSplitDataset(Dataset):
    """
    从训练数据集中分割测试数据的包装类。
    
    参数说明：
    - base_dataset: 基础数据集实例
    - test_ratio: 测试集比例（0~1），默认为0.2（20%）
    - is_test: 是否返回测试集部分（True=测试集，False=训练集）
    """
    
    def __init__(self, base_dataset, test_ratio=0.2, is_test=False):
        self.base_dataset = base_dataset
        self.test_ratio = test_ratio
        self.is_test = is_test
        
        # 计算分割点
        total_size = len(base_dataset)
        test_size = int(total_size * test_ratio)
        
        if is_test:
            # 测试集：最后 test_size 个样本
            self.indices = list(range(total_size - test_size, total_size))
        else:
            # 训练集：前 total_size - test_size 个样本
            self.indices = list(range(0, total_size - test_size))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 从基础数据集中获取对应的样本
        return self.base_dataset[self.indices[idx]]