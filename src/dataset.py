import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from config import SCRATCH_NUM, SCRATCH_WIDTH

class ImageInpaintingDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_type="random_rect", mask_ratio=0.2, mask_dir=None, use_external_mask=False, return_masked=False, max_retry=5, seed=None):
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
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def __len__(self):
        return len(self.image_paths)

    def _generate_mask(self, h, w):
        mask = torch.zeros(1, h, w)
        if self.mask_type == "random_rect":
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
            center_h, center_w = h // 2, w // 2
            size = int(min(h, w) * np.sqrt(self.mask_ratio))
            y0, y1 = center_h - size // 2, center_h + size // 2
            x0, x1 = center_w - size // 2, center_w + size // 2
            mask[:, max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = 1
        elif self.mask_type == "scratches":
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

    def _load_external_mask(self, img_path, h, w):
        if not self.use_external_mask:
            return None
        base = os.path.splitext(os.path.basename(img_path))[0]
        candidates = [os.path.join(self.mask_dir, base + ext) for ext in (".png", ".jpg", ".jpeg")]
        for p in candidates:
            if os.path.isfile(p):
                m = Image.open(p).convert("L").resize((w, h))
                arr = np.array(m)
                t = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
                return (t > 0.5).float()
        return None

    def __getitem__(self, idx):
        attempts = 0
        while attempts < self.max_retry:
            path = self.image_paths[idx]
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                h, w = img.shape[1], img.shape[2]
                mask = self._load_external_mask(path, h, w)
                if mask is None:
                    mask = self._generate_mask(h, w)
                if self.return_masked:
                    masked = img * (1 - mask)
                    return img, mask, masked
                return img, mask
            except Exception:
                idx = (idx + 1) % len(self.image_paths)
                attempts += 1
        raise RuntimeError("Failed to load image after multiple attempts")