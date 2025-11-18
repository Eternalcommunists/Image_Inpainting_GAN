import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm  # 进度条（需pip install tqdm）

# 导入自定义模块
from models import UNetGenerator, PatchDiscriminator
from dataset import ImageInpaintingDataset, TrainTestSplitDataset
from losses import PerceptualLoss, L1MaskedLoss, GANLoss
from config import *  # 导入所有配置
from torch.amp import autocast, GradScaler
from PIL import Image, ImageDraw
import numpy as np
import random
import io

class RobustImageInpaintingDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_type="random_rect", mask_ratio=0.2):
        self.data_dir = data_dir
        self.image_paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]
        self.transform = transform
        self.mask_type = mask_type
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.image_paths)

    def _generate_mask(self, h, w):
        mask = torch.zeros(1, h, w)
        if self.mask_type == "random_rect":
            target = float(self.mask_ratio)
            tol = 0.05
            max_rect = 20
            count = 0
            while (mask.sum().item() / (h * w)) < max(target - tol, 0.0) and count < max_rect:
                rect_h = np.random.randint(int(h * 0.1), int(h * 0.3))
                rect_w = np.random.randint(int(w * 0.1), int(w * 0.3))
                x = np.random.randint(0, max(1, w - rect_w))
                y = np.random.randint(0, max(1, h - rect_h))
                mask[:, y:y+rect_h, x:x+rect_w] = 1
                count += 1
        elif self.mask_type == "center":
            center_h, center_w = h // 2, w // 2
            mask_size = int(min(h, w) * np.sqrt(self.mask_ratio))
            mask[:, center_h-mask_size//2:center_h+mask_size//2,
                 center_w-mask_size//2:center_w+mask_size//2] = 1
        elif self.mask_type == "scratches":
            scratch_img = Image.new('L', (w, h), 0)
            draw = ImageDraw.Draw(scratch_img)
            for _ in range(SCRATCH_NUM):
                x1 = np.random.randint(0, w)
                y1 = np.random.randint(0, h)
                length = np.random.randint(int(min(h, w)*0.2), int(min(h, w)*0.6))
                angle = np.random.uniform(0, 2*np.pi)
                x2 = int(np.clip(x1 + length*np.cos(angle), 0, w-1))
                y2 = int(np.clip(y1 + length*np.sin(angle), 0, h-1))
                width = np.random.randint(SCRATCH_WIDTH[0], SCRATCH_WIDTH[1]+1)
                draw.line([(x1, y1), (x2, y2)], fill=255, width=width)
            scratch_tensor = torch.from_numpy(np.array(scratch_img)).float().unsqueeze(0) / 255.0
            mask = (scratch_tensor > 0.5).float()
        return mask

    def __getitem__(self, idx):
        attempts = 0
        while attempts < 5:
            img_path = self.image_paths[idx]
            try:
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                h, w = img.shape[1], img.shape[2]
                mask = self._generate_mask(h, w)
                return img, mask
            except Exception:
                idx = (idx + 1) % len(self.image_paths)
                attempts += 1
        raise RuntimeError("Failed to load image after multiple attempts")
# 参数来源提示：此脚本同时从 dataset.py 引入 LR、LAMBDA_PERCEP，
# 与 config.py 的 LEARNING_RATE、LAMBDA_PERCEPTUAL 存在重复与潜在不一致；
# 建议在实际训练中统一以 config.py 为准，避免多处修改导致的参数漂移。


def main():
    # 设备
    device = torch.device(DEVICE)
    # 混合精度说明：config.USE_MIXED_PRECISION 已提供，但当前未启用 torch.cuda.amp/GradScaler；
    # 如需节省显存与加速，可在训练/验证阶段引入 autocast 与 GradScaler。

    # -------------------------- 训练集数据增强（更强，提升泛化能力） --------------------------
    train_transform = transforms.Compose([
        # 1. 先Resize到比目标尺寸稍大的尺寸（为后续随机裁剪留空间）
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),  # 例如：256+32=288×288
        # 2. 随机裁剪到目标尺寸（增加随机性，避免图像边缘固定）
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        # 3. 随机水平翻转（50%概率，适用于无方向偏好的图像，如人脸、风景）
        transforms.RandomHorizontalFlip(p=0.5),
        # 4. 随机旋转（-10°到10°之间，注意：参数必须是元组！）
        transforms.RandomRotation(degrees=(-10, 10)),  # 修正：用元组表示角度范围
        # 5. 随机亮度/对比度调整（针对老照片等色彩不稳定的场景）
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomGrayscale(p=0.1),
        # 6. 转换为Tensor（[0,1]范围）
        transforms.ToTensor(),
        # 7. 归一化到[-1,1]（适配GAN的Tanh输出）
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # -------------------------- 验证集/测试集预处理（无增强，保证评估一致性） --------------------------
    val_test_transform = transforms.Compose([
        # 直接Resize到目标尺寸（无随机操作，确保结果可复现）
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # 加载数据集时分别使用对应的transform
    # 掩码类型：MASK_TYPE 可选 ["random_rect", "center"]；
    # - random_rect：生成若干随机矩形缺失，更贴近期望的随机遮挡场景；
    # - center：在图像中心生成方形掩码，适合特定分布或验证一致性。
    
    # 首先加载完整的训练数据集（用于训练，带增强）
    full_train_dataset = ImageInpaintingDataset(
        data_dir=DATA_DIR_TRAIN,
        transform=train_transform,
        mask_type=MASK_TYPE,
        mask_dir="../data/mask",
        use_external_mask=True,
        external_mask_is_valid_region=False,
        external_mask_dilate=0,
        external_mask_mode="random"
    )

    # 同一训练目录的评估视图（用于val/test评估，无增强，掩码选择可复现）
    full_train_eval_dataset = ImageInpaintingDataset(
        data_dir=DATA_DIR_TRAIN,
        transform=val_test_transform,
        mask_type=MASK_TYPE,
        mask_dir="../data/mask",
        use_external_mask=True,
        external_mask_is_valid_region=False,
        external_mask_dilate=0,
        external_mask_mode="cycle"
    )

    # 划分比例：训练70%，验证10%，测试20%（均取自训练集）
    total = len(full_train_dataset)
    test_size = int(total * 0.20)
    val_size = int(total * 0.10)
    train_size = max(0, total - test_size - val_size)

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(total - test_size, total))

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_eval_dataset, val_indices)
    test_dataset = Subset(full_train_eval_dataset, test_indices)
    def visualize_samples(dataset, out_dir, prefix, count=3):
        os.makedirs(out_dir, exist_ok=True)
        n = min(count, len(dataset))
        for i in range(n):
            img, mask = dataset[i]
            img_vis = img * 0.5 + 0.5
            masked = img * (1 - mask)
            masked_vis = masked * 0.5 + 0.5
            save_image(img_vis, os.path.join(out_dir, f"{prefix}_{i}_img.png"))
            save_image(mask, os.path.join(out_dir, f"{prefix}_{i}_mask.png"))
            save_image(masked_vis, os.path.join(out_dir, f"{prefix}_{i}_masked.png"))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preview_dir = os.path.join(OUTPUT_DIR, "preview")
    visualize_samples(train_dataset, preview_dir, "train", count=3)
    visualize_samples(val_dataset, preview_dir, "val", count=2)
    visualize_samples(test_dataset, preview_dir, "test", count=2)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    # DataLoader 提示：Windows 平台上 num_workers 过大可能导致启动缓慢或卡住；
    # 如遇问题可将训练/验证的 num_workers 降为 0 或 1 进行排查。

    # 初始化模型、损失、优化器
    gen = UNetGenerator().to(device)
    dis = PatchDiscriminator().to(device)
    perceptual_loss = PerceptualLoss().to(device)
    l1_masked_loss = L1MaskedLoss().to(device)
    gan_loss = GANLoss(loss_type="mse").to(device)
    # 感知损失说明：PerceptualLoss 依赖 torchvision 的 VGG19 预训练权重；
    # 首次运行会自动下载，离线/受限网络环境需预置权重或临时禁用该项。

    opt_gen = Adam(gen.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    opt_dis = Adam(dis.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    scaler_gen = GradScaler(device='cuda', enabled=USE_MIXED_PRECISION)
    scaler_dis = GradScaler(device='cuda', enabled=USE_MIXED_PRECISION)

    # 记录最佳验证损失（用于保存最佳模型）
    best_val_loss = float('inf')
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

    # 训练循环
    for epoch in range(EPOCHS):
        # -------------------------- 训练阶段 --------------------------
        gen.train()
        dis.train()
        train_gen_loss_total = 0.0
        train_dis_loss_total = 0.0

        for real_imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Train)"):
            real_imgs = real_imgs.to(device)
            masks = masks.to(device)

            # 训练判别器
            opt_dis.zero_grad()
            with autocast(device_type='cuda', enabled=USE_MIXED_PRECISION):
                fake_imgs = gen(real_imgs, masks)
                real_pred = dis(real_imgs)
                fake_pred = dis(fake_imgs.detach())
                loss_dis_real = gan_loss(real_pred, target_is_real=True)
                loss_dis_fake = gan_loss(fake_pred, target_is_real=False)
                loss_dis = (loss_dis_real + loss_dis_fake) / 2
            scaler_dis.scale(loss_dis).backward()
            scaler_dis.step(opt_dis)
            scaler_dis.update()

            # 训练生成器
            opt_gen.zero_grad()
            with autocast(device_type='cuda', enabled=USE_MIXED_PRECISION):
                fake_imgs = gen(real_imgs, masks)
                fake_pred = dis(fake_imgs)
                loss_gan = gan_loss(fake_pred, target_is_real=True)
                loss_l1 = l1_masked_loss(fake_imgs, real_imgs, masks)
                loss_percep = perceptual_loss(fake_imgs, real_imgs)
                loss_gen = LAMBDA_GAN * loss_gan + LAMBDA_L1 * loss_l1 + LAMBDA_PERCEPTUAL * loss_percep
            scaler_gen.scale(loss_gen).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()

            # 累计损失
            train_gen_loss_total += loss_gen.item()
            train_dis_loss_total += loss_dis.item()

        # 计算训练集平均损失
        avg_train_gen_loss = train_gen_loss_total / len(train_loader)
        avg_train_dis_loss = train_dis_loss_total / len(train_loader)
        print(f"\nEpoch {epoch+1} Train: Gen Loss={avg_train_gen_loss:.4f}, Dis Loss={avg_train_dis_loss:.4f}")

        # -------------------------- 验证阶段 --------------------------
        gen.eval()
        dis.eval()
        val_gen_loss_total = 0.0
        with torch.no_grad():  # 关闭梯度计算，节省显存
            for real_imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Val)"):
                real_imgs = real_imgs.to(device)
                masks = masks.to(device)
                with autocast(device_type='cuda', enabled=USE_MIXED_PRECISION):
                    fake_imgs = gen(real_imgs, masks)

                # 仅计算生成器损失（验证阶段重点关注生成效果）
                # 验证指标说明：此处综合 GAN + 掩码 L1 + 感知损失，不计判别器损失；
                # 可根据需求扩展 PSNR/SSIM/LPIPS 等指标用于更全面的客观评估与早停策略。
                fake_pred = dis(fake_imgs)
                loss_gan = gan_loss(fake_pred, target_is_real=True)
                loss_l1 = l1_masked_loss(fake_imgs, real_imgs, masks)
                loss_percep = perceptual_loss(fake_imgs, real_imgs)
                loss_gen = LAMBDA_GAN * loss_gan + LAMBDA_L1 * loss_l1 + LAMBDA_PERCEPTUAL * loss_percep
                val_gen_loss_total += loss_gen.item()

        val_batches = len(val_loader)
        if val_batches == 0:
            print(f"Epoch {epoch+1} Val: skipped (empty)")
        else:
            avg_val_gen_loss = val_gen_loss_total / val_batches
            print(f"Epoch {epoch+1} Val: Gen Loss={avg_val_gen_loss:.4f}")
            if avg_val_gen_loss < best_val_loss:
                best_val_loss = avg_val_gen_loss
                torch.save(gen.state_dict(), os.path.join(SAVE_MODEL_DIR, "gen_best.pth"))
                torch.save(dis.state_dict(), os.path.join(SAVE_MODEL_DIR, "dis_best.pth"))
                print(f"Best model saved (Val Loss: {best_val_loss:.4f})")

        # 每10个epoch额外保存一次
        # 保存频率说明：当前硬编码为 10；如需与 config.SAVE_FREQ 保持一致，可改为读取配置常量。
        # -------------------------- 测试评估 --------------------------
        test_gen_loss_total = 0.0
        with torch.no_grad():
            for real_imgs, masks in tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} (Test)"):
                real_imgs = real_imgs.to(device)
                masks = masks.to(device)
                with autocast(device_type='cuda', enabled=USE_MIXED_PRECISION):
                    fake_imgs = gen(real_imgs, masks)
                fake_pred = dis(fake_imgs)
                loss_gan = gan_loss(fake_pred, target_is_real=True)
                loss_l1 = l1_masked_loss(fake_imgs, real_imgs, masks)
                loss_percep = perceptual_loss(fake_imgs, real_imgs)
                loss_gen = LAMBDA_GAN * loss_gan + LAMBDA_L1 * loss_l1 + LAMBDA_PERCEPTUAL * loss_percep
                test_gen_loss_total += loss_gen.item()

        test_batches = len(test_loader)
        if test_batches == 0:
            print(f"Epoch {epoch+1} Test: skipped (empty)")
        else:
            avg_test_gen_loss = test_gen_loss_total / test_batches
            print(f"Epoch {epoch+1} Test: Gen Loss={avg_test_gen_loss:.4f}")

        if (epoch + 1) % SAVE_FREQ == 0:
            torch.save(gen.state_dict(), os.path.join(SAVE_MODEL_DIR, f"gen_epoch_{epoch+1}.pth"))
            torch.save(dis.state_dict(), os.path.join(SAVE_MODEL_DIR, f"dis_epoch_{epoch+1}.pth"))


if __name__ == "__main__":
    main()