import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


# -------------------------- 1. 生成器：U-Net --------------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_filters=64):
        super().__init__()
        # 编码器（下采样）
        self.down1 = self._down_block(in_channels, num_filters)  # 256→128
        self.down2 = self._down_block(num_filters, num_filters * 2)  # 128→64
        self.down3 = self._down_block(num_filters * 2, num_filters * 4)  # 64→32
        self.down4 = self._down_block(num_filters * 4, num_filters * 8)  # 32→16

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_filters * 8, num_filters * 16, 4, 2, 1),
            nn.BatchNorm2d(num_filters * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 16→8

        # 解码器（上采样）
        self.up1 = self._up_block(num_filters * 16, num_filters * 8)  # 8→16
        self.up2 = self._up_block(num_filters * 8 * 2, num_filters * 4)  # 16→32（拼接编码器特征）
        self.up3 = self._up_block(num_filters * 4 * 2, num_filters * 2)  # 32→64
        self.up4 = self._up_block(num_filters * 2 * 2, num_filters)  # 64→128

        # 输出层（3通道RGB）
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 2, out_channels, 4, 2, 1),
            nn.Tanh()  # 输出像素值[-1,1]，与预处理匹配
        )  # 128→256

    # 下采样模块：Conv + BN + LeakyReLU
    def _down_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    # 上采样模块：ConvTranspose + BN + ReLU
    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, mask):
        # 掩码融合：仅保留原图有效区域，缺失区域设为0
        x_masked = x * (1 - mask)  # x: (N,3,256,256), mask: (N,1,256,256)

        # 编码器
        d1 = self.down1(x_masked)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        # 瓶颈
        b = self.bottleneck(d4)

        # 解码器（跳跃连接：拼接编码器对应层特征）
        u1 = self.up1(b)
        u1 = torch.cat([u1, d4], dim=1)  # 通道数翻倍
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)

        # 输出修复图像
        out = self.out_conv(u4)
        # 融合：修复区域用生成结果，有效区域保留原图
        out = x * (1 - mask) + out * mask
        return out


# -------------------------- 2. 判别器：PatchGAN --------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, num_filters=64):
        super().__init__()
        # 4层卷积，输出16×16的patch概率图
        self.model = nn.Sequential(
            # 输入：(N,3,256,256) → (N,64,128,128)
            nn.Conv2d(in_channels, num_filters, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # (N,64,128,128) → (N,128,64,64)
            nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (N,128,64,64) → (N,256,32,32)
            nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (N,256,32,32) → (N,512,16,16)
            nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出：(N,1,16,16)，每个像素对应1个patch的真实概率
            nn.Conv2d(num_filters * 8, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)
