import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class PerceptualLoss(nn.Module):
    """
    感知损失：基于预训练VGG19的特征层计算损失，模拟人类视觉对高层语义和纹理的感知
    输入：生成图像（generated）和真实图像（real），形状均为 (B, 3, H, W)
    输出：特征层面的L1损失值
    """

    def __init__(self, feature_layer=8):
        super().__init__()
        # 加载预训练VGG19，取前N层作为特征提取器（默认前8层，包含低-中层特征）
        # 前8层结构：conv1_1 → relu1_1 → conv1_2 → relu1_2 → maxpool1 → conv2_1 → relu2_1 → conv2_2
        self.vgg = vgg19(pretrained=True).features[:feature_layer].eval()
        # 冻结VGG参数，仅用于特征提取（不参与训练更新）
        for param in self.vgg.parameters():
            param.requires_grad = False
        # 特征归一化：VGG训练时使用的图像均值（RGB）
        self.normalize = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))  # 均值
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))  # 标准差

    def forward(self, generated, real):
        # 将输入图像（[-1,1]）转换为VGG期望的输入范围（[0,1]，并归一化）
        generated = (generated + 1) / 2  # 从[-1,1]映射到[0,1]
        real = (real + 1) / 2

        # 应用VGG的归一化
        generated = (generated - self.normalize) / self.std
        real = (real - self.normalize) / self.std

        # 提取特征
        gen_feat = self.vgg(generated)
        real_feat = self.vgg(real)

        # 计算特征层的L1损失
        return F.l1_loss(gen_feat, real_feat)


class L1MaskedLoss(nn.Module):
    """
    掩码L1损失：仅计算图像中缺失区域（掩码为1的区域）的L1损失，忽略有效区域（掩码为0）
    输入：
        generated：生成图像 (B, 3, H, W)
        real：真实图像 (B, 3, H, W)
        mask：掩码 (B, 1, H, W)，1表示缺失区域，0表示有效区域
    输出：缺失区域的平均L1损失
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="mean")

    def forward(self, generated, real, mask):
        # 将掩码从单通道扩展到3通道（与图像匹配）
        mask_3ch = mask.repeat(1, 3, 1, 1)  # (B,1,H,W) → (B,3,H,W)
        # 仅计算缺失区域的误差
        generated_masked = generated * mask_3ch
        real_masked = real * mask_3ch
        return self.l1(generated_masked, real_masked)


class GANLoss(nn.Module):
    """
    GAN损失：支持多种GAN损失类型（如MSE、BCE），适配判别器输出
    输入：
        pred：判别器的预测结果（real_pred或fake_pred），形状为 (B, 1, H_patch, W_patch)
        target_is_real：布尔值，True表示目标是“真实图像”（标签1），False表示“生成图像”（标签0）
    输出：GAN损失值
    """

    def __init__(self, loss_type="mse"):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == "mse":
            self.loss = nn.MSELoss(reduction="mean")
        elif loss_type == "bce":
            self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        else:
            raise ValueError(f"不支持的损失类型：{loss_type}，可选'mse'或'bce'")

    def forward(self, pred, target_is_real):
        # 生成目标标签（与pred形状一致）
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        # 计算损失（若为BCE，pred需是logits；若为MSE，pred是概率）
        return self.loss(pred, target)


class WGANGPLoss(nn.Module):
    """
    WGAN-GP损失：用于稳定GAN训练，避免模式崩溃（无sigmoid输出的判别器适用）
    参考：https://arxiv.org/abs/1704.00028
    """

    def __init__(self, lambda_gp=10.0):
        super().__init__()
        self.lambda_gp = lambda_gp  # 梯度惩罚系数

    def forward(self, discriminator, real, fake, mask=None):
        """
        计算WGAN-GP的判别器损失和生成器损失
        Args:
            discriminator：判别器模型
            real：真实图像 (B,3,H,W)
            fake：生成图像 (B,3,H,W)
            mask：可选，用于生成混合图像时的掩码（默认无）
        Returns:
            dis_loss：判别器损失（含梯度惩罚）
            gen_loss：生成器损失
        """
        # 判别器对真实图像和生成图像的输出
        real_pred = discriminator(real)
        fake_pred = discriminator(fake.detach())  # 生成器不更新

        # 原始WGAN损失（最大化real_pred，最小化fake_pred）
        dis_loss = -torch.mean(real_pred) + torch.mean(fake_pred)

        # 梯度惩罚（Gradient Penalty）
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)  # 随机插值系数
        # 生成真实图像和生成图像的混合样本
        if mask is not None:
            # 若有掩码，混合时仅在有效区域插值（修复任务常用）
            mixed = real * (1 - mask) + (alpha * real + (1 - alpha) * fake) * mask
        else:
            mixed = alpha * real + (1 - alpha) * fake
        mixed.requires_grad_(True)
        mixed_pred = discriminator(mixed)

        # 计算混合样本的梯度
        grad = torch.autograd.grad(
            outputs=mixed_pred,
            inputs=mixed,
            grad_outputs=torch.ones_like(mixed_pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        # 梯度惩罚：强制梯度范数接近1
        grad_flat = grad.view(grad.size(0), -1)
        grad_penalty = torch.mean((grad_flat.norm(2, dim=1) - 1) ** 2)
        dis_loss += self.lambda_gp * grad_penalty

        # 生成器损失：最大化判别器对生成图像的输出
        gen_loss = -torch.mean(discriminator(fake))

        return dis_loss, gen_loss