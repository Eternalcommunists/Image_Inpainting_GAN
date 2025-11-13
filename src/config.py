import torch

# -------------------------- 1. 超参数配置（模型训练核心参数） --------------------------
# 训练轮数（根据数据集大小调整，小数据集50-100，大数据集200-500）
EPOCHS = 100

# 批次大小（根据GPU显存调整：12GB显存建议8-16，8GB显存建议4-8）
BATCH_SIZE = 4

# 学习率（GAN训练常用2e-4，若不稳定可降为1e-4）
LEARNING_RATE = 2e-4

# 损失函数权重（平衡不同损失的影响）
LAMBDA_L1 = 100.0      # 掩码L1损失权重（确保修复区域与真实值接近）
LAMBDA_PERCEPTUAL = 10.0  # 感知损失权重（确保高层语义一致）
LAMBDA_GAN = 1.0       # GAN损失权重（确保修复结果真实）

# 图像尺寸（需与模型输入匹配，256×256适合多数场景，512×512需更大显存）
IMAGE_SIZE = 512

# 优化器参数（GAN常用Adam，beta1=0.5是稳定训练的关键）
ADAM_BETAS = (0.5, 0.999)


# -------------------------- 2. 路径配置（数据、模型、输出文件路径） --------------------------
# 数据集路径（根据实际项目结构调整，相对路径基于src/目录）
DATA_DIR_TRAIN = "../data/train/"    # 训练集图像目录
DATA_DIR_VAL = "../data/val/"        # 验证集图像目录
DATA_DIR_TEST = "../data/test/"      # 测试集图像目录

# 模型保存路径（训练过程中保存的权重文件）
SAVE_MODEL_DIR = "../saved_models/"

# 输出结果路径（修复后的图像保存目录）
OUTPUT_DIR = "../outputs/"

# 日志路径（可选，用于保存训练损失曲线等）
LOG_DIR = "../logs/"


# -------------------------- 3. 设备配置（自动选择GPU/CPU） --------------------------
# 优先使用GPU（cuda），若无则使用CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 若使用多GPU（可选），这里可指定GPU编号（如"cuda:0"、"cuda:1"）
# 单GPU无需修改，多GPU需配合torch.nn.DataParallel使用
# DEVICE = "cuda:0"


# -------------------------- 4. 掩码配置（数据集生成掩码的参数） --------------------------
# 训练时默认掩码类型："random_rect"（随机矩形遮挡）、"center"（中心遮挡）、"scratches"（划痕）
MASK_TYPE = "random_rect"

# 掩码覆盖区域比例（0.1-0.3较常用，即10%-30%区域被遮挡）
MASK_RATIO = 0.2

# 若为"scratches"掩码，可配置划痕数量和粗细
SCRATCH_NUM = 5  # 划痕数量
SCRATCH_WIDTH = (1, 3)  # 划痕粗细范围（像素）


# -------------------------- 5. 训练辅助配置 --------------------------
# 是否使用混合精度训练（节省显存，加速训练，需PyTorch>=1.6）
USE_MIXED_PRECISION = True

# 日志打印频率（每多少个batch打印一次损失）
PRINT_FREQ = 10

# 模型保存频率（除最佳模型外，每多少个epoch额外保存一次）
SAVE_FREQ = 10

# 验证频率（每多少个epoch在验证集上评估一次，建议1个epoch一次）
VAL_FREQ = 1