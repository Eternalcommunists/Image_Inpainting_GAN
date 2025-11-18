import os
from torchvision import transforms
from torchvision.utils import save_image
from dataset import ImageInpaintingDataset
from config import DATA_DIR_TRAIN, DATA_DIR_VAL, DATA_DIR_TEST, IMAGE_SIZE, OUTPUT_DIR, MASK_TYPE


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


def main():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    train_dataset = ImageInpaintingDataset(
        data_dir=DATA_DIR_TRAIN,
        transform=train_transform,
        mask_type=MASK_TYPE,
        mask_dir="../data/mask",
        use_external_mask=True,
        external_mask_is_valid_region=False,
        external_mask_dilate=0,
        external_mask_mode="random",
    )
    val_dataset = ImageInpaintingDataset(
        data_dir=DATA_DIR_VAL,
        transform=val_test_transform,
        mask_type=MASK_TYPE,
        mask_dir="../data/mask",
        use_external_mask=True,
        external_mask_is_valid_region=False,
        external_mask_dilate=0,
        external_mask_mode="cycle",
    )
    test_dataset = ImageInpaintingDataset(
        data_dir=DATA_DIR_TEST,
        transform=val_test_transform,
        mask_type=MASK_TYPE,
        mask_dir="../data/mask",
        use_external_mask=True,
        external_mask_is_valid_region=False,
        external_mask_dilate=0,
        external_mask_mode="cycle",
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preview_dir = os.path.join(OUTPUT_DIR, "preview")
    visualize_samples(train_dataset, preview_dir, "train", count=3)
    visualize_samples(val_dataset, preview_dir, "val", count=2)
    visualize_samples(test_dataset, preview_dir, "test", count=2)


if __name__ == "__main__":
    main()