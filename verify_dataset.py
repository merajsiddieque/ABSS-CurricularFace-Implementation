#!/usr/bin/env python3
"""
verify_dataset.py
Verifies that the converted CASIA-WebFace dataset loads correctly using the
original authors' dataset classes (FaceDataset and ImageDataset) without
modifying any repository source code.
"""

import sys
import torch
import torchvision.transforms as transforms
from dataset.datasets import FaceDataset, ImageDataset

def main():
    print("=" * 70)
    print("VERIFYING DATASET LOADING WITH AUTHORS' DATASET CLASSES")
    print("=" * 70)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 1. Test FaceDataset (as used in train.py)
    print("\n[1/2] Testing FaceDataset (used in train.py)...")
    data_root = "data/CASIA-WebFace"
    record_dir = "data/CASIA-WebFace/casia_train.txt"

    face_ds = FaceDataset(root_dir=data_root, record_dir=record_dir, transform=transform)
    num_samples = len(face_ds)
    num_classes = len(face_ds.classes)
    sample_nums = face_ds.get_sample_num_of_each_class()

    print(f"  - Total samples loaded:    {num_samples}")
    print(f"  - Total classes loaded:    {num_classes}")
    print(f"  - Min samples in a class:  {min(sample_nums)}")
    print(f"  - Max samples in a class:  {max(sample_nums)}")

    # Check sample item
    img_0, label_0 = face_ds[0]
    print(f"  - Sample 0 tensor shape:   {img_0.shape}")
    print(f"  - Sample 0 tensor dtype:   {img_0.dtype}")
    print(f"  - Sample 0 identity label: {label_0}")

    img_last, label_last = face_ds[-1]
    print(f"  - Sample -1 identity label:{label_last}")

    assert num_samples == 490623, f"Expected 490623 samples, got {num_samples}"
    assert num_classes == 10572, f"Expected 10572 classes, got {num_classes}"
    assert img_0.shape == torch.Size([3, 112, 112]), f"Unexpected tensor shape {img_0.shape}"

    # DataLoader test
    loader = torch.utils.data.DataLoader(face_ds, batch_size=16, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    print(f"  - Batch X tensor shape:    {batch_x.shape}")
    print(f"  - Batch Y tensor shape:    {batch_y.shape}")

    print("  -> FaceDataset verification: PASSED")

    # 2. Test ImageDataset (standard directory-based dataset in dataset/datasets.py)
    print("\n[2/2] Testing ImageDataset (folder-based loader)...")
    img_ds = ImageDataset(root_dir="data/CASIA-WebFace", transform=transform)
    assert len(img_ds) == 490623, f"Expected 490623 samples, got {len(img_ds)}"
    assert len(img_ds.classes) == 10572, f"Expected 10572 classes, got {len(img_ds.classes)}"
    img_first, lbl_first = img_ds[0]
    assert img_first.shape == torch.Size([3, 112, 112])
    print(f"  - Total samples loaded:    {len(img_ds)}")
    print(f"  - Total classes loaded:    {len(img_ds.classes)}")
    print(f"  - Sample 0 tensor shape:   {img_first.shape}")
    print(f"  - Sample 0 identity label: {lbl_first}")
    print("  -> ImageDataset verification: PASSED")

    print("\n" + "=" * 70)
    print("ALL DATASET VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()

