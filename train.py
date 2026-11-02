
import os
import sys
import time
import random
import argparse
import tempfile
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Sampler
from tensorboardX import SummaryWriter
from tqdm import tqdm

from config import configurations
from backbone.model_irse import IR_101
from head.metrics import CurricularFace
from dataset.datasets import FaceDataset


def separate_irse_bn_paras(modules):
    """Original CurricularFace BN parameter separation logic."""
    if not isinstance(modules, list):
        modules = [*modules.modules()]

    paras_only_bn = []
    paras_wo_bn = []

    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue

        if 'batchnorm' in str(layer.__class__):
            paras_only_bn.extend(list(layer.parameters()))
        else:
            paras_wo_bn.extend(list(layer.parameters()))

    return paras_only_bn, paras_wo_bn


def adjust_learning_rate(optimizer, epoch, cfg):
    lr = cfg["LR"]

    for milestone in cfg["STAGES"]:
        if epoch >= milestone:
            lr *= 0.1

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


class ResumableRandomSampler(Sampler):
    """Random permutation sampler that can resume at a batch offset."""

    def __init__(self, data_source):
        self.data_source = data_source
        self.permutation = None
        self.start_index = 0

    def set_epoch_state(self, permutation=None, start_batch=0,
                        batch_size=1):
        if permutation is None:
            self.permutation = torch.randperm(
                len(self.data_source)
            ).tolist()
            self.start_index = 0
        else:
            self.permutation = permutation
            self.start_index = start_batch * batch_size

    def __iter__(self):
        if self.permutation is None:
            self.set_epoch_state()

        return iter(self.permutation[self.start_index:])

    def __len__(self):
        if self.permutation is None:
            return len(self.data_source)

        return len(self.permutation) - self.start_index


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, backbone, head, optimizer,
                    epoch, next_batch, permutation, global_step):
    checkpoint = {
        "epoch": epoch,
        "next_batch": next_batch,
        "permutation": permutation,
        "global_step": global_step,
        "backbone": backbone.state_dict(),
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Write a temporary file before replacing the previous checkpoint.
    temp_path = path + ".tmp"

    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)


def load_checkpoint(path, backbone, head, optimizer, device):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    backbone.load_state_dict(checkpoint["backbone"])
    head.load_state_dict(checkpoint["head"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    restore_rng_state(checkpoint["rng_state"])

    print(f"Resumed checkpoint: {path}")
    print(f"Epoch: {checkpoint['epoch'] + 1}")
    print(f"Next batch: {checkpoint['next_batch'] + 1}")
    print(f"Global step: {checkpoint['global_step']}")

    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--model-root", type=str, default=None)
    parser.add_argument("--log-root", type=str, default=None)
    args = parser.parse_args()

    cfg = configurations[1].copy()

    if args.resume is not None:
        cfg["RESUME_PATH"] = args.resume
    if args.checkpoint_every_steps is not None:
        cfg["SAVE_EVERY_STEPS"] = args.checkpoint_every_steps
    if args.model_root is not None:
        cfg["MODEL_ROOT"] = args.model_root
    if args.log_root is not None:
        cfg["LOG_ROOT"] = args.log_root

    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Reproducibility
    random.seed(cfg["SEED"])
    np.random.seed(cfg["SEED"])
    torch.manual_seed(cfg["SEED"])
    torch.cuda.manual_seed_all(cfg["SEED"])

    torch.backends.cudnn.benchmark = True

    os.makedirs(cfg["MODEL_ROOT"], exist_ok=True)
    os.makedirs(cfg["LOG_ROOT"], exist_ok=True)

    checkpoint_path = os.path.join(
        cfg["MODEL_ROOT"], "latest_checkpoint.pth"
    )

    print("=" * 65)
    print("CURRICULARFACE SINGLE-GPU TRAINING")
    print("=" * 65)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Batch size:", cfg["BATCH_SIZE"])
    print("Epochs:", cfg["NUM_EPOCH"])
    print("Learning rate:", cfg["LR"])
    print("Milestones:", cfg["STAGES"])

    # Dataset
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg["RGB_MEAN"],
            std=cfg["RGB_STD"]
        ),
    ])

    dataset = FaceDataset(
        cfg["DATA_ROOT"],
        cfg["RECORD_DIR"],
        train_transform
    )

    num_classes = len(dataset.classes)

    if num_classes != 10572:
        raise ValueError(
            f"Expected 10572 classes, found {num_classes}"
        )

    sampler = ResumableRandomSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=cfg["BATCH_SIZE"],
        sampler=sampler,
        num_workers=cfg["NUM_WORKERS"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg["NUM_WORKERS"] > 0,
    )

    print("Training images:", len(dataset))
    print("Training identities:", num_classes)
    print("Batches per epoch:", len(dataset) // cfg["BATCH_SIZE"])

    # Model
    backbone = IR_101(cfg["INPUT_SIZE"]).to(device)
    head = CurricularFace(
        in_features=cfg["EMBEDDING_SIZE"],
        out_features=num_classes
    ).to(device)

    # Optimizer: preserve original BN weight-decay separation.
    bn_params, non_bn_params = separate_irse_bn_paras(backbone)

    optimizer = optim.SGD(
        [
            {
                "params": non_bn_params + list(head.parameters()),
                "weight_decay": cfg["WEIGHT_DECAY"],
            },
            {
                "params": bn_params,
                "weight_decay": 0,
            },
        ],
        lr=cfg["LR"],
        momentum=cfg["MOMENTUM"],
    )

    criterion = nn.CrossEntropyLoss()

    # Resume state
    start_epoch = 0
    start_batch = 0
    global_step = 0
    saved_permutation = None

    if cfg["RESUME_PATH"]:
        if not os.path.isfile(cfg["RESUME_PATH"]):
            raise FileNotFoundError(cfg["RESUME_PATH"])

        state = load_checkpoint(
            cfg["RESUME_PATH"],
            backbone,
            head,
            optimizer,
            device
        )

        start_epoch = state["epoch"]
        start_batch = state["next_batch"]
        global_step = state["global_step"]
        saved_permutation = state["permutation"]

    writer = SummaryWriter(cfg["LOG_ROOT"])

    try:
        for epoch in range(start_epoch, cfg["NUM_EPOCH"]):

            current_lr = adjust_learning_rate(
                optimizer, epoch, cfg
            )

            if epoch == start_epoch and saved_permutation is not None:
                sampler.set_epoch_state(
                    saved_permutation,
                    start_batch,
                    cfg["BATCH_SIZE"]
                )
                batch_offset = start_batch
            else:
                sampler.set_epoch_state()
                batch_offset = 0

            backbone.train()
            head.train()

            running_loss = 0.0
            running_correct = 0
            processed = 0

            progress = tqdm(
                loader,
                total=len(loader),
                initial=batch_offset,
                desc=f"Epoch {epoch + 1}/{cfg['NUM_EPOCH']}"
            )

            for batch_idx, (images, labels) in enumerate(progress):
                actual_batch = batch_offset + batch_idx

                images = images.to(
                    device, non_blocking=True
                )
                labels = labels.to(
                    device, non_blocking=True
                )

                optimizer.zero_grad(set_to_none=True)

                embeddings, _ = backbone(images)
                logits, original_logits = head(
                    embeddings, labels
                )

                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    predictions = original_logits.argmax(dim=1)
                    correct = (predictions == labels).sum().item()

                batch_count = images.size(0)
                running_loss += loss.item() * batch_count
                running_correct += correct
                processed += batch_count

                global_step += 1

                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{current_lr:.5g}"
                )

                if global_step % 100 == 0:
                    writer.add_scalar(
                        "Training/Batch_Loss",
                        loss.item(),
                        global_step
                    )

                # Periodic resumable checkpoint.
                if (
                    global_step % cfg["SAVE_EVERY_STEPS"] == 0
                    and actual_batch + 1 < len(loader)
                ):
                    save_checkpoint(
                        checkpoint_path,
                        backbone,
                        head,
                        optimizer,
                        epoch,
                        actual_batch + 1,
                        sampler.permutation,
                        global_step
                    )

                if args.max_steps is not None and global_step >= args.max_steps:
                    print(
                        f"Controlled stop after {global_step} steps. "
                        f"Checkpoint: {checkpoint_path}"
                    )
                    return

            epoch_loss = running_loss / max(processed, 1)
            epoch_accuracy = 100.0 * running_correct / max(processed, 1)

            print(
                f"\nEpoch {epoch + 1} complete | "
                f"Loss: {epoch_loss:.4f} | "
                f"Top-1: {epoch_accuracy:.2f}% | "
                f"LR: {current_lr:.6g}"
            )

            writer.add_scalar(
                "Training/Epoch_Loss", epoch_loss, epoch + 1
            )
            writer.add_scalar(
                "Training/Epoch_Top1", epoch_accuracy, epoch + 1
            )
            writer.add_scalar(
                "Training/Learning_Rate", current_lr, epoch + 1
            )

            # Mark the epoch as completed.
            save_checkpoint(
                checkpoint_path,
                backbone,
                head,
                optimizer,
                epoch + 1,
                0,
                None,
                global_step
            )

            print("Checkpoint saved:", checkpoint_path)

            # Save a separate epoch snapshot.
            epoch_path = os.path.join(
                cfg["MODEL_ROOT"],
                f"checkpoint_epoch_{epoch + 1:03d}.pth"
            )

            save_checkpoint(
                epoch_path,
                backbone,
                head,
                optimizer,
                epoch + 1,
                0,
                None,
                global_step
            )

            saved_permutation = None
            start_batch = 0

        print("\nTraining completed successfully.")

    finally:
        writer.close()


if __name__ == "__main__":
    main()
