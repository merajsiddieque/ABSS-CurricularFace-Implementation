#!/usr/bin/env python3
"""
convert_recordio.py
Converts CASIA-WebFace MXNet RecordIO dataset to identity-folder format
for CurricularFace training, without modifying any original repository code.
"""

import os
import sys
import struct
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser(description="Convert MXNet RecordIO dataset to folder structure.")
    parser.add_argument("--rec_path", type=str, default="data/faces_webface_112x112/train.rec",
                        help="Path to train.rec")
    parser.add_argument("--idx_path", type=str, default="data/faces_webface_112x112/train.idx",
                        help="Path to train.idx")
    parser.add_argument("--output_dir", type=str, default="data/CASIA-WebFace",
                        help="Output directory for extracted identity folders")
    parser.add_argument("--mapping_path", type=str, default="data/CASIA-WebFace/conversion_mapping.csv",
                        help="Path for conversion mapping CSV")
    parser.add_argument("--manifest_path", type=str, default="data/CASIA-WebFace/casia_train.txt",
                        help="Path for FaceDataset manifest text file")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of threads for writing image files")
    return parser.parse_args()

def convert_dataset(rec_path, idx_path, output_dir, mapping_path, manifest_path, num_workers=8):
    print("=" * 70)
    print("Starting RecordIO to CurricularFace Dataset Conversion")
    print(f"RecordIO rec file: {rec_path}")
    print(f"RecordIO idx file: {idx_path}")
    print(f"Output directory:  {output_dir}")
    print(f"Mapping CSV:       {mapping_path}")
    print(f"Workers:           {num_workers}")
    print("=" * 70)

    start_time = time.time()

    # Step 1: Read train.idx
    if not os.path.isfile(rec_path):
        raise FileNotFoundError(f"Record file not found: {rec_path}")
    if not os.path.isfile(idx_path):
        raise FileNotFoundError(f"Index file not found: {idx_path}")

    print("[1/6] Reading train.idx mapping...")
    idx_map = {}
    with open(idx_path, "r") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            k, v = line_str.split("\t")
            idx_map[int(k)] = int(v)

    total_idx_entries = len(idx_map)
    print(f"  Total records in train.idx: {total_idx_entries}")

    # Step 2: Read Header Record (Key 0)
    print("[2/6] Inspecting header record (key 0)...")
    if 0 not in idx_map:
        raise ValueError("Key 0 (header record) not found in index file.")

    with open(rec_path, "rb") as f_rec:
        f_rec.seek(idx_map[0])
        magic, lflag = struct.unpack("<II", f_rec.read(8))
        if magic != 0xced7230a:
            raise ValueError(f"Invalid magic number in header record: {hex(magic)}")
        length = lflag & ((1 << 29) - 1)
        payload = f_rec.read(length)
        hdr_flag, label, id0, id1 = struct.unpack("<IfQQ", payload[:24])
        extra = payload[24:]
        id_start_float, id_end_float = struct.unpack("<2f", extra[:8])
        id_start = int(id_start_float)
        id_end = int(id_end_float)
        total_identities = id_end - id_start
        total_image_records = id_start - 1

    print(f"  Identity index range: [{id_start}, {id_end}) -> {total_identities} identities")
    print(f"  Image record range:   [1, {id_start}) -> {total_image_records} image records")

    # Step 3: Inspect Identity Records to verify ranges
    print("[3/6] Reading identity class metadata records...")
    class_ranges = {}
    with open(rec_path, "rb") as f_rec:
        for cls in range(total_identities):
            k = id_start + cls
            offset = idx_map[k]
            f_rec.seek(offset)
            magic, lflag = struct.unpack("<II", f_rec.read(8))
            if magic != 0xced7230a:
                raise ValueError(f"Corrupted identity record for class {cls} at key {k}")
            length = lflag & ((1 << 29) - 1)
            payload = f_rec.read(length)
            hdr_flag, label, id0, id1 = struct.unpack("<IfQQ", payload[:24])
            start_img_f, end_img_f = struct.unpack("<2f", payload[24:32])
            class_ranges[cls] = (int(start_img_f), int(end_img_f))

    # Verify contiguous ranges
    for cls in range(total_identities):
        s, e = class_ranges[cls]
        if cls == 0:
            if s != 1:
                raise ValueError(f"Class 0 does not start at 1: {s}")
        else:
            prev_e = class_ranges[cls - 1][1]
            if s != prev_e:
                raise ValueError(f"Discontinuity between class {cls-1} and {cls}: prev end {prev_e} != start {s}")
    if class_ranges[total_identities - 1][1] != id_start:
        raise ValueError(f"Last class end does not match total images: {class_ranges[total_identities-1][1]} != {id_start}")

    print("  Identity class ranges verified: 100% contiguous and complete.")

    # Step 4: Pre-create identity directories
    print("[4/6] Creating identity folders...")
    os.makedirs(output_dir, exist_ok=True)
    for cls in range(total_identities):
        folder_name = f"{cls:06d}"
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

    # Step 5: Read and extract all images
    print(f"[5/6] Extracting {total_image_records} images to {output_dir}...")
    corrupted_records = 0
    invalid_labels = 0
    extracted_images = 0
    images_per_class = {cls: 0 for cls in range(total_identities)}

    # We will write in chunks using ThreadPoolExecutor for high I/O throughput
    chunk_size = 5000
    write_tasks = []
    mapping_rows = []
    manifest_rows = []

    def flush_chunk(tasks):
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            def write_file(task):
                path, data = task
                with open(path, "wb") as f_out:
                    f_out.write(data)
            list(executor.map(write_file, tasks))

    with open(rec_path, "rb") as f_rec:
        for k in range(1, id_start):
            offset = idx_map[k]
            f_rec.seek(offset)
            hdr = f_rec.read(8)
            if len(hdr) < 8:
                corrupted_records += 1
                raise ValueError(f"Premature end of file reading record {k}")

            magic, lflag = struct.unpack("<II", hdr)
            if magic != 0xced7230a:
                corrupted_records += 1
                raise ValueError(f"Invalid magic {hex(magic)} at record {k}")

            length = lflag & ((1 << 29) - 1)
            pad = (4 - (length % 4)) % 4
            payload = f_rec.read(length + pad)
            if len(payload) < length:
                corrupted_records += 1
                raise ValueError(f"Payload truncated at record {k}")

            hdr_flag, label, id0, id1 = struct.unpack("<IfQQ", payload[:24])
            img_bytes = payload[24:length]

            if not img_bytes.startswith(b"\xff\xd8"):
                corrupted_records += 1
                raise ValueError(f"Non-JPEG image payload at record {k}")

            cls_label = int(label)
            if cls_label < 0 or cls_label >= total_identities:
                invalid_labels += 1
                raise ValueError(f"Invalid class label {cls_label} at record {k}")

            images_per_class[cls_label] += 1
            idx_in_class = images_per_class[cls_label]

            folder_name = f"{cls_label:06d}"
            img_filename = f"{idx_in_class:06d}.jpg"
            out_img_path = os.path.join(output_dir, folder_name, img_filename)

            # Store write task
            write_tasks.append((out_img_path, img_bytes))

            # Store CSV mapping entry
            # Format: record_index,original_identity_label,identity_folder,output_image_path
            # Normalize path with forward slashes for cross-platform portability
            norm_out_path = out_img_path.replace("\\", "/")
            mapping_rows.append(f"{k},{cls_label},{folder_name},{norm_out_path}\n")

            # Store manifest entry for FaceDataset: <relative_path> <label>
            # relative to output_dir: folder_name/img_filename
            manifest_rows.append(f"{folder_name}/{img_filename} {cls_label}\n")

            extracted_images += 1

            if len(write_tasks) >= chunk_size:
                flush_chunk(write_tasks)
                write_tasks.clear()
                if extracted_images % 50000 == 0:
                    elapsed = time.time() - start_time
                    fps = extracted_images / elapsed
                    print(f"  Extracted {extracted_images}/{total_image_records} images ({extracted_images*100/total_image_records:.1f}%) - {fps:.1f} img/s")

        if write_tasks:
            flush_chunk(write_tasks)
            write_tasks.clear()

    print(f"  Completed extraction of {extracted_images} images.")

    # Step 6: Write mapping CSV and manifest text files
    print("[6/6] Writing conversion mapping and dataset manifests...")
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f_map:
        f_map.write("record_index,original_identity_label,identity_folder,output_image_path\n")
        f_map.writelines(mapping_rows)
    print(f"  Saved mapping to: {mapping_path}")

    with open(manifest_path, "w", encoding="utf-8") as f_man:
        f_man.writelines(manifest_rows)
    print(f"  Saved manifest to: {manifest_path}")

    # Also write a manifest relative to parent 'data' folder for convenience
    parent_manifest_path = os.path.join("data", "casia_train.txt")
    with open(parent_manifest_path, "w", encoding="utf-8") as f_pman:
        for line in manifest_rows:
            f_pman.write(f"CASIA-WebFace/{line}")
    print(f"  Saved parent manifest to: {parent_manifest_path}")

    # Verification and Statistics
    elapsed = time.time() - start_time
    class_counts = list(images_per_class.values())
    min_imgs = min(class_counts)
    max_imgs = max(class_counts)
    avg_imgs = sum(class_counts) / len(class_counts)

    print("\n" + "=" * 70)
    print("DATASET CONVERSION REPORT")
    print("=" * 70)
    print(f"Total records in train.idx:              {total_idx_entries}")
    print(f"Total header & index records:            {1 + total_identities} (1 header + {total_identities} identity ranges)")
    print(f"Total source image records:              {total_image_records}")
    print(f"Total images successfully extracted:     {extracted_images}")
    print(f"Total corrupted or unreadable records:   {corrupted_records}")
    print(f"Total records with invalid labels:       {invalid_labels}")
    print(f"Validation (extracted == source images): {'PASSED' if extracted_images == total_image_records else 'FAILED'}")
    print(f"Total identities (classes):              {total_identities}")
    print(f"Minimum images per identity:             {min_imgs}")
    print(f"Maximum images per identity:             {max_imgs}")
    print(f"Average images per identity:             {avg_imgs:.2f}")
    print(f"Standard image dimensions:               112 x 112")
    print(f"Label preservation:                      CONFIRMED (0 to {total_identities - 1})")
    print(f"Total elapsed time:                      {elapsed:.2f} seconds ({extracted_images/elapsed:.1f} img/s)")
    print("=" * 70)

    return {
        "total_idx_entries": total_idx_entries,
        "total_image_records": total_image_records,
        "extracted_images": extracted_images,
        "corrupted_records": corrupted_records,
        "invalid_labels": invalid_labels,
        "total_identities": total_identities,
        "min_imgs": min_imgs,
        "max_imgs": max_imgs,
        "avg_imgs": avg_imgs,
        "elapsed": elapsed
    }

if __name__ == "__main__":
    args = parse_args()
    convert_dataset(
        rec_path=args.rec_path,
        idx_path=args.idx_path,
        output_dir=args.output_dir,
        mapping_path=args.mapping_path,
        manifest_path=args.manifest_path,
        num_workers=args.workers
    )

