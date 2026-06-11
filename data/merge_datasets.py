import os
import argparse
import shutil
from pathlib import Path
from PIL import Image
 
 
# =============================================================================
# CLASS NAME MAPPING
#
# Keys   = folder names as they appear in ds2 (lowercased for matching)
# Values = folder names as they appear in ds1 (the target names)
#
# Classes in ds2 that are NOT in this dict are skipped entirely.
# Add or edit entries here if your actual folder names differ.
# =============================================================================
DS2_TO_DS1 = {
    "baseball":      "Baseball",
    "basketball":    "Basketball",
    "billiard ball": "Billiards",
    "bowling ball":  "Bowling",
    "cricket ball":  "Cricket",
    "golf ball":     "Golf",
    "rugby ball":    "Rugby",
    "soccer ball":   "Football",   # ds2 calls it "soccer ball", ds1 calls it "Football"
    "tennis ball":   "Tennis",
    "volleyball":    "Volleyball",
    # "football"    is American football in ds2 -- no equivalent in ds1, so omitted
    # "ping pong"   not in ds1
    # "hockey"      not in ds1
    # "lacrosse"    not in ds1
    # "baseball"    already covered
}
 
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
 
 
def copy_image(src: Path, dst: Path):
    """
    Copy an image to dst, converting to RGB JPEG along the way.
    This normalises palette images, RGBA PNGs, etc. into a consistent format.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(src).convert("RGB")
        img.save(dst.with_suffix(".jpg"), "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"  [skip] {src.name}: {e}")
        return False
 
 
def collect_ds1(ds1_root: Path, out_root: Path):
    """
    Copy all images from ds1 into the output folder.
    ds1 is already in the right structure: ds1_root/<ClassName>/image.jpg
    """
    print("\n--- Copying Dataset 1 ---")
    counts = {}
 
    for class_dir in sorted(ds1_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        n = 0
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            dst = out_root / class_name / img_path.name
            if copy_image(img_path, dst):
                n += 1
        counts[class_name] = n
        print(f"  {class_name:15s}: {n} images")
 
    return counts
 
 
def collect_ds2(ds2_root: Path, out_root: Path):
    """
    Copy matching images from ds2 into the output folder.
    ds2 has a train/ and test/ split; we merge both.
    Files are prefixed with 'ds2_' to avoid name collisions.
    """
    print("\n--- Copying Dataset 2 ---")
    counts = {}
    skipped_classes = set()
 
    for split in ["train", "test"]:
        split_dir = ds2_root / split
        if not split_dir.exists():
            # Some downloads have no split subfolder — try root directly
            split_dir = ds2_root
 
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
 
            ds2_class = class_dir.name.lower()
            ds1_class = DS2_TO_DS1.get(ds2_class)
 
            if ds1_class is None:
                skipped_classes.add(class_dir.name)
                continue
 
            n = 0
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                dst = out_root / ds1_class / f"ds2_{split}_{img_path.name}"
                if copy_image(img_path, dst):
                    n += 1
 
            counts[ds1_class] = counts.get(ds1_class, 0) + n
 
    for cls, n in sorted(counts.items()):
        print(f"  {cls:15s}: {n} images")
 
    if skipped_classes:
        print(f"\n  Skipped (no equivalent in ds1): {sorted(skipped_classes)}")
 
    return counts
 
 
def print_summary(ds1_counts, ds2_counts, out_root):
    print("\n" + "="*50)
    print("  MERGED DATASET SUMMARY")
    print("="*50)
    print(f"  {'Class':<15}  {'DS1':>6}  {'DS2':>6}  {'Total':>7}")
    print(f"  {'-'*15}  {'-'*6}  {'-'*6}  {'-'*7}")
 
    grand_total = 0
    all_classes = sorted(set(list(ds1_counts.keys()) + list(ds2_counts.keys())))
    for cls in all_classes:
        d1 = ds1_counts.get(cls, 0)
        d2 = ds2_counts.get(cls, 0)
        total = d1 + d2
        grand_total += total
        print(f"  {cls:<15}  {d1:>6}  {d2:>6}  {total:>7}")
 
    print(f"  {'-'*15}  {'-'*6}  {'-'*6}  {'-'*7}")
    print(f"  {'TOTAL':<15}  {'':>6}  {'':>6}  {grand_total:>7}")
    print(f"\n  Output saved to: {out_root.resolve()}")
    print("="*50)
 
 
def main(ds1_path, ds2_path, out_path):
    ds1_root = Path(ds1_path)
    ds2_root = Path(ds2_path)
    out_root = Path(out_path)
 
    if not ds1_root.exists():
        raise FileNotFoundError(f"DS1 not found: {ds1_root}")
    if not ds2_root.exists():
        raise FileNotFoundError(f"DS2 not found: {ds2_root}")
 
    if out_root.exists():
        print(f"Output folder {out_root} already exists. Remove it first to start fresh.")
        ans = input("Continue anyway and add to it? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return
 
    out_root.mkdir(parents=True, exist_ok=True)
 
    ds1_counts = collect_ds1(ds1_root, out_root)
    ds2_counts = collect_ds2(ds2_root, out_root)
    print_summary(ds1_counts, ds2_counts, out_root)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two sports ball datasets")
    parser.add_argument("--ds1", required=True,
                        help="Path to mdkabinhasan dataset root")
    parser.add_argument("--ds2", required=True,
                        help="Path to samuelcortinhas dataset root")
    parser.add_argument("--out", required=True,
                        help="Output path for the merged dataset")
    args = parser.parse_args()
 
    main(args.ds1, args.ds2, args.out)

