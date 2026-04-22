"""
merge_dataset.py
================
Merge the official OIA-ODIR distribution into a single unified dataset.

The official OIA-ODIR release is split into three subsets, each with its
own Excel annotation file and image folder:
    - Training Set
    - Off-site Test Set
    - On-site Test Set

This script consolidates them into:
    OIA-ODIR-Merged/
    ├── Images/                  (all 10,000 fundus images in one folder)
    ├── all_annotations.xlsx     (merged Excel annotations)
    └── all_annotations.csv      (CSV backup for compatibility)

The merged CSV retains a `dataset` column indicating the origin subset
of each patient (Training / Off-site / On-site) so provenance is preserved.

Usage:
    # Expects the original dataset at ./OIA-ODIR/ with the three subfolders
    python merge_dataset.py
    python merge_dataset.py CustomOutputDir/
"""

import os
import shutil
import pandas as pd


def merge_oia_odir_dataset(output_dir: str = "./OIA-ODIR-Merged",
                            copy_images: bool = True) -> pd.DataFrame:
    """
    Merge all OIA-ODIR data into a single folder with unified annotations.

    Args:
        output_dir: Output directory for merged data.
        copy_images: If True, copy images into a single folder;
                     if False, just consolidate annotations.

    Returns:
        pandas.DataFrame containing all merged annotations.
    """
    base_path = "./OIA-ODIR"
    dataset_subsets = [
        ("Off-site Test Set", "Off-site"),
        ("On-site Test Set",  "On-site"),
        ("Training Set",      "Training"),
    ]

    print("=" * 70)
    print("MERGING OIA-ODIR DATASET")
    print("=" * 70)

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    images_output = os.path.join(output_dir, "Images")
    os.makedirs(images_output, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Images will be saved to: {images_output}\n")

    all_annotations = []
    image_count = 0

    for subset_folder, subset_name in dataset_subsets:
        print(f"\n{'=' * 70}\nProcessing: {subset_name}\n{'=' * 70}")

        subset_path     = os.path.join(base_path, subset_folder)
        images_path     = os.path.join(subset_path, "Images")
        annotation_path = os.path.join(subset_path, "Annotation")

        if not os.path.exists(subset_path):
            print(f"Subset not found, skipping: {subset_path}")
            continue

        # Copy images
        if os.path.exists(images_path):
            images = [f for f in os.listdir(images_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Found {len(images)} images")

            if copy_images:
                for img in images:
                    src = os.path.join(images_path, img)
                    dst = os.path.join(images_output, img)

                    # If a file with the same name already exists,
                    # prefix with subset name to avoid collisions
                    if os.path.exists(dst):
                        dst = os.path.join(images_output, f"{subset_name}_{img}")

                    shutil.copy2(src, dst)
                    image_count += 1

        # Load English annotation file
        if os.path.exists(annotation_path):
            annotation_files = [f for f in os.listdir(annotation_path)
                                if f.endswith('.xlsx')]
            for anno_file in annotation_files:
                if "English" in anno_file:
                    anno_full_path = os.path.join(annotation_path, anno_file)
                    print(f"Loading annotations: {anno_file}")
                    try:
                        df = pd.read_excel(anno_full_path)
                        df['dataset'] = subset_name

                        # Standardize column name if needed
                        if 'ID' not in df.columns and 'id' in df.columns:
                            df.rename(columns={'id': 'ID'}, inplace=True)

                        all_annotations.append(df)
                        print(f"   Loaded {len(df)} records")
                    except Exception as e:
                        print(f"   Error loading {anno_file}: {e}")

    # Merge annotations
    print(f"\n{'=' * 70}\nMerging annotations...\n{'=' * 70}")

    if not all_annotations:
        print("No annotations found. Check that ./OIA-ODIR/ exists with the expected subfolders.")
        return None

    merged_df = pd.concat(all_annotations, ignore_index=True)

    output_xlsx = os.path.join(output_dir, "all_annotations.xlsx")
    output_csv  = os.path.join(output_dir, "all_annotations.csv")
    merged_df.to_excel(output_xlsx, index=False)
    merged_df.to_csv(output_csv, index=False)

    print(f"\nMerged {len(merged_df)} records from {len(all_annotations)} sources")
    print(f"Saved: {output_xlsx}")
    print(f"Saved: {output_csv}")

    # Summary
    print(f"\nDataset Summary:")
    print(f"   Total records: {len(merged_df)}")
    print(f"   Total images:  {image_count}")
    print(f"\n   By Source:")
    for subset_name in merged_df['dataset'].unique():
        count = len(merged_df[merged_df['dataset'] == subset_name])
        print(f"      - {subset_name}: {count} patients")

    print(f"\nColumns in merged file:")
    for i, col in enumerate(merged_df.columns, 1):
        print(f"      {i}. {col}")

    print(f"\n{'=' * 70}\nMERGE COMPLETE\n{'=' * 70}")
    return merged_df


if __name__ == "__main__":
    import sys

    output_dir = sys.argv[1] if len(sys.argv) > 1 else "OIA-ODIR-Merged"

    print("\nStarting OIA-ODIR merge...\n")

    merged_df = merge_oia_odir_dataset(output_dir=output_dir, copy_images=True)

    if merged_df is not None:
        print(f"\nDataset merge complete.")
